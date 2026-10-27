"""Initial-batch seed-data generation pipeline: RFdiffusion -> ProteinMPNN -> AF3.

Produces an ``initial_csv`` in the benchmark CSV schema consumed by
``binderopt.optimizer._load_initial_data`` and ``binderopt.targets.ensure_benchmark_csv``,
so a campaign can start from nothing rather than from a pool someone else made.

The loop itself does not need this module: SCALE is an optimiser placed *after* a
generator, and any generator will do. This is one convenient generator, not the required
one.

Pipeline, per target:

1. RFdiffusion backbone generation (subprocess into the RFdiffusion checkout named by
   ``SCALE_RFDIFFUSION_DIR``, ``cd``'d into, since RFdiffusion's hydra config resolution
   is relative-path sensitive).
2. ProteinMPNN sequence design per backbone (subprocess into ``SCALE_PROTEINMPNN_DIR``,
   mirroring the invocation convention in ``binderopt/mpnn_scorer.py`` -- each tool is run
   by the interpreter of its own conda environment, given by absolute path, because their
   dependencies conflict).
3. AF3 complex evaluation per (backbone, sequence) pair, via the existing
   ``binderopt.af3_interface.AlphaFold3Interface`` (no new AF3 subprocess
   code is written here).
4. Assembly into the benchmark CSV schema:
   sequence,target_sequence,target,seed,design_json,cif_file,sample_dir,nll,
   design_dir,af3_output_dir,iptm,ptm,ranking_score,iteration

Resumability: every step checks for pre-existing output on disk before
re-running (backbone PDBs, ProteinMPNN ``seqs/*.fa`` + ``scores/*.npz``
files, and AF3 output directories/summary jsons), mirroring the cache-reuse
pattern already used by ``AlphaFold3Interface._prepare_target_msa``. This
lets a single seeding job be split/interrupted/resumed across multiple
sbatch submissions.

Important ProteinMPNN flag correction vs. the old reference script
(``run_pos_rfd.sh``): that script passes ``--model_type "soluble"``, but
``protein_mpnn_run.py``'s actual argparse (verified by reading the source
directly) has no ``--model_type`` flag at all -- the real flag is
``--use_soluble_model`` (a boolean ``action="store_true"`` flag, not a
value-taking option). This module uses the correct flag.

Important ProteinMPNN *output-parsing* note: this module runs ProteinMPNN in
its default sequence-*generation* mode (no ``--score_only``), which is a
different code path from ``mpnn_scorer.py`` (which uses ``--score_only 1``).
In generation mode, ``--save_score 1`` writes per-sequence scores to
``<out_folder>/scores/<pdb_basename>.npz`` with a ``score`` array of shape
``(num_seq_per_target,)`` -- one entry per sequence, in the same order as
the sequences appear in ``<out_folder>/seqs/<pdb_basename>.fa`` (verified by
reading ``protein_mpnn_run.py``'s main sampling loop: ``score_list`` is
appended to in the same nested temperature/batch loop that writes each
``>T=...`` fasta record, then dumped via
``np.savez(score_file, score=np.array(score_list, ...))``). This is *not*
the ``score_only/<pdb_basename>_fasta_N.npz`` file ``mpnn_scorer.py`` parses
-- that file only exists when ``--score_only 1`` is passed, which we don't
do here.
"""
from __future__ import annotations

import csv
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .af3_interface import AlphaFold3Interface
from .external import tool_path

logger = logging.getLogger(__name__)

_BENCHMARK_FIELDNAMES = [
    "sequence", "target_sequence", "target", "seed", "design_json",
    "cif_file", "sample_dir", "nll", "design_dir", "af3_output_dir",
    "iptm", "ptm", "ranking_score", "iteration",
]

# Matches a ProteinMPNN-generated fasta record header, e.g.:
# >T=0.1, sample=1, score=1.2345, global_score=1.5678, seq_recovery=0.1234
_FASTA_HEADER_RE = re.compile(r"^>T=")


@dataclass
class SeedingConfig:
    """Parameters for one seed-data generation run, normally sourced from a
    target yaml's ``seeding:`` block (see configs/targets/her2.yaml)."""

    target_name: str
    target_pdb: str
    contig: str
    hotspot_residues: List[str]
    binder_length: int = 100
    n_backbones: int = 50
    n_sequences_per_backbone: int = 15
    sampling_temp: str = "0.1"

    # RFdiffusion diffuser/denoiser knobs (defaults match run_pos_rfd.sh)
    diffuser_T: int = 50
    noise_scale_ca: float = 0.5
    noise_scale_frame: float = 0.5

    # Resolved from the environment when not given explicitly; see binderopt/external.py.
    # A missing variable raises at construction, naming what to set, rather than handing a
    # subprocess a path that exists only on the machine this was written on.
    rfdiffusion_dir: str = field(
        default_factory=lambda: tool_path("SCALE_RFDIFFUSION_DIR"))
    rfdiffusion_python: str = field(
        default_factory=lambda: tool_path("SCALE_RFDIFFUSION_PYTHON"))
    mpnn_repo_dir: str = field(
        default_factory=lambda: tool_path("SCALE_PROTEINMPNN_DIR"))
    mpnn_python: str = field(
        default_factory=lambda: tool_path("SCALE_PROTEINMPNN_PYTHON"))
    mpnn_model_name: str = "v_48_020"

    @classmethod
    def from_yaml_block(cls, target_name: str, block: Dict) -> "SeedingConfig":
        """Build a SeedingConfig from a target yaml's parsed ``seeding:`` dict."""
        required = ["target_pdb", "contig", "hotspot_residues"]
        missing = [k for k in required if k not in block]
        if missing:
            raise ValueError(
                f"Target '{target_name}' seeding block is missing required key(s): {missing}"
            )
        kwargs = dict(block)
        kwargs.pop("target_name", None)
        return cls(target_name=target_name, **kwargs)


def build_rfdiffusion_command(config: SeedingConfig, output_prefix: Path, num_designs: int) -> List[str]:
    """Construct the RFdiffusion ``run_inference.py`` command line (no execution)."""
    hotspot_str = ",".join(config.hotspot_residues)
    return [
        config.rfdiffusion_python, "scripts/run_inference.py",
        f"inference.output_prefix={output_prefix}",
        f"inference.input_pdb={config.target_pdb}",
        f"inference.num_designs={num_designs}",
        f"contigmap.contigs=[{config.contig}]",
        f"ppi.hotspot_res=[{hotspot_str}]",
        f"diffuser.T={config.diffuser_T}",
        f"denoiser.noise_scale_ca={config.noise_scale_ca}",
        f"denoiser.noise_scale_frame={config.noise_scale_frame}",
    ]


def build_proteinmpnn_command(
    config: SeedingConfig, backbone_pdb: Path, out_folder: Path,
) -> List[str]:
    """Construct the ProteinMPNN ``protein_mpnn_run.py`` command line (no execution).

    Uses ``--use_soluble_model`` (a bare flag), not ``--model_type "soluble"``
    (not a real argparse option) -- see module docstring.
    """
    return [
        config.mpnn_python, "protein_mpnn_run.py",
        "--pdb_path", str(backbone_pdb),
        "--pdb_path_chains", "A",
        "--out_folder", str(out_folder),
        "--num_seq_per_target", str(config.n_sequences_per_backbone),
        "--sampling_temp", config.sampling_temp,
        "--batch_size", "1",
        "--use_soluble_model",
        "--save_score", "1",
        "--save_probs", "0",
        "--model_name", config.mpnn_model_name,
    ]


def run_rfdiffusion(config: SeedingConfig, work_dir: Path, dry_run: bool = False) -> List[Path]:
    """Generate backbones with RFdiffusion into ``work_dir/backbones``.

    Resumable: if ``work_dir/backbones`` already contains
    ``config.n_backbones`` matching ``<prefix>_*.pdb`` files, skips the
    subprocess call entirely and just returns the existing paths.
    """
    backbones_dir = work_dir / "backbones"
    backbones_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{config.target_name}_binder"
    output_prefix = backbones_dir / prefix

    existing = sorted(backbones_dir.glob(f"{prefix}_*.pdb"))
    if len(existing) >= config.n_backbones:
        logger.info(
            "Found %d existing backbones in %s (>= requested %d); skipping RFdiffusion.",
            len(existing), backbones_dir, config.n_backbones,
        )
        return existing[: config.n_backbones]

    cmd = build_rfdiffusion_command(config, output_prefix, config.n_backbones)
    logger.info("RFdiffusion command: %s", " ".join(cmd))
    if dry_run:
        return existing

    proc = subprocess.run(cmd, cwd=config.rfdiffusion_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"RFdiffusion failed (exit {proc.returncode}):\nstdout={proc.stdout[-4000:]}\n"
            f"stderr={proc.stderr[-4000:]}"
        )

    backbones = sorted(backbones_dir.glob(f"{prefix}_*.pdb"))
    logger.info("Generated %d backbone structures in %s", len(backbones), backbones_dir)
    return backbones


def _parse_mpnn_fasta(fasta_path: Path) -> List[str]:
    """Parse a ProteinMPNN generation-mode ``.fa`` file, returning generated sequences
    in file order, skipping the first record (the native/input sequence, written with a
    plain ``>name, score=...`` header lacking ``T=``).
    """
    sequences: List[str] = []
    header = None
    seq_lines: List[str] = []

    def _flush():
        if header is not None and _FASTA_HEADER_RE.match(header):
            sequences.append("".join(seq_lines))

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header = line
                seq_lines = []
            else:
                seq_lines.append(line.strip())
    _flush()
    return sequences


def run_proteinmpnn(
    config: SeedingConfig, backbone_pdb: Path, work_dir: Path, dry_run: bool = False,
) -> Tuple[List[str], List[float]]:
    """Run ProteinMPNN sequence design for one backbone; return (sequences, nll_scores).

    Resumable: if ``<out_folder>/seqs/<basename>.fa`` and
    ``<out_folder>/scores/<basename>.npz`` already exist, parses those
    instead of re-running.
    """
    basename = backbone_pdb.stem
    out_folder = work_dir / "sequences" / basename
    fasta_path = out_folder / "seqs" / f"{basename}.fa"
    score_path = out_folder / "scores" / f"{basename}.npz"

    if fasta_path.exists() and score_path.exists():
        logger.info("Found existing ProteinMPNN output for %s; skipping.", basename)
    else:
        cmd = build_proteinmpnn_command(config, backbone_pdb, out_folder)
        logger.info("ProteinMPNN command: %s", " ".join(cmd))
        if dry_run:
            return [], []

        out_folder.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, cwd=config.mpnn_repo_dir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ProteinMPNN failed for {basename} (exit {proc.returncode}):\n"
                f"stdout={proc.stdout[-4000:]}\nstderr={proc.stderr[-4000:]}"
            )

    if not fasta_path.exists():
        raise RuntimeError(f"ProteinMPNN did not produce expected fasta: {fasta_path}")
    sequences = _parse_mpnn_fasta(fasta_path)

    nll_scores: List[float] = [float("nan")] * len(sequences)
    if score_path.exists():
        data = np.load(score_path)
        scores = data["score"]
        if len(scores) == len(sequences):
            nll_scores = [float(s) for s in scores]
        else:
            logger.warning(
                "ProteinMPNN score count (%d) != sequence count (%d) for %s; "
                "leaving nll as NaN.", len(scores), len(sequences), basename,
            )
    else:
        logger.warning("Expected ProteinMPNN score file not found: %s", score_path)

    return sequences, nll_scores


@dataclass
class SeedRow:
    sequence: str
    target_sequence: str
    target: str
    backbone_name: str
    backbone_pdb: str
    nll: float
    af3_output_dir: str
    iptm: Optional[float]
    ptm: Optional[float]
    ranking_score: Optional[float]


class SeedDataGenerator:
    """Drives the full RFdiffusion -> ProteinMPNN -> AF3 seed-data pipeline for one target."""

    def __init__(
        self,
        config: SeedingConfig,
        target_sequence: str,
        af3_interface: Optional[AlphaFold3Interface],
        work_dir: str,
    ):
        self.config = config
        self.target_sequence = target_sequence
        self.af3 = af3_interface
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "af3_structures").mkdir(exist_ok=True)

    def generate_backbones(self, dry_run: bool = False) -> List[Path]:
        return run_rfdiffusion(self.config, self.work_dir, dry_run=dry_run)

    def design_sequences(self, backbone_pdb: Path, dry_run: bool = False) -> Tuple[List[str], List[float]]:
        return run_proteinmpnn(self.config, backbone_pdb, self.work_dir, dry_run=dry_run)

    def evaluate_with_af3(
        self, backbone_name: str, seq_idx: int, sequence: str,
    ) -> Tuple[str, Dict]:
        """Run (or reuse) AF3 prediction for one (backbone, sequence) pair.

        Resumable: checks for an existing summary_confidences.json under the
        expected af3_output_dir before invoking AlphaFold3Interface.predict
        (mirrors the cache-reuse pattern in
        AlphaFold3Interface._prepare_target_msa).
        """
        seq_id = f"{backbone_name}_seq{seq_idx}"
        structure_dir = self.work_dir / "af3_structures"
        dest_dir = structure_dir / seq_id

        existing_summary = list(dest_dir.glob("*_summary_confidences.json")) if dest_dir.exists() else []
        if existing_summary:
            logger.info("Found existing AF3 output for %s; skipping prediction.", seq_id)
            import json
            with open(existing_summary[0]) as f:
                summary = json.load(f)
            metrics = {
                "iptm": summary.get("iptm"),
                "ptm": summary.get("ptm"),
                "ranking_score": summary.get("ranking_score"),
            }
            return seq_id, metrics

        _confidence, metrics = self.af3.predict(
            sequence, save_structure=True, structure_dir=str(structure_dir), sequence_id=seq_id,
        )
        return seq_id, metrics

    def run(
        self,
        dry_run: bool = False,
        skip_af3: bool = False,
        output_csv: Optional[str] = None,
    ) -> int:
        """Run the pipeline, writing each completed row immediately to output_csv.

        Resumable: if output_csv already exists, any (backbone, seq_idx) pair
        whose seq_id appears in an existing row's af3_output_dir column is
        skipped -- AF3 was already done for it. RFdiffusion/ProteinMPNN steps
        are also skipped via on-disk file checks as before.

        Returns the total number of rows written this run (not counting rows
        already in the CSV from a previous run).
        """
        # --- load already-completed seq_ids for resume ---
        completed_seq_ids: set = set()
        if output_csv:
            csv_path = Path(output_csv)
            if csv_path.exists() and csv_path.stat().st_size > 0:
                with open(csv_path, newline="") as f:
                    for row in csv.DictReader(f):
                        sid = row.get("af3_output_dir", "")
                        if sid:
                            completed_seq_ids.add(sid)
                if completed_seq_ids:
                    logger.info(
                        "Resume: found %d already-completed AF3 evaluations in %s",
                        len(completed_seq_ids), output_csv,
                    )

        # --- open CSV for incremental append ---
        csv_file = None
        csv_writer = None
        if output_csv and not dry_run:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            csv_file = open(csv_path, "a", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=_BENCHMARK_FIELDNAMES)
            if write_header:
                csv_writer.writeheader()
                csv_file.flush()

        def _write_row(row: SeedRow) -> None:
            if csv_writer is None:
                return
            csv_writer.writerow({
                "sequence": row.sequence,
                "target_sequence": row.target_sequence,
                "target": row.target,
                "seed": 0,
                "design_json": "",
                "cif_file": "",
                "sample_dir": row.backbone_pdb,
                "nll": row.nll,
                "design_dir": row.backbone_name,
                "af3_output_dir": row.af3_output_dir,
                "iptm": row.iptm if row.iptm is not None else "",
                "ptm": row.ptm if row.ptm is not None else "",
                "ranking_score": row.ranking_score if row.ranking_score is not None else "",
                "iteration": 0,
            })
            csv_file.flush()  # write immediately, don't buffer

        backbones = self.generate_backbones(dry_run=dry_run)
        if dry_run:
            logger.info("[dry_run] Would generate %d backbones.", self.config.n_backbones)

        n_written = 0
        try:
            for backbone_pdb in backbones:
                backbone_name = backbone_pdb.stem
                sequences, nlls = self.design_sequences(backbone_pdb, dry_run=dry_run)
                if dry_run:
                    logger.info(
                        "[dry_run] Would design %d sequences for backbone %s.",
                        self.config.n_sequences_per_backbone, backbone_name,
                    )
                    continue

                for seq_idx, (sequence, nll) in enumerate(zip(sequences, nlls)):
                    seq_id = f"{backbone_name}_seq{seq_idx}"

                    if skip_af3:
                        row = SeedRow(
                            sequence=sequence,
                            target_sequence=self.target_sequence,
                            target=self.config.target_name,
                            backbone_name=backbone_name,
                            backbone_pdb=str(backbone_pdb),
                            nll=nll,
                            af3_output_dir="",
                            iptm=None, ptm=None, ranking_score=None,
                        )
                        _write_row(row)
                        n_written += 1
                        continue

                    if seq_id in completed_seq_ids:
                        logger.info("Resume: skipping already-done %s", seq_id)
                        continue

                    seq_id_out, metrics = self.evaluate_with_af3(backbone_name, seq_idx, sequence)
                    row = SeedRow(
                        sequence=sequence,
                        target_sequence=self.target_sequence,
                        target=self.config.target_name,
                        backbone_name=backbone_name,
                        backbone_pdb=str(backbone_pdb),
                        nll=nll,
                        af3_output_dir=seq_id_out,
                        iptm=metrics.get("iptm"),
                        ptm=metrics.get("ptm"),
                        ranking_score=metrics.get("ranking_score"),
                    )
                    _write_row(row)
                    n_written += 1
                    logger.info("Written %d rows so far (latest: %s)", n_written, seq_id_out)
        finally:
            if csv_file is not None:
                csv_file.close()

        return n_written


def write_benchmark_csv(rows: List[SeedRow], output_csv_path: str) -> str:
    """Write SeedRow records to the benchmark CSV schema used throughout binderopt."""
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BENCHMARK_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sequence": row.sequence,
                "target_sequence": row.target_sequence,
                "target": row.target,
                "seed": 0,
                "design_json": "",
                "cif_file": "",
                "sample_dir": row.backbone_pdb,
                "nll": row.nll,
                "design_dir": row.backbone_name,
                "af3_output_dir": row.af3_output_dir,
                "iptm": row.iptm if row.iptm is not None else "",
                "ptm": row.ptm if row.ptm is not None else "",
                "ranking_score": row.ranking_score if row.ranking_score is not None else "",
                "iteration": 0,
            })
    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return str(output_path)
