"""Unified external sequence injection module.

Merges two earlier injector modules into one. Supports three source types, each backed by a concrete
`CandidateSource` subclass (see that class's docstring for the adapter
contract other tools should implement to plug in a new source type):

- "proteinmpnn_batch" (`ProteinMPNNBatchSource`): recursively reads
  design_*.json files from a ProteinMPNN batch_output directory tree
  (residue_design dict -> sequence).
- "benchmark_batch" (`BenchmarkBatchSource`): reads a benchmark-format CSV
  (same schema as `initial_csv`, i.e. columns sequence/nll/...) and extracts
  sequences. This generalizes the old (effectively unused) "benchmark_batch"
  source type, which pointed at a directory of *reference* structures rather
  than a real sequence source -- here it is given a concrete, working
  implementation: point `path` at any benchmark-format CSV.
- "csv" (`CSVSource`): a plain CSV with a configurable sequence column.

Each loaded sequence also carries an `nll` value when the source provides
one (ProteinMPNN design JSON `nll` field, or a CSV `nll` column), else NaN.
This lets the optimizer's `nll` surrogate head learn from real
ProteinMPNN/RFdiffusion-sourced sequences as they get injected.
"""
from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


def _sequence_from_residue_design(residue_design: Dict[str, str]) -> str:
    """Convert a ProteinMPNN `residue_design` dict (e.g. {"P0_1": "MET", ...}) to a 1-letter sequence."""
    positions = sorted(residue_design.keys(), key=lambda x: int(x.split("_")[1]))
    return "".join(_AA_3TO1.get(residue_design[pos], "X") for pos in positions)


class CandidateSource(ABC):
    """Abstract adapter for upstream tools that produce candidate sequences.

    This is the extension point for plugging in candidate-sequence
    generators (ProteinMPNN, RFdiffusion, BoltzGen, a plain CSV of designs,
    etc). To add a new source type, subclass `CandidateSource`, implement
    `load()` to parse that tool's output format into a list of dicts (each
    must include a `sequence` key; an `nll` key and other metadata such as
    `cif_file` are optional), and reference your source's name via the
    `type` field in a target yaml's `injection.sources.<name>` block (wiring
    a new `type` string to your subclass is currently done in
    `SequenceInjector._SOURCE_CLASSES` below).
    """

    def __init__(self, source_cfg: Dict):
        self.source_cfg = source_cfg

    @abstractmethod
    def load(self) -> List[Dict]:
        """Load and return candidate records: dicts with at least a `sequence` key."""
        raise NotImplementedError


class ProteinMPNNBatchSource(CandidateSource):
    """Recursively loads sequences (+ nll if present) from ProteinMPNN design_*.json files."""

    def load(self) -> List[Dict]:
        batch_path = self.source_cfg.get("path") or self.source_cfg.get("csv_path")
        batch_dir = Path(batch_path)
        if not batch_dir.exists():
            logger.warning("proteinmpnn_batch directory not found: %s", batch_path)
            return []

        design_files = list(batch_dir.rglob("design_*.json"))
        logger.info("Found %d design JSON files in %s", len(design_files), batch_path)

        out: Dict[str, float] = {}
        for design_file in design_files:
            try:
                with open(design_file) as f:
                    data = json.load(f)
            except Exception:
                continue

            if "residue_design" not in data:
                continue
            seq = _sequence_from_residue_design(data["residue_design"])
            if not seq:
                continue
            nll = data.get("nll", np.nan)
            out[seq] = nll if nll is not None else np.nan

        logger.info("Extracted %d unique sequences from ProteinMPNN designs", len(out))
        return [{"sequence": seq, "nll": nll} for seq, nll in out.items()]


class BenchmarkBatchSource(CandidateSource):
    """Loads sequences (+ nll) from a benchmark-format CSV (same schema as initial_csv).

    This is also how RFdiffusion-origin candidates are consumed in this
    codebase: an RFdiffusion run's raw summary CSV gets converted to this
    benchmark schema (see `binderopt.targets.ensure_benchmark_csv`), and a
    `benchmark_batch` source then points at that converted CSV (see
    `configs/targets/bhrf1.yaml`).
    """

    def load(self) -> List[Dict]:
        csv_path = self.source_cfg.get("path") or self.source_cfg.get("csv_path")
        sequence_column = self.source_cfg.get("sequence_column", "sequence")

        if not csv_path or not Path(csv_path).exists():
            logger.warning("benchmark_batch CSV not found: %s", csv_path)
            return []

        out = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if sequence_column not in (reader.fieldnames or []):
                logger.warning("'%s' not in columns of %s", sequence_column, csv_path)
                return []
            for row in reader:
                seq = row.get(sequence_column, "").strip()
                if not seq:
                    continue
                nll_raw = row.get("nll", "")
                try:
                    nll = float(nll_raw) if nll_raw not in (None, "") else np.nan
                except ValueError:
                    nll = np.nan
                out.append({"sequence": seq, "nll": nll})
        logger.info("Loaded %d sequences from benchmark CSV %s", len(out), csv_path)
        return out


class CSVSource(CandidateSource):
    """Loads sequences (+ nll if a column named 'nll' exists) from a generic CSV.

    Currently identical in behavior to `BenchmarkBatchSource` (same schema
    assumptions); kept as a distinct, separately-named adapter so a plain
    "I just have a CSV of sequences" source reads clearly in target configs.
    """

    def load(self) -> List[Dict]:
        return BenchmarkBatchSource(self.source_cfg).load()


class SequenceInjector:
    """Loads candidate sequences (with optional NLL) from configured sources and doles them out over iterations."""

    _SOURCE_CLASSES: Dict[str, type] = {
        "proteinmpnn_batch": ProteinMPNNBatchSource,
        "benchmark_batch": BenchmarkBatchSource,
        "csv": CSVSource,
    }

    def __init__(self, injection_sources: Dict[str, Dict]):
        """
        Args:
            injection_sources: e.g. {"proteinmpnn": {"type": "proteinmpnn_batch", "path": "..."}}.
        """
        self.injection_sources = injection_sources
        self.injected_sequences: Set[str] = set()
        self._all_available: List[Tuple[str, float]] = None  # (sequence, nll) pairs

    def _initialize_available_sequences(self, already_evaluated: Set[str]) -> None:
        """Load all configured sources once, filtering out already-evaluated sequences."""
        if self._all_available is not None:
            return

        all_candidates: List[Tuple[str, float]] = []
        for source_name, source_cfg in self.injection_sources.items():
            source_type = source_cfg.get("type", "csv")
            source_cls = self._SOURCE_CLASSES.get(source_type)

            if source_cls is None:
                logger.warning("Unknown injection source type '%s' for '%s'", source_type, source_name)
                records = []
            else:
                records = source_cls(source_cfg).load()

            seqs = [(r["sequence"], r.get("nll", np.nan)) for r in records]
            logger.info("Source '%s' (%s): %d sequences", source_name, source_type, len(seqs))
            all_candidates.extend(seqs)

        self._all_available = [(s, nll) for s, nll in all_candidates if s not in already_evaluated]
        logger.info("Total available for injection: %d sequences", len(self._all_available))

    def get_sequences_to_inject(
        self, iteration: int, n_seqs: int, already_evaluated: Set[str]
    ) -> List[Tuple[str, float]]:
        """Return up to `n_seqs` new (sequence, nll) pairs not yet injected or evaluated."""
        self._initialize_available_sequences(already_evaluated)

        candidates = [
            (s, nll) for s, nll in self._all_available
            if s not in self.injected_sequences and s not in already_evaluated
        ]
        if not candidates:
            logger.info("No more sequences available for injection")
            return []

        selected = candidates[:n_seqs]
        self.injected_sequences.update(s for s, _ in selected)
        logger.info("Injecting %d sequences at iteration %d (%d remaining)",
                    len(selected), iteration, len(candidates) - len(selected))
        return selected
