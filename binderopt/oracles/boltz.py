"""Boltz-2 backend.

Boltz-2 is called through its command line (``boltz predict``) on a two-chain YAML, one
job per binder sequence. Two details of its output shape this adapter.

First, the confidence JSON (``confidence_<name>_model_0.json``) reports
``confidence_score``, ``ptm``, ``iptm``, ``complex_plddt``, ``chains_ptm`` and
``pair_chains_iptm``, but **no summarised interface PAE** - the AlphaFold 3 field
``chain_pair_pae_min`` has no counterpart. The loop's ``min_pae_interaction`` is
therefore computed here from the per-token matrix in ``pae_<name>_model_0.npz``, which
requires ``--write_full_pae``; without it the metric is returned as NaN rather than
silently replaced by a different quantity.

Flag names verified against the boltz 2.2.1 command line, not inferred from prose: the
options are ``--write_full_pae`` and ``--use_potentials`` (there is no ``--write_pae``
and no ``--no_potentials``; potentials are off by default).

Second, chains are indexed by order of appearance ("0", "1", ...), so the receptor is
chain 0 and the binder chain 1 under the convention used throughout this repository.

The numbers are not interchangeable with AlphaFold 3's. Boltz-2 confidence has its own
calibration, so the manuscript's hit criteria do not transfer, and results from the two
oracles must not be pooled. Metrics carry ``oracle="boltz2"`` to make that explicit.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..af3_interface import StructureEvaluator
from .metrics import RESIDUE_METRICS, normalise, validate_residue_metrics
from .registry import register_oracle

RECEPTOR_CHAIN, BINDER_CHAIN = "0", "1"


def _installed_version(executable: str) -> str:
    """The installed Boltz version, for the provenance stamp.

    Recorded rather than hardcoded because successive Boltz releases are not guaranteed
    to be numerically comparable either: "boltz2" alone does not identify a calibration.
    Falls back to a generic label if the version cannot be read.
    """
    try:
        from importlib.metadata import version
        return "boltz-%s" % version("boltz")
    except Exception:          # noqa: BLE001 - provenance must never break a prediction
        return "boltz2-cli"


class Boltz2Oracle(StructureEvaluator):
    """Predict binder-receptor complexes with Boltz-2.

    Args:
        target_sequence: receptor sequence, written as chain A (index "0").
        work_dir: where YAML inputs and predictions are kept. A temporary directory is
            used when omitted, and removed by :meth:`cleanup`.
        executable: ``boltz`` entry point.
        msa_path: optional A3M for the receptor. ``None`` requests single-sequence mode
            (``msa: empty``), which is markedly less accurate for the receptor fold,
            unless ``use_msa_server`` asks Boltz to fetch one instead.
        use_msa_server: let Boltz build the receptor MSA from the public MMseqs2 server
            (``--use_msa_server``). Mutually exclusive with ``msa_path``. The binder is
            always single-sequence: it is a design, so it has no homologues to align.
            Note that this sends the receptor sequence to a third-party server.
        diffusion_samples: samples per prediction; the ranked best is read back.
        extra_args: appended to the command line verbatim.
    """

    #: Provenance stamp, replaced in ``__init__`` by the installed version. The class
    #: attribute keeps ``_collect`` usable on an instance built without ``__init__``,
    #: as the parsing tests do to skip the PATH check.
    oracle_version = "boltz2-cli"

    def __init__(
        self,
        target_sequence: str,
        work_dir: Optional[str] = None,
        executable: str = "boltz",
        msa_path: Optional[str] = None,
        use_msa_server: bool = False,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        use_potentials: bool = False,
        devices: int = 1,
        accelerator: str = "gpu",
        extra_args: Optional[List[str]] = None,
        timeout_s: int = 3600,
    ):
        if shutil.which(executable) is None:
            raise FileNotFoundError(
                "%r not found on PATH; install with `pip install boltz`" % executable)
        if msa_path and use_msa_server:
            raise ValueError(
                "pass either msa_path (your own A3M) or use_msa_server (fetch one), not both")
        self.target_sequence = target_sequence
        self.executable = executable
        self.msa_path = msa_path
        self.use_msa_server = use_msa_server
        self.diffusion_samples = diffusion_samples
        self.recycling_steps = recycling_steps
        self.use_potentials = use_potentials
        self.devices = devices
        self.accelerator = accelerator
        self.extra_args = list(extra_args or [])
        self.timeout_s = timeout_s
        self._tmp = work_dir is None
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="boltz_oracle_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.oracle_version = _installed_version(executable)

    # -- input ------------------------------------------------------------------
    def _write_yaml(self, binder_sequence: str, name: str) -> Path:
        # With --use_msa_server the receptor's `msa` key must be ABSENT: an explicit
        # `msa: empty` is a request for single-sequence mode and silently wins over the
        # flag. The binder keeps `msa: empty` either way -- it is a design, so a search
        # would find nothing and would only cost a server round-trip.
        doc = [
            "version: 1",
            "sequences:",
            "  - protein:",
            "      id: A",
            "      sequence: %s" % self.target_sequence,
        ]
        if not self.use_msa_server:
            doc.append("      msa: %s" % (self.msa_path or "empty"))
        doc += [
            "  - protein:",
            "      id: B",
            "      sequence: %s" % binder_sequence,
            "      msa: empty",
            "",
        ]
        path = self.work_dir / ("%s.yaml" % name)
        path.write_text("\n".join(doc))
        return path

    # -- output -----------------------------------------------------------------
    @staticmethod
    def _load_pae(pae_npz: Path, n_receptor: int) -> np.ndarray:
        """The full token-by-token PAE matrix, checked against the receptor length."""
        with np.load(pae_npz) as handle:
            pae = handle[handle.files[0]]
        pae = np.asarray(pae, dtype=float)
        if pae.ndim == 3:
            pae = pae[0]
        if pae.shape[0] <= n_receptor:
            raise ValueError("PAE matrix %s smaller than the receptor (%d)"
                             % (pae.shape, n_receptor))
        return pae

    @staticmethod
    def _min_interface_pae(pae: np.ndarray, n_receptor: int) -> float:
        """Smallest cross-chain PAE, the analogue of AF3's ``chain_pair_pae_min``."""
        cross = np.concatenate([pae[:n_receptor, n_receptor:].ravel(),
                                pae[n_receptor:, :n_receptor].ravel()])
        return float(cross.min())

    @staticmethod
    def _residue_pae(pae: np.ndarray, n_receptor: int) -> Dict[str, np.ndarray]:
        """Per-binder-residue interface and packing PAE, in Angstrom.

        PAE is not symmetric: ``pae[i, j]`` is the error in j's position when the frame is
        aligned on i, so a binder residue has two interface values. They correlate closely
        (0.92 on a measured complex) but not exactly, and neither direction alone is the
        quantity of interest -- how badly placed this residue is relative to the receptor
        -- so both are averaged, as AF3's ``chain_pair_pae_min`` also considers both.
        """
        to_receptor = pae[n_receptor:, :n_receptor].mean(axis=1)
        from_receptor = pae[:n_receptor, n_receptor:].mean(axis=0)
        return {
            "pae_inter": (to_receptor + from_receptor) / 2.0,
            "pae_intra": pae[n_receptor:, n_receptor:].mean(axis=1),
        }

    @staticmethod
    def _residue_plddt(plddt_npz: Path, n_receptor: int) -> np.ndarray:
        """Per-binder-residue pLDDT. Boltz writes 0-1, which is the contract's scale."""
        with np.load(plddt_npz) as handle:
            plddt = np.asarray(handle[handle.files[0]], dtype=float)
        if plddt.ndim == 2:
            plddt = plddt[0]
        return plddt[n_receptor:]

    def _collect_residue_metrics(self, pred_dir: Path, pae: Optional[np.ndarray],
                                 n_receptor: int) -> Dict[str, List[float]]:
        """Per-residue arrays for the guided sampler, or ``{}`` when unavailable.

        Optional by contract: a missing file means the sampler mutates uniformly rather
        than the run failing. Lists rather than arrays, because these travel through the
        notebook's JSON checkpoint.
        """
        residue: Dict[str, np.ndarray] = {}
        if pae is not None:
            residue.update(self._residue_pae(pae, n_receptor))
        plddt_files = sorted(pred_dir.glob("plddt_*_model_0.npz"))
        if plddt_files:
            residue["plddt"] = self._residue_plddt(plddt_files[0], n_receptor)
        if not residue:
            return {}
        validate_residue_metrics(residue, where="Boltz2Oracle")
        return {k: [float(v) for v in a] for k, a in residue.items()}

    def _collect(self, out_dir: Path, name: str) -> Dict:
        pred_dir = out_dir / "predictions" / name
        conf_files = sorted(pred_dir.glob("confidence_*_model_0.json"))
        if not conf_files:
            raise FileNotFoundError("no confidence JSON under %s" % pred_dir)
        conf = json.loads(conf_files[0].read_text())

        n_receptor = len(self.target_sequence)
        pae_files = sorted(pred_dir.glob("pae_*_model_0.npz"))
        if pae_files:
            pae = self._load_pae(pae_files[0], n_receptor)
            min_pae = self._min_interface_pae(pae, n_receptor)
        else:
            pae = None
            min_pae = float("nan")   # --write_full_pae was not passed; do not substitute

        structures = sorted(pred_dir.glob("*_model_0.cif"))
        pair = conf.get("pair_chains_iptm", {})
        pairwise = pair.get(RECEPTOR_CHAIN, {}).get(BINDER_CHAIN)
        return normalise(
            {
                "i_ptm": pairwise if pairwise is not None else conf.get("iptm"),
                "ptm_binder": conf.get("chains_ptm", {}).get(BINDER_CHAIN),
                "min_pae_interaction": min_pae,
                # Passed through under Boltz's own names; the loop reads only the
                # contract names above.
                "complex_iptm": float(conf.get("iptm", float("nan"))),
                "complex_ptm": float(conf.get("ptm", float("nan"))),
                "complex_plddt": float(conf.get("complex_plddt", float("nan"))),
                "confidence_score": float(conf.get("confidence_score", float("nan"))),
                RESIDUE_METRICS: self._collect_residue_metrics(pred_dir, pae, n_receptor),
            },
            oracle="boltz2",
            oracle_version=self.oracle_version,
            structure_file=str(structures[0]) if structures else None,
        )

    # -- StructureEvaluator -------------------------------------------------------
    def predict(self, binder_sequence: str, sequence_id: Optional[str] = None,
                save_structure: bool = False, structure_dir: Optional[str] = None,
                **kwargs) -> Tuple[float, Dict]:
        """Predict one complex.

        ``save_structure``/``structure_dir`` are named rather than swallowed by
        ``**kwargs``: :class:`~binderopt.oracles.evaluator.OracleEvaluator` passes them for
        every design, and absorbing them silently left the caller's structure directory
        empty while ``structure_file`` still pointed into the scratch tree.
        """
        # hashlib, not hash(): the builtin is salted per process and would break caching.
        name = sequence_id or ("binder_%s" % hashlib.sha256(binder_sequence.encode()).hexdigest()[:16])
        yaml_path = self._write_yaml(binder_sequence, name)
        out_dir = self.work_dir / ("out_%s" % name)
        cmd = [
            self.executable, "predict", str(yaml_path),
            "--out_dir", str(out_dir),
            "--output_format", "mmcif",
            "--diffusion_samples", str(self.diffusion_samples),
            "--recycling_steps", str(self.recycling_steps),
            "--accelerator", self.accelerator,
            "--devices", str(self.devices),
            "--write_full_pae",
        ]
        if self.use_potentials:
            cmd.append("--use_potentials")
        if self.use_msa_server:
            cmd.append("--use_msa_server")
        cmd += self.extra_args
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError("boltz predict failed (%d): %s"
                               % (proc.returncode, proc.stderr[-800:]))
        metrics = self._collect(out_dir / ("boltz_results_%s" % name)
                                if (out_dir / ("boltz_results_%s" % name)).exists() else out_dir,
                                name)

        # Copy the model out of the scratch tree, which cleanup() may remove, and point
        # structure_file at the copy so a later reader cannot land on a deleted path.
        if save_structure and structure_dir and metrics.get("structure_file"):
            dest_dir = Path(structure_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            src = Path(metrics["structure_file"])
            dest = dest_dir / ("%s%s" % (name, src.suffix))
            shutil.copyfile(src, dest)
            metrics["structure_file"] = str(dest)

        return metrics["i_ptm"], metrics

    def cleanup(self) -> None:
        if self._tmp and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


@register_oracle("boltz2")
def _build_boltz2(**kwargs) -> Boltz2Oracle:
    return Boltz2Oracle(**kwargs)
