"""Active per-candidate ProteinMPNN scoring (true NLL, not just inherited metadata)."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .external import optional_tool_path

try:
    from Bio.PDB import MMCIFParser, PDBIO

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

logger = logging.getLogger(__name__)

HAS_PROTEINMPNN = (
    HAS_BIOPYTHON  # narrowed further per-instance in __init__ (paths may not exist)
)
if not HAS_BIOPYTHON:
    logger.warning(
        "BioPython not available; ProteinMPNNScorer will be unusable (cif->pdb conversion needs it)."
    )


class ProteinMPNNScorer:
    """Scores a (structure, binder_sequence) pair with real ProteinMPNN `score_only` mode."""

    def __init__(
        self,
        mpnn_repo_dir: Optional[str] = None,
        mpnn_python: Optional[str] = None,
        model_name: str = "v_48_020",
        enabled: bool = True,
        timeout: int = 300,
        binder_chain: str = "B",
        seed: int = 42,
    ):
        mpnn_repo_dir = mpnn_repo_dir or optional_tool_path("SCALE_PROTEINMPNN_DIR")
        mpnn_python = mpnn_python or optional_tool_path("SCALE_PROTEINMPNN_PYTHON")
        self.mpnn_repo_dir = Path(mpnn_repo_dir) if mpnn_repo_dir else None
        self.mpnn_python = mpnn_python
        self.model_name = model_name
        self.timeout = timeout
        self.binder_chain = binder_chain
        self.seed = seed

        self.enabled = bool(enabled)
        if self.enabled and not HAS_BIOPYTHON:
            logger.warning(
                "ProteinMPNNScorer disabled: BioPython not available for cif->pdb conversion."
            )
            self.enabled = False
        if self.enabled and (self.mpnn_repo_dir is None or self.mpnn_python is None):
            logger.warning(
                "ProteinMPNNScorer disabled: set SCALE_PROTEINMPNN_DIR and "
                "SCALE_PROTEINMPNN_PYTHON (or pass them) to enable it."
            )
            self.enabled = False
        if self.enabled:
            run_script = self.mpnn_repo_dir / "protein_mpnn_run.py"
            if not run_script.exists():
                logger.warning("ProteinMPNNScorer disabled: %s not found.", run_script)
                self.enabled = False
            elif not Path(self.mpnn_python).exists():
                logger.warning(
                    "ProteinMPNNScorer disabled: mpnn python binary not found at %s.",
                    self.mpnn_python,
                )
                self.enabled = False

        if self.enabled:
            logger.info(
                "ProteinMPNNScorer enabled (repo=%s, python=%s, model=%s)",
                self.mpnn_repo_dir,
                self.mpnn_python,
                self.model_name,
            )
        else:
            logger.warning(
                "ProteinMPNNScorer disabled; score() will return NaN for all calls."
            )

    # ------------------------------------------------------------------

    def score(self, structure_cif_path: str, binder_sequence: str) -> float:
        """Return the real ProteinMPNN score_only NLL for `binder_sequence` against the
        structure in `structure_cif_path` (an AF3 `*_model.cif`). NaN on any failure or
        if this scorer is disabled.
        """
        if not self.enabled:
            return float("nan")

        cif_path = Path(structure_cif_path)
        if not cif_path.exists():
            logger.warning(
                "ProteinMPNN scoring skipped: structure file not found: %s", cif_path
            )
            return float("nan")

        try:
            with tempfile.TemporaryDirectory(prefix="mpnn_score_") as tmp_dir:
                tmp_dir = Path(tmp_dir)
                pdb_basename = (
                    cif_path.stem
                )  # used as the npz output key by protein_mpnn_run.py
                pdb_path = tmp_dir / f"{pdb_basename}.pdb"

                binder_chain = self._convert_cif_to_pdb(
                    cif_path, pdb_path, len(binder_sequence)
                )
                if binder_chain is None:
                    return float("nan")

                fasta_path = tmp_dir / "binder.fasta"
                with open(fasta_path, "w") as f:
                    f.write(">binder\n")
                    f.write(f"{binder_sequence}\n")

                out_dir = tmp_dir / "out"
                out_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    self.mpnn_python,
                    str(self.mpnn_repo_dir / "protein_mpnn_run.py"),
                    "--pdb_path",
                    str(pdb_path),
                    "--pdb_path_chains",
                    binder_chain,
                    "--path_to_fasta",
                    str(fasta_path),
                    "--score_only",
                    "1",
                    "--save_score",
                    "1",
                    "--out_folder",
                    str(out_dir),
                    "--model_name",
                    self.model_name,
                    "--seed",
                    str(self.seed),
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.mpnn_repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if proc.returncode != 0:
                    logger.warning(
                        "ProteinMPNN scoring failed (exit %d) for %s: stdout=%s stderr=%s",
                        proc.returncode,
                        cif_path,
                        proc.stdout[-2000:],
                        proc.stderr[-2000:],
                    )
                    return float("nan")

                npz_path = out_dir / "score_only" / f"{pdb_basename}_fasta_1.npz"
                if not npz_path.exists():
                    logger.warning(
                        "ProteinMPNN scoring: expected output not found at %s", npz_path
                    )
                    return float("nan")

                data = np.load(npz_path)
                return float(data["score"].mean())
        except subprocess.TimeoutExpired:
            logger.warning(
                "ProteinMPNN scoring timed out (>%ds) for %s", self.timeout, cif_path
            )
            return float("nan")
        except Exception as exc:
            logger.warning("ProteinMPNN scoring failed for %s: %s", cif_path, exc)
            return float("nan")

    # ------------------------------------------------------------------

    def _convert_cif_to_pdb(
        self, cif_path: Path, pdb_path: Path, expected_length=None
    ) -> Optional[str]:
        """Convert an AF3 `.cif` to `.pdb` and return the binder chain ID, or None on failure."""
        try:
            parser = MMCIFParser(QUIET=True, auth_chains=False, auth_residues=False)
            structure = parser.get_structure("pred", str(cif_path))
            binder_chain = self.binder_chain
            if binder_chain not in structure[0]:
                raise ValueError(f"Binder chain {binder_chain!r} is absent")
            length = sum(residue.has_id("CA") for residue in structure[0][binder_chain])
            if expected_length is not None and length != expected_length:
                raise ValueError("Binder sequence length differs from its structure")

            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_path))
            return binder_chain
        except Exception as exc:
            logger.warning("cif->pdb conversion failed for %s: %s", cif_path, exc)
            return None
