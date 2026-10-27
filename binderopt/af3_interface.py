"""AlphaFold3 subprocess interface: MSA preparation + complex prediction.

Cleaned up from the original script: type hints, logging instead of bare
prints, pathlib throughout. Behavior is unchanged.

Also defines `StructureEvaluator`, the abstract adapter interface for
structure/binding-confidence scorers (see that class's docstring). AF3 via
`AlphaFold3Interface` is currently the only implementation; a future
BoltzGen or other structure-scoring backend would implement the same
interface.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod

from .external import tool_path
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _run_alphafold_script() -> str:
    """Path to AlphaFold 3's run_alphafold.py, from SCALE_AF3_RUN.

    Resolved per call rather than at import so that importing this module -- which the
    oracle registry does unconditionally -- never depends on AF3 being installed.
    """
    return tool_path("SCALE_AF3_RUN")


class StructureEvaluator(ABC):
    """Abstract adapter for tools that score a sequence's predicted binding/structure confidence.

    This is the extension point for plugging in a structure-prediction
    backend other than AlphaFold3 (e.g. a future BoltzGen integration).
    Implementations take a candidate binder sequence and return per-sequence
    confidence/structure metrics. `AlphaFold3Interface` below is the sole
    current implementation, via its `predict()` method (see
    `binderopt.optimizer.AF3Evaluator.evaluate()` for the call site that
    wraps `predict()` per-sequence and adds derived loss metrics).
    """

    @abstractmethod
    def predict(self, binder_sequence: str, **kwargs) -> Tuple[float, Dict]:
        """Predict a binder:target complex for one sequence; return (confidence, metrics)."""
        raise NotImplementedError


class AlphaFold3Interface(StructureEvaluator):
    """Runs AlphaFold3 to predict binder:target complexes and parses confidence outputs."""

    def __init__(
        self,
        db_dir: str,
        model_dir: str,
        target_names: List[str],
        target_sequences: List[str],
        target_msa_dirs: Optional[List[str]] = None,
        af3_results_dir: str = "af3_batch_results",
        msa_cache_dir: str = "target_msa_cache",
    ):
        """
        Args:
            db_dir: AF3 database directory.
            model_dir: AF3 model directory.
            target_names: list of target names (e.g. ["BHRF1"]).
            target_sequences: list of target protein sequences, same length as target_names.
            target_msa_dirs: pre-computed MSA directories; if None, MSAs are computed and cached.
            af3_results_dir: directory AF3 writes raw per-sequence prediction output to.
            msa_cache_dir: directory used to cache target-only MSA results across runs.
        """
        self.db_dir = db_dir
        self.model_dir = model_dir
        self.target_names = list(target_names)
        self.target_sequences = list(target_sequences)
        self.msa_cache_dir = Path(msa_cache_dir)
        self.af3_results_dir = Path(af3_results_dir)

        if len(self.target_names) != len(self.target_sequences):
            raise ValueError("Number of target names must match number of sequences")

        logger.info("AlphaFold3Interface targets: %s", ", ".join(self.target_names))

        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info("AF3 temp directory: %s", self.temp_dir)

        if target_msa_dirs is None:
            self.target_msa_dirs = [
                self._prepare_target_msa(name, seq)
                for name, seq in zip(self.target_names, self.target_sequences)
            ]
        else:
            self.target_msa_dirs = list(target_msa_dirs)
            logger.info("Using existing target MSAs: %s", self.target_msa_dirs)

    def _prepare_target_msa(self, target_name: str, target_sequence: str) -> str:
        """Run (or reuse cached) MSA search for a single target sequence."""
        target_output = self.msa_cache_dir / f"{target_name}_msa"
        target_output.mkdir(exist_ok=True, parents=True)

        existing_data_files = list(target_output.glob(f"*/{target_name}_only_data.json"))
        if existing_data_files:
            logger.info("Found existing MSA for %s", target_name)
            return str(existing_data_files[0].parent)

        target_json = {
            "dialect": "alphafold3",
            "version": 4,
            "name": f"{target_name}_only",
            "sequences": [{"protein": {"id": "TARGET", "sequence": target_sequence}}],
            "modelSeeds": [42],
        }
        target_json_file = target_output / f"{target_name}_only.json"
        with open(target_json_file, "w") as f:
            json.dump(target_json, f, indent=2)

        cmd = [
            "python3", _run_alphafold_script(),
            f"--json_path={target_json_file}",
            f"--db_dir={self.db_dir}",
            f"--model_dir={self.model_dir}",
            f"--output_dir={target_output}",
            "--jackhmmer_n_cpu=8",
            "--run_data_pipeline",
            "--norun_inference",
        ]
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"

        logger.info("Running MSA search for %s...", target_name)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(f"Target MSA failed for {target_name}: {result.stderr}")

        target_data_files = list(target_output.glob(f"*/{target_name}_only_data.json"))
        if not target_data_files:
            raise RuntimeError(f"Target MSA data not found for {target_name}")

        target_data_dir = target_data_files[0].parent
        logger.info("Target MSA saved to: %s", target_data_dir)
        return str(target_data_dir)

    def predict(
        self,
        binder_sequence: str,
        save_structure: bool = True,
        structure_dir: str = "af3_structures",
        sequence_id: Optional[str] = None,
    ) -> Tuple[float, Dict]:
        """Predict a binder:target complex and return (confidence, metrics)."""
        seq_id = sequence_id or f"seq_{len(binder_sequence):03d}"
        combined_json_file = self._create_complex_json(binder_sequence, seq_id)
        return self._run_inference(combined_json_file, save_structure, structure_dir, seq_id)

    def _create_complex_json(self, binder_sequence: str, seq_id: str) -> str:
        """Create the AF3 input JSON: target chain(s) with cached MSA + binder chain without MSA.

        Each target gets its own per-target MSA dir/file from ``_prepare_target_msa``
        (named ``{target_name}_only_data.json``, not a single shared
        "target_chains_data.json" -- there is no combined multi-target MSA file
        in this codebase's MSA-prep flow, each target's data pipeline runs
        separately).
        """
        chain_ids = ["A", "B", "C", "D", "E", "F"]
        sequences = []
        for i, (name, seq) in enumerate(zip(self.target_names, self.target_sequences)):
            target_msa_dir = Path(self.target_msa_dirs[i])
            data_files = list(target_msa_dir.glob(f"{name}_only_data.json")) or list(
                target_msa_dir.glob("*_data.json")
            )
            if not data_files:
                raise FileNotFoundError(
                    f"No *_data.json MSA file found in {target_msa_dir} for target {name}"
                )
            with open(data_files[0]) as f:
                target_data = json.load(f)

            target_protein = next(
                (entry["protein"] for entry in target_data["sequences"]
                 if "protein" in entry and entry["protein"]["sequence"] == seq),
                None,
            )
            if target_protein is None:
                raise ValueError(f"Target sequence for {name} not found in MSA file {data_files[0]}")
            sequences.append({
                "protein": {
                    "id": chain_ids[i],
                    "sequence": seq,
                    "modifications": [],
                    "unpairedMsa": target_protein.get("unpairedMsa", ""),
                    "pairedMsa": target_protein.get("pairedMsa", ""),
                    "templates": target_protein.get("templates", []),
                }
            })

        binder_chain_id = chain_ids[len(self.target_names)]
        sequences.append({
            "protein": {
                "id": binder_chain_id,
                "sequence": binder_sequence,
                "modifications": [],
                "unpairedMsa": "",
                "pairedMsa": "",
                "templates": [],
            }
        })

        complex_json = {
            "dialect": "alphafold3",
            "version": 4,
            "name": f"{'_'.join(self.target_names)}_{seq_id}",
            "sequences": sequences,
            "modelSeeds": [42],
        }

        output_dir = self.temp_dir / seq_id
        output_dir.mkdir(exist_ok=True)
        combined_json_file = output_dir / f"{seq_id}_complex_data.json"
        with open(combined_json_file, "w") as f:
            json.dump(complex_json, f, indent=2)

        return str(combined_json_file)

    def _run_inference(
        self,
        combined_json_file: str,
        save_structure: bool,
        structure_dir: Optional[str],
        seq_id: str,
    ) -> Tuple[float, Dict]:
        """Run AF3 inference and extract metrics from the nested output directory."""
        output_dir = self.af3_results_dir / seq_id
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3", _run_alphafold_script(),
            "--json_path", str(combined_json_file),
            "--model_dir", str(self.model_dir),
            "--db_dir", str(self.db_dir),
            "--output_dir", str(output_dir),
            "--norun_data_pipeline",
            "--run_inference",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"AF3 prediction failed for {seq_id}: {result.stderr}")

        summary_files = list(output_dir.rglob("*_summary_confidences.json"))
        if not summary_files:
            raise RuntimeError(f"No summary file found in {output_dir}")

        with open(summary_files[0]) as f:
            summary = json.load(f)

        confidence = summary.get("ranking_score", 0)
        metrics = {
            "iptm": summary.get("iptm"),
            "ptm": summary.get("ptm"),
            "ranking_score": summary.get("ranking_score"),
            "chain_pair_pae_min": summary.get("chain_pair_pae_min", [[0, 0], [0, 0]]),
            "chain_ptm": summary.get("chain_ptm", [0, 0]),
        }

        if save_structure:
            model_files = list(output_dir.rglob("*_model.cif"))
            if model_files and structure_dir:
                nested_subdir = output_dir / f"{'_'.join(self.target_names)}_{seq_id}"
                if nested_subdir.exists():
                    root_files = [f for f in model_files if f.parent == nested_subdir]
                    structure_file = root_files[0] if root_files else model_files[0]
                else:
                    structure_file = model_files[0]

                dest_dir = Path(structure_dir) / seq_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(structure_file, dest_dir / structure_file.name)

                conf_json = structure_file.parent / structure_file.name.replace(
                    "_model.cif", "_confidences.json"
                )
                if conf_json.exists():
                    shutil.copy2(conf_json, dest_dir / conf_json.name)
                shutil.copy2(summary_files[0], dest_dir / summary_files[0].name)

        return confidence, metrics

    def cleanup(self) -> None:
        """Remove the temporary directory used for per-sequence AF3 input JSONs."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temp directory: %s", self.temp_dir)
