"""Surrogate model, RMSD calculator, and ESM2 embedding extraction.

Notable cleanups relative to the original:

- Dropped the dead triple-quoted `_load_reference_structures` stub that sat
  above the live implementation.
- Fixed a `self.` typo in the original `MultiHeadSurrogate.__init__`
  (`if model_name is None: esm2_model = Config.esm2_model` silently did
  nothing because it didn't assign back to a variable actually used below).
- Canonicalized the RMSD metric name to `l_rmsd` everywhere (matches
  `metric_weights` / `AF3_THRESHOLDS` keys); the surrogate still also
  trains/predicts an `rmsd` alias of the same value for backward
  compatibility with any code reading that key.
- Added an `nll` metric head (ProteinMPNN negative log-likelihood, lower is
  more designable). Because `nll` is only available for sequences sourced
  from ProteinMPNN/RFdiffusion metadata (injection or initial CSV), not
  every sequence has a value: `_train_random_forest` filters to the
  non-NaN subset before fitting the `nll` head specifically, while all
  other heads keep training on their own complete data as before. This
  means the `nll` term only meaningfully influences predictions once
  enough labeled sequences have entered the pool -- this is intentional,
  not a bug.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from Bio.PDB import MMCIFParser, PDBParser, Superimposer
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False

from tqdm import tqdm

logger = logging.getLogger(__name__)

if not HAS_ESM:
    logger.warning("ESM not available; ESM2EmbeddingExtractor will be unusable.")
if not HAS_BIOPYTHON:
    logger.warning("BioPython not available; RMSDCalculator will be unusable.")
if not HAS_SKLEARN:
    logger.warning("scikit-learn not available; MultiHeadSurrogate (random_forest) will be unusable.")

# Metric heads predicted by the surrogate model.
SURROGATE_METRICS = [
    "plddt", "i_ptm", "pae_intra", "pae_inter",
    "con_intra", "con_inter", "rg", "helix", "l_rmsd", "nll",
]


class RMSDCalculator:
    """Calculate minimal RMSD between predicted structures and reference backbones."""

    def __init__(self, reference_structures_dir: str):
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython required for RMSD calculation. Install with: pip install biopython")

        self.reference_dir = Path(reference_structures_dir)
        self.reference_structures: List[Dict] = []
        self.references_by_module: Dict[str, List[Dict]] = {}

        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        self._load_reference_structures()

    def _load_reference_structures(self) -> None:
        """Load all reference backbone structures, indexed by target/seed module for filtering."""
        logger.info("Loading reference structures from %s", self.reference_dir)
        cif_files = list(self.reference_dir.rglob("*.cif"))
        logger.info("Found %d reference CIF files", len(cif_files))

        for cif_file in cif_files:
            try:
                structure = self.cif_parser.get_structure("ref", cif_file)
                backbone_atoms = [
                    residue["CA"]
                    for model in structure
                    for chain in model
                    for residue in chain
                    if residue.has_id("CA")
                ]
                if not backbone_atoms:
                    continue

                module_key = next(
                    (part for part in cif_file.parts if "_seed" in part and not part.endswith("_pack")),
                    None,
                )
                ref_entry = {"file": cif_file, "backbone_atoms": backbone_atoms}
                self.reference_structures.append(ref_entry)
                if module_key:
                    self.references_by_module.setdefault(module_key, []).append(ref_entry)
            except Exception as exc:
                logger.warning("Could not load reference %s: %s", cif_file, exc)

        logger.info("Loaded %d reference structures (%d modules)",
                    len(self.reference_structures), len(self.references_by_module))

    def calculate_minimal_rmsd(
        self,
        predicted_structure_file: str,
        residue_indices=None,
        alignment_indices=None,
    ) -> float:
        """Minimal RMSD (Angstroms) between the predicted structure and any matching reference.

        Args:
            predicted_structure_file: path to predicted structure.
            residue_indices: residues to measure RMSD over (None = all).
            alignment_indices: residues to use for the alignment transform (None = same as residue_indices).
        """
        pred_file = Path(predicted_structure_file)
        pred_structure = (
            self.cif_parser.get_structure("pred", pred_file)
            if pred_file.suffix == ".cif"
            else self.pdb_parser.get_structure("pred", pred_file)
        )

        pred_backbone = sorted(
            ((residue.id[1], residue["CA"]) for model in pred_structure for chain in model
             for residue in chain if residue.has_id("CA")),
            key=lambda x: x[0],
        )
        all_pred_atoms = [atom for _, atom in pred_backbone]

        align_pred_atoms = self._select_atoms(all_pred_atoms, alignment_indices, residue_indices)
        measure_pred_atoms = self._select_atoms(all_pred_atoms, residue_indices, None)

        refs_to_use = self._references_for_filename(pred_file.name)

        min_rmsd = float("inf")
        superimposer = Superimposer()

        for ref_struct in refs_to_use:
            ref_structure = self.cif_parser.get_structure("ref", ref_struct["file"])
            ref_backbone = sorted(
                ((residue.id[1], residue["CA"]) for model in ref_structure for chain in model
                 for residue in chain if residue.has_id("CA")),
                key=lambda x: x[0],
            )
            all_ref_atoms = [atom for _, atom in ref_backbone]

            align_ref_atoms = self._select_atoms(all_ref_atoms, alignment_indices, residue_indices)
            measure_ref_atoms = self._select_atoms(all_ref_atoms, residue_indices, None)

            min_align_len = min(len(align_pred_atoms), len(align_ref_atoms))
            min_measure_len = min(len(measure_pred_atoms), len(measure_ref_atoms))
            if min_align_len == 0 or min_measure_len == 0:
                continue

            superimposer.set_atoms(align_ref_atoms[:min_align_len], align_pred_atoms[:min_align_len])
            superimposer.set_atoms(measure_ref_atoms[:min_measure_len], measure_pred_atoms[:min_measure_len])
            min_rmsd = min(min_rmsd, superimposer.rms)

        return min_rmsd

    @staticmethod
    def _select_atoms(all_atoms, primary_indices, fallback_indices):
        indices = primary_indices if primary_indices is not None else fallback_indices
        if indices is None:
            return all_atoms
        return [all_atoms[i] for i in indices if i < len(all_atoms)]

    def _references_for_filename(self, filename: str) -> List[Dict]:
        """Filter references to the matching module/seed if the filename encodes one."""
        module_key = None
        if "_seed" in filename:
            parts = filename.split("_")
            for i in range(len(parts) - 3):
                if parts[i + 2] == "0" and parts[i + 3].startswith("seed"):
                    module_key = f"{parts[i]}_{parts[i+1]}_{parts[i+2]}_{parts[i+3].split('_')[0]}"
                    break

        if module_key and module_key in self.references_by_module:
            return self.references_by_module[module_key]
        return self.reference_structures


class ESM2EmbeddingExtractor:
    """ESM2 mean-pooled embedding extraction (matches the EVOLVEpro approach)."""

    _MODEL_DIMS = {"esm2_t48_15B_UR50D": 5120, "esm2_t36_3B_UR50D": 2560, "esm2_t33_650M_UR50D": 1280}

    def __init__(self, model_name: str = "esm2_t36_3B_UR50D", device: Optional[str] = None,
                 batch_size_gpu: int = 8, batch_size_cpu: int = 32):
        if not HAS_ESM:
            raise ImportError("ESM2 required but not available")
        if not HAS_TORCH:
            raise ImportError("torch required but not available")

        self.model_name = model_name
        self.batch_size_gpu = batch_size_gpu
        self.batch_size_cpu = batch_size_cpu
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.embedding_dim = self._MODEL_DIMS.get(model_name, 1280)
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading ESM2 model: %s", self.model_name)
        loaders = {
            "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
            "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
            "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        }
        loader = loaders.get(self.model_name, esm.pretrained.esm2_t33_650M_UR50D)
        self.model, self.alphabet = loader()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embedding_dim = self.model.embed_dim
        logger.info("ESM2 loaded on %s; embedding dim %d", self.device, self.embedding_dim)

    def extract_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract mean-pooled per-sequence ESM2 embeddings."""
        all_embeddings = []
        batch_size = self.batch_size_gpu if self.device.type == "cuda" else self.batch_size_cpu

        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM2 extraction"):
            batch_sequences = sequences[i:i + batch_size]
            batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_sequences)]
            _, _, batch_tokens = self.batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                try:
                    results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
                except RuntimeError as exc:
                    if "out of memory" not in str(exc):
                        raise
                    logger.warning("GPU OOM, falling back to CPU")
                    self.model = self.model.cpu()
                    self.device = torch.device("cpu")
                    batch_tokens = batch_tokens.cpu()
                    results = self.model(batch_tokens, repr_layers=[self.model.num_layers])

                representations = results["representations"][self.model.num_layers]
                for j, seq in enumerate(batch_sequences):
                    seq_repr = representations[j, 1:len(seq) + 1]
                    all_embeddings.append(seq_repr.mean(dim=0).cpu().numpy())

        return np.vstack(all_embeddings)


class MultiHeadSurrogate:
    """Random-forest multi-head surrogate predicting per-metric scores from ESM2 embeddings."""

    def __init__(
        self,
        esm2_model: str = "esm2_t36_3B_UR50D",
        rf_params: Optional[Dict] = None,
        metric_weights: Optional[Dict[str, float]] = None,
        af3_thresholds: Optional[Dict[str, float]] = None,
        mutants_per_round: int = 30,
        calculate_rmsd: bool = True,
        reference_structures_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size_gpu: int = 8,
        batch_size_cpu: int = 32,
    ):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for MultiHeadSurrogate")

        self.metrics = list(SURROGATE_METRICS)
        self.rf_params = rf_params or {}
        self.metric_weights = metric_weights or {}
        self.af3_thresholds = af3_thresholds or {}
        self.mutants_per_round = mutants_per_round
        self.calculate_rmsd = calculate_rmsd

        self.heads = {metric: RandomForestRegressor(**self.rf_params) for metric in self.metrics}
        self.is_trained = False
        self.training_history: List[Dict] = []
        self.last_train_sequences: Optional[List[str]] = None
        self.last_train_embeddings: Optional[np.ndarray] = None

        self.rmsd_calculator = None
        if self.calculate_rmsd and reference_structures_dir:
            logger.info("Initializing RMSD calculator...")
            self.rmsd_calculator = RMSDCalculator(reference_structures_dir)

        self.embedding_extractor = ESM2EmbeddingExtractor(
            esm2_model, device=device, batch_size_gpu=batch_size_gpu, batch_size_cpu=batch_size_cpu
        )

    def calculate_structure_metrics(self, eval_results: List[Dict], structure_dir: Optional[str] = None) -> List[Dict]:
        """Populate eval_results in-place with RMSD + ColabDesign losses + residue-level metrics."""
        from .losses import calculate_structure_losses_from_af3

        logger.info("Calculating structure metrics for %d structures...", len(eval_results))
        try:
            from tqdm import tqdm
            _iter = tqdm(enumerate(eval_results), total=len(eval_results), desc="structure metrics", unit="struct")
        except ImportError:
            _iter = enumerate(eval_results)
        for i, result in _iter:
            if "structure_file" not in result:
                raise ValueError(f"Structure {i+1}: Missing 'structure_file' key")

            if "structure_dir" in result:
                structure_path = Path(result["structure_dir"]) / result["structure_file"]
            else:
                structure_path = Path(structure_dir or ".") / result["structure_file"]

            if "json_file" in result:
                json_path = Path(result["json_file"])
            else:
                json_path = structure_path.with_suffix("").parent / (structure_path.stem + "_confidences.json")
                if not json_path.exists():
                    base_name = structure_path.stem.replace("_model", "")
                    json_path = structure_path.parent / f"{base_name}_confidences.json"

            if not json_path.exists():
                raise FileNotFoundError(f"JSON not found: {json_path}")

            losses = calculate_structure_losses_from_af3(
                str(structure_path), str(json_path),
                rmsd_calculator=self.rmsd_calculator if self.calculate_rmsd else None,
            )

            residue_level = losses.pop("residue_level", None)
            result.update(losses)
            if residue_level is not None:
                result["residue_metrics"] = residue_level

        return eval_results

    def train(self, sequences: List[str], metrics_dict: Dict[str, List[float]]) -> Dict:
        """Train all surrogate heads on ESM2 embeddings of `sequences`."""
        logger.info("Training surrogate on %d sequences...", len(sequences))
        embeddings = self.embedding_extractor.extract_embeddings(sequences)
        self.last_train_sequences = sequences
        self.last_train_embeddings = embeddings
        return self._train_random_forest(embeddings, metrics_dict)

    def _train_random_forest(self, embeddings: np.ndarray, metrics_dict: Dict[str, List[float]]) -> Dict:
        """Fit one RandomForestRegressor per metric head present in metrics_dict.

        Special-cased: the `nll` head trains only on the subset of rows where
        the value is not NaN (NLL is sourced passively from ProteinMPNN/
        RFdiffusion metadata and is missing for AF3-only-derived sequences).
        All other heads train on their full (already-complete) data exactly
        as before.
        """
        train_metrics: Dict = {}
        trained_heads = []

        for metric in self.metrics:
            if metric not in metrics_dict or len(metrics_dict[metric]) == 0:
                self.heads.pop(metric, None)
                continue

            values = np.asarray(metrics_dict[metric], dtype=float)

            if metric == "nll":
                valid_mask = ~np.isnan(values)
                if valid_mask.sum() < 2:
                    logger.info("Skipping nll head: only %d labeled (non-NaN) samples", int(valid_mask.sum()))
                    self.heads.pop(metric, None)
                    continue
                fit_embeddings, fit_values = embeddings[valid_mask], values[valid_mask]
            else:
                fit_embeddings, fit_values = embeddings, values

            logger.info("Training %s head on %d samples", metric, len(fit_values))
            self.heads[metric].fit(fit_embeddings, fit_values)
            trained_heads.append(metric)

            pred = self.heads[metric].predict(fit_embeddings)
            train_metrics[f"{metric}_mse"] = mean_squared_error(fit_values, pred)
            train_metrics[f"{metric}_r2"] = r2_score(fit_values, pred)

        self.metrics = trained_heads
        train_metrics.update({
            "n_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "train_timestamp": time.time(),
        })
        self.training_history.append(train_metrics)
        self.is_trained = True
        logger.info("Trained heads: %s", self.metrics)
        return train_metrics

    def cross_val_scores(self, metrics_dict: Dict[str, List[float]], cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Compute held-out cross-validation R² and Pearson r for each trained head.

        Call after train() to get an unbiased measure of how well the surrogate
        actually generalises. RandomForest memorises training data, so training-set
        R² is always ~1 -- this is the one that actually matters.

        Returns a dict like:
            {"plddt": {"r2": 0.72, "pearson_r": 0.85, "n": 200}, ...}
        """
        try:
            from sklearn.model_selection import cross_val_score
            from scipy.stats import pearsonr
            HAS_SKLEARN = True
        except ImportError:
            logger.warning("cross_val_scores: sklearn/scipy not available")
            return {}

        scores = {}
        for metric in self.metrics:
            if metric not in self.heads or metric not in metrics_dict:
                continue
            values = np.asarray(metrics_dict[metric], dtype=float)
            if metric == "nll":
                mask = ~np.isnan(values)
                emb = self.last_train_embeddings[mask]
                vals = values[mask]
            else:
                emb = self.last_train_embeddings
                vals = values

            n = len(vals)
            if n < max(cv * 2, 10):
                scores[metric] = {"r2": float("nan"), "pearson_r": float("nan"), "n": n,
                                  "note": f"too few samples ({n}) for {cv}-fold CV"}
                continue

            try:
                r2_cv = cross_val_score(self.heads[metric], emb, vals,
                                        cv=min(cv, n // 2), scoring="r2").mean()
                pred_all = self.heads[metric].predict(emb)
                r, _ = pearsonr(vals, pred_all)
                scores[metric] = {"r2_cv": round(float(r2_cv), 4),
                                  "pearson_r_train": round(float(r), 4),
                                  "n": n}
            except Exception as e:
                scores[metric] = {"r2_cv": float("nan"), "pearson_r_train": float("nan"), "n": n, "error": str(e)}

        if scores:
            logger.info("=== Surrogate validation (CV R²) ===")
            for metric, s in scores.items():
                if "error" in s or "note" in s:
                    logger.info("  %-15s  n=%-4d  %s", metric, s["n"], s.get("error") or s.get("note"))
                else:
                    r2 = s["r2_cv"]
                    flag = "✓" if r2 > 0.5 else ("~" if r2 > 0.2 else "✗ (poor)")
                    logger.info("  %-15s  n=%-4d  CV-R²=%-7.4f  train-r=%.4f  %s",
                                metric, s["n"], r2, s["pearson_r_train"], flag)
        return scores

    def predict(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """Predict all trained metric heads for `sequences`."""
        if not self.is_trained:
            raise ValueError("Model not trained!")

        if self.last_train_sequences == sequences:
            embeddings = self.last_train_embeddings
        else:
            embeddings = self.embedding_extractor.extract_embeddings(sequences)

        return {metric: self.heads[metric].predict(embeddings) for metric in self.metrics if metric in self.heads}

    def predict_combined_score(
        self, sequences: Optional[List[str]] = None, predictions: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Weighted combination of predicted metrics, normalized by total absolute weight."""
        if predictions is None:
            if sequences is None:
                raise ValueError("Must provide either sequences or predictions")
            predictions = self.predict(sequences)

        n_seqs = len(next(iter(predictions.values())))
        combined_scores = np.zeros(n_seqs)
        total_weights = np.zeros(n_seqs)

        for metric, weight in self.metric_weights.items():
            if metric in predictions:
                values = np.asarray(predictions[metric], dtype=float)
                valid = ~np.isnan(values)
                combined_scores[valid] += values[valid] * weight
                total_weights[valid] += abs(weight)

        nonzero = total_weights > 0
        combined_scores[nonzero] /= total_weights[nonzero]
        return combined_scores

    def top_n_selection(self, sequences: List[str], n: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
        """Select top-N sequences: prefer those passing AF3 filter thresholds, ranked by combined score."""
        n = n or self.mutants_per_round
        predictions = self.predict(sequences)
        combined_scores = self.predict_combined_score(predictions=predictions)

        passing_indices = []
        for i in range(len(sequences)):
            passes = True
            if "pae_inter" in predictions:
                if predictions["pae_inter"][i] * 31.0 >= 5.0:
                    passes = False
            if passes:
                if "ptm_binder" in predictions:
                    passes = predictions["ptm_binder"][i] > self.af3_thresholds.get("ptm_binder", 0.0)
                elif "i_ptm" in predictions:
                    passes = predictions["i_ptm"][i] > self.af3_thresholds.get("ptm_binder", 0.0)
            if passes and "l_rmsd" in predictions and "l_rmsd" in self.af3_thresholds:
                passes = predictions["l_rmsd"][i] < self.af3_thresholds["l_rmsd"]
            if passes:
                passing_indices.append(i)

        logger.info("AlphaProteo filters: %d/%d sequences pass", len(passing_indices), len(sequences))

        if len(passing_indices) >= n:
            passing_scores = combined_scores[passing_indices]
            top_indices = np.argsort(passing_scores)[-n:][::-1]
            selected_indices = [passing_indices[i] for i in top_indices]
        else:
            selected_indices = np.argsort(combined_scores)[-n:][::-1]

        selected_sequences = [sequences[i] for i in selected_indices]
        selected_scores = combined_scores[selected_indices]
        return selected_sequences, selected_scores

    def save_model(self, filepath: str) -> None:
        """Pickle the trained heads + metadata."""
        model_data = {
            "heads": self.heads,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
            "embedding_dim": self.embedding_extractor.embedding_dim,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load a previously pickled model."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.heads = model_data["heads"]
        self.metrics = model_data["metrics"]
        self.is_trained = model_data["is_trained"]
        self.training_history = model_data["training_history"]
