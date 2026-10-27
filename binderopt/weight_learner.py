"""Adaptive weight learning for the combined score, based on AF3 filter success.

Optional and
off by default (`RunConfig.learn_weights = False`); the optimizer falls
back to the fixed `metric_weights` from config when this is disabled.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class AdaptiveWeightLearner:
    """Learns metric_weights via logistic regression against AF3 experimental thresholds."""

    def __init__(self, af3_thresholds: Dict[str, float]):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for AdaptiveWeightLearner")
        self.af3_thresholds = af3_thresholds
        self.weights: Optional[Dict[str, float]] = None
        self.weight_history: List[Dict] = []

    def define_success(self, metrics_dict: Dict[str, List[float]]) -> np.ndarray:
        """Boolean array: True where a sequence passes all configured AF3 thresholds."""
        n_seqs = len(next(iter(metrics_dict.values())))
        success = np.ones(n_seqs, dtype=bool)

        if "min_pae_interaction" in metrics_dict and "min_pae_interaction" in self.af3_thresholds:
            success &= np.array(metrics_dict["min_pae_interaction"]) < self.af3_thresholds["min_pae_interaction"]
        if "ptm_binder" in metrics_dict and "ptm_binder" in self.af3_thresholds:
            success &= np.array(metrics_dict["ptm_binder"]) > self.af3_thresholds["ptm_binder"]
        if "l_rmsd" in metrics_dict and "l_rmsd" in self.af3_thresholds:
            success &= np.array(metrics_dict["l_rmsd"]) < self.af3_thresholds["l_rmsd"]

        return success

    def learn_weights(
        self, metrics_dict: Dict[str, List[float]], metric_names: List[str], iteration: int = 0
    ) -> Optional[Dict[str, float]]:
        """Fit a logistic regression predicting AF3-filter success from metrics; return normalized weights."""
        y = self.define_success(metrics_dict)
        success_rate = y.mean()
        n_success = int(y.sum())
        n_total = len(y)

        logger.info("[Weight Learning iter %d] success rate %.1f%% (%d/%d)",
                    iteration, success_rate * 100, n_success, n_total)

        if n_success < 2:
            logger.error("Only %d successful sequences, need at least 2", n_success)
            return None
        if success_rate >= 0.95:
            logger.error("Success rate too high (%.1f%%), need some failures for learning", success_rate * 100)
            return None

        X_cols, valid_metrics = [], []
        for metric in metric_names:
            if metric in metrics_dict:
                X_cols.append(np.array(metrics_dict[metric]))
                valid_metrics.append(metric)

        if not X_cols:
            logger.error("No valid metrics for weight learning")
            return None

        X = np.column_stack(X_cols)
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
        try:
            lr.fit(X, y)
        except Exception as exc:
            logger.error("Logistic regression training failed: %s", exc)
            return None

        try:
            cv_folds = min(5, n_success)
            cv_scores = cross_val_score(lr, X, y, cv=cv_folds, scoring="roc_auc")
            cv_mean = float(cv_scores.mean())
            logger.info("Cross-validation AUC: %.3f +/- %.3f", cv_mean, cv_scores.std())
        except Exception as exc:
            logger.warning("Cross-validation failed: %s", exc)
            cv_mean = 0.0

        raw_weights = lr.coef_[0]
        normalized_weights = raw_weights / np.abs(raw_weights).sum()
        learned_weights = {metric: float(w) for metric, w in zip(valid_metrics, normalized_weights)}

        self.weight_history.append({
            "iteration": iteration,
            "weights": learned_weights,
            "cv_auc": cv_mean,
            "success_rate": float(success_rate),
            "n_samples": n_total,
            "n_success": n_success,
            "metrics_used": valid_metrics,
        })
        self.weights = learned_weights
        return learned_weights

    def calculate_hit_rate(
        self, metrics_dict: Dict[str, List[float]], predicted_scores: np.ndarray, top_n: int = 100
    ) -> Dict:
        """Of the top-N sequences by predicted score, what fraction pass AF3 filters (vs random baseline)?"""
        success = self.define_success(metrics_dict)
        top_n = min(top_n, len(predicted_scores))
        top_indices = np.argsort(predicted_scores)[-top_n:]

        hits = int(success[top_indices].sum())
        hit_rate = hits / top_n
        random_hit_rate = float(success.mean())
        enrichment = hit_rate / random_hit_rate if random_hit_rate > 0 else 0.0

        return {
            "hit_rate": hit_rate, "hits": hits, "top_n": top_n,
            "random_baseline": random_hit_rate, "enrichment": enrichment,
        }

    def save_history(self, filepath: str) -> None:
        """Save the full weight-learning history to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.weight_history, f, indent=2)
