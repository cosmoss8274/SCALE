"""Scalar objectives and weighted metric scores; larger values are preferred."""

from __future__ import annotations

import math
from typing import Callable, Dict, Sequence
import numpy as np

Objective = Callable[[Dict], float]


def _value(metrics, key):
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"Objective requires a finite {key!r} value")
    return value


def maximise(metric: str) -> Objective:
    """Maximise a finite scalar metric."""

    def f(metrics):
        return _value(metrics, metric)

    f.description = f"maximise {metric}"
    return f


def minimise(metric: str) -> Objective:
    """Minimise a finite scalar metric by maximising its negative."""

    def f(metrics):
        return -_value(metrics, metric)

    f.description = f"minimise {metric}"
    return f


def _weights(weights):
    if not weights or any(not math.isfinite(float(w)) for w in weights.values()):
        raise ValueError("Weights must be a nonempty mapping of finite numbers")
    result = {key: float(w) for key, w in weights.items() if w != 0}
    if not result:
        raise ValueError("At least one metric weight must be nonzero")
    return result


def weighted(weights: Dict[str, float]) -> Objective:
    """Weighted sum; assign negative weights to quantities to minimise."""
    weights = _weights(weights)

    def f(metrics):
        return float(sum(w * _value(metrics, key) for key, w in weights.items()))

    f.description = "weighted sum " + ", ".join(
        f"{w:+g}*{key}" for key, w in weights.items()
    )
    return f


def weighted_scores(metrics, weights):
    """Normalised weighted scores for aligned metric columns.

    Columns absent or entirely unmeasured in the batch are inactive. A missing value within an active
    column makes that row ineligible for selection (negative infinity).
    """
    weights = _weights(weights)
    active = {
        key: w
        for key, w in weights.items()
        if key in metrics and not np.isnan(np.asarray(metrics[key], dtype=float)).all()
    }
    if not active:
        raise ValueError("No weighted metrics are available")
    arrays = [np.asarray(metrics[key], dtype=float) for key in active]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Metric columns must be aligned one-dimensional arrays")
    matrix = np.column_stack(arrays)
    valid = np.isfinite(matrix).all(axis=1)
    scores = np.full(len(matrix), -np.inf)
    coefficients = np.asarray(list(active.values()))
    scores[valid] = matrix[valid] @ coefficients / np.abs(coefficients).sum()
    if not valid.any():
        raise ValueError("No row has finite values for all active weighted metrics")
    return scores


def selectivity(
    on_target: str,
    off_targets: Sequence[str],
    penalty: float = 2.5,
    metric: str = "i_ptm",
    *,
    higher_is_better: bool = True,
) -> Objective:
    """On-target utility minus a nonnegative penalty on the strongest off-target utility."""
    off_targets = tuple(off_targets)
    if (
        not off_targets
        or on_target in off_targets
        or len(set(off_targets)) != len(off_targets)
    ):
        raise ValueError("Provide distinct on-target and off-target names")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Off-target penalty must be finite and nonnegative")
    sign = 1 if higher_is_better else -1

    def f(metrics):
        on = sign * _value(metrics[on_target], metric)
        off = max(sign * _value(metrics[target], metric) for target in off_targets)
        return float(on - penalty * off)

    f.description = f"{on_target} {metric} selectivity (penalty={penalty:g})"
    return f


def describe(objective: Objective) -> str:
    """Human-readable objective label."""
    return getattr(
        objective, "description", getattr(objective, "__name__", "custom objective")
    )
