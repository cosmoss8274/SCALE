"""What the loop is trying to maximise.

An objective is any callable ``f(metrics: dict) -> float`` that the loop maximises, where
``metrics`` is what the oracle returned for one design. Three are provided; anything else is
a plain function:

    objective = maximise("i_ptm")
    objective = minimise("min_pae_interaction")
    objective = selectivity("TRKA", ["TRKB", "TRKC"], penalty=2.5)

    def objective(m):                      # or write your own
        return m["i_ptm"] - 0.1 * m["min_pae_interaction"]

Keeping the objective outside the surrogate matters: the surrogate predicts *metrics*, and
the objective combines them. Changing what you are optimising for therefore does not
invalidate a trained surrogate, and two runs with different objectives remain comparable on
the underlying metrics.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence

Objective = Callable[[Dict], float]


def maximise(metric: str) -> Objective:
    """Maximise one metric, e.g. interface pTM."""
    def f(metrics: Dict) -> float:
        return float(metrics[metric])
    f.__doc__ = "maximise %s" % metric
    f.description = "maximise %s" % metric
    return f


def minimise(metric: str) -> Objective:
    """Minimise one metric, e.g. interface PAE. Returned as a negated value to maximise."""
    def f(metrics: Dict) -> float:
        return -float(metrics[metric])
    f.description = "minimise %s" % metric
    return f


def weighted(weights: Dict[str, float]) -> Objective:
    """Weighted sum of metrics; use negative weights for quantities to be reduced."""
    def f(metrics: Dict) -> float:
        return float(sum(w * metrics[k] for k, w in weights.items()))
    f.description = "weighted sum " + ", ".join("%+g*%s" % (w, k) for k, w in weights.items())
    return f


def selectivity(on_target: str, off_targets: Sequence[str], penalty: float = 2.5,
                metric: str = "i_ptm") -> Objective:
    """On-target metric minus ``penalty`` times the WORST off-target metric.

    The maximum over off-targets, rather than their sum or mean, is deliberate: with a sum a
    design can score well by shedding one off-target while keeping affinity for another. The
    penalty is linear rather than hinged because a threshold penalty leaves the gradient at
    zero for most candidates and lets designs just under the threshold through.

    Expects per-receptor metrics, i.e. ``metrics[target][metric]``.
    """
    def f(metrics: Dict) -> float:
        worst = max(float(metrics[t][metric]) for t in off_targets)
        return float(metrics[on_target][metric]) - penalty * worst
    f.description = "%s %s - %g * max(%s)" % (on_target, metric, penalty, ", ".join(off_targets))
    return f


def describe(objective: Objective) -> str:
    return getattr(objective, "description", getattr(objective, "__name__", "custom objective"))
