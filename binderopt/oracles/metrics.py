"""Predictor metric names, units, scales and validation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

#: Names the loop may rely on. An adapter that cannot supply one returns NaN.
REQUIRED = ("i_ptm", "ptm_binder", "min_pae_interaction")

#: Provenance every adapter stamps on.
PROVENANCE = ("oracle", "oracle_version")

#: Optional, by convention: the path to the predicted complex, or None.
STRUCTURE = "structure_file"

#: Optional: per-binder-residue arrays for guided sampling. See the module docstring for
#: the scale of each; ``RESIDUE_METRICS`` is the key they live under.
RESIDUE_METRICS = "residue_metrics"
RESIDUE_KEYS = ("plddt", "pae_inter", "pae_intra")


class MetricContractError(ValueError):
    """An oracle returned metrics the loop cannot read."""


def normalise(
    metrics: Dict,
    *,
    oracle: str,
    oracle_version: str,
    structure_file: Optional[str] = None,
) -> Dict:
    """Stamp provenance and fill any absent required metric with NaN.

    Adapters call this as the last step of ``predict``. It does not rename anything --
    translation is the adapter's job, since only it knows its backend's vocabulary.
    """
    out = dict(metrics)
    for key in REQUIRED:
        value = out.get(key)
        out[key] = float("nan") if value is None else float(value)
    out.setdefault(STRUCTURE, structure_file)
    out["oracle"] = oracle
    out["oracle_version"] = oracle_version
    return out


def validate(metrics: Dict, *, where: str = "oracle") -> Dict:
    """Raise unless ``metrics`` satisfies the contract. Returns it unchanged.

    Used by the evaluator so that a backend which quietly drops a metric fails at the
    point of the mistake rather than hundreds of predictions later in a plotting call.
    """
    missing = [k for k in REQUIRED + PROVENANCE if k not in metrics]
    if missing:
        raise MetricContractError(
            "%s returned no %s; every oracle must supply %s (NaN where unmeasured). "
            "See binderopt/oracles/metrics.py."
            % (where, ", ".join(missing), ", ".join(REQUIRED + PROVENANCE))
        )
    for key in REQUIRED:
        try:
            value = float(metrics[key])
        except (TypeError, ValueError) as exc:
            raise MetricContractError(f"{where}: {key} must be numeric") from exc
        if math.isinf(value) or (
            not math.isnan(value)
            and (value < 0 or (key != "min_pae_interaction" and value > 1))
        ):
            raise MetricContractError(f"{where}: invalid {key} value {value}")
    return metrics


def validate_residue_metrics(
    residue: Dict, *, where: str = "oracle", expect_length: Optional[int] = None
) -> Dict:
    """Raise unless per-residue arrays are the right length and on the documented scale."""
    for key, array in residue.items():
        try:
            values = [float(v) for v in array]
        except (TypeError, ValueError) as exc:
            raise MetricContractError(f"{where}: invalid residue array {key}") from exc
        if expect_length is not None and len(values) != expect_length:
            raise MetricContractError(
                f"{where}: residue metric {key} has {len(values)} values; expected {expect_length}"
            )
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise MetricContractError(
                f"{where}: {key} must contain finite nonnegative values"
            )
        if key in {"plddt", "helix"} and any(v > 1 for v in values):
            raise MetricContractError(f"{where}: {key} must use the 0-1 scale")
    return residue


def oracles_present(records: Iterable[Dict]) -> List[str]:
    """The distinct oracles that produced ``records``."""
    return sorted({r.get("oracle", "unknown") for r in records})


def check_comparable(records: Iterable[Dict], *, what: str = "these records") -> None:
    """Raise if ``records`` mix oracles, whose metrics are not on a common scale."""
    found = oracles_present(records)
    if len(found) > 1:
        raise MetricContractError(
            f"{what} mix predictor backends ({', '.join(found)}); compare each backend separately"
        )


def describe(metrics: Dict) -> str:
    """One line for logs: the contract metrics plus provenance."""

    def fmt(key):
        value = metrics.get(key, float("nan"))
        return (
            "n/a" if isinstance(value, float) and math.isnan(value) else "%.3f" % value
        )

    return "i_ptm=%s ptm_binder=%s min_pae=%s [%s]" % (
        fmt("i_ptm"),
        fmt("ptm_binder"),
        fmt("min_pae_interaction"),
        metrics.get("oracle", "?"),
    )
