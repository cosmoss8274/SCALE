"""The metric contract every oracle must satisfy.

An oracle is swappable only if the loop can read its output without knowing which
predictor produced it. That requires one vocabulary, defined here, and an adapter per
backend that translates its native field names into it -- AlphaFold 3 calls the interface
confidence ``iptm``, Boltz-2 calls it ``iptm`` too but nests the per-pair value under
``pair_chains_iptm``, and a third predictor will call it something else again.

    ================================  =========================================
    ``i_ptm``                         interface pTM for the receptor-binder pair
    ``ptm_binder``                    pTM of the binder chain alone
    ``min_pae_interaction``           smallest cross-chain PAE, in Angstrom
    ``structure_file``                path to the predicted complex, or None
    ``oracle`` / ``oracle_version``   provenance, stamped by the adapter
    ================================  =========================================

Everything else an adapter returns is passed through untouched, so a backend can expose
extra metrics (Boltz-2's ``complex_plddt``, AF3's ``ranking_score``) without the loop
needing to know about them. Only the names above may be relied on by the loop, an
objective, or a figure.

**A shared vocabulary is not a shared scale.** An ``i_ptm`` of 0.80 from Boltz-2 does not
denote what 0.80 from AlphaFold 3 denotes; thresholds calibrated on one do not transfer.
That is why every record is stamped with its oracle, and why :func:`check_comparable`
exists -- pooling two oracles into one figure or filtering them with one threshold is a
mistake the data should be able to catch.

Missing is NaN, never a stand-in. A substituted number is indistinguishable from a
measured one once it reaches a CSV.

Per-residue metrics
-------------------

An adapter may also return ``residue_metrics``: one array per binder residue, used by the
sampler to decide *where* to mutate rather than mutating uniformly at random. It is
optional -- a backend that cannot supply it returns nothing and the sampler falls back to
random positions -- but the names and scales below are fixed, because the sampler combines
them into one score and cannot ask which predictor they came from.

    ================  ==================================================================
    ``plddt``         per-residue confidence, **0-1, higher is better**
    ``pae_inter``     mean PAE in Angstrom to the receptor, higher is worse
    ``pae_intra``     mean PAE in Angstrom within the binder, higher is worse
    ================  ==================================================================

The 0-1 pLDDT convention is the adapter's to enforce. Boltz-2 writes 0-1 natively and
AlphaFold 3 writes 0-100; normalising at the adapter -- not in the consumer -- is the same
rule the scalar metrics follow, and it is the reason a scale error cannot reach the sampler
as a silently near-constant term.
"""
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


def normalise(metrics: Dict, *, oracle: str, oracle_version: str,
              structure_file: Optional[str] = None) -> Dict:
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
            % (where, ", ".join(missing), ", ".join(REQUIRED + PROVENANCE)))
    for key in REQUIRED:
        try:
            float(metrics[key])
        except (TypeError, ValueError):
            raise MetricContractError(
                "%s returned %r for %s, which is not a number" % (where, metrics[key], key))
    return metrics


def validate_residue_metrics(residue: Dict, *, where: str = "oracle",
                             expect_length: Optional[int] = None) -> Dict:
    """Raise unless per-residue arrays are the right length and on the documented scale.

    The pLDDT check is the point of this function. A predictor writing 0-100 into a
    consumer expecting 0-1 does not fail -- it yields a near-constant term that silently
    stops contributing, which is exactly how the guided sampler came to be unguided. A
    loud error at the adapter boundary is cheap; a term that quietly does nothing is not.
    """
    for key, array in residue.items():
        values = [float(v) for v in array]
        if expect_length is not None and len(values) != expect_length:
            raise MetricContractError(
                "%s returned %d values for residue metric %r but the binder is %d residues"
                % (where, len(values), key, expect_length))
        finite = [v for v in values if not math.isnan(v)]
        if key == "plddt" and finite and max(finite) > 1.0:
            raise MetricContractError(
                "%s returned pLDDT up to %.1f; the contract is 0-1, so divide by 100 in the "
                "adapter. See binderopt/oracles/metrics.py." % (where, max(finite)))
    return residue


def oracles_present(records: Iterable[Dict]) -> List[str]:
    """The distinct oracles that produced ``records``."""
    return sorted({r.get("oracle", "unknown") for r in records})


def check_comparable(records: Iterable[Dict], *, what: str = "these records") -> None:
    """Raise if ``records`` mix oracles, whose metrics are not on a common scale."""
    found = oracles_present(records)
    if len(found) > 1:
        raise MetricContractError(
            "%s mix predictions from %s. Confidence metrics are calibrated per predictor, "
            "so they must not be pooled in one figure or filtered with one threshold. "
            "Split by the 'oracle' field first." % (what, " and ".join(found)))


def describe(metrics: Dict) -> str:
    """One line for logs: the contract metrics plus provenance."""
    def fmt(key):
        value = metrics.get(key, float("nan"))
        return "n/a" if isinstance(value, float) and math.isnan(value) else "%.3f" % value
    return "i_ptm=%s ptm_binder=%s min_pae=%s [%s]" % (
        fmt("i_ptm"), fmt("ptm_binder"), fmt("min_pae_interaction"),
        metrics.get("oracle", "?"))
