"""AlphaFold 3 backend.

AlphaFold 3 names its confidences differently from the loop's vocabulary, so this adapter
translates rather than passes through:

    ==========================================  =========================
    AlphaFold 3 summary field                   contract name
    ==========================================  =========================
    ``iptm``                                    ``i_ptm``
    ``chain_ptm[binder]``                       ``ptm_binder``
    ``min(chain_pair_pae_min`` off-diagonal)    ``min_pae_interaction``
    ==========================================  =========================

``ranking_score`` and the raw arrays are passed through under their own names for anyone
who wants them, but the loop reads only the contract names. Before this translation
existed, selecting ``af3`` raised KeyError in the first objective call, because the loop,
the objectives and the figures were all written against the Boltz-2 adapter's names --
the backend-agnostic claim was not true.

Chain order follows the convention used throughout this repository: the receptor is chain
0, the binder chain 1.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

from ..af3_interface import AlphaFold3Interface, StructureEvaluator
from .metrics import normalise
from .registry import register_oracle

RECEPTOR_INDEX, BINDER_INDEX = 0, 1


def _min_cross_chain_pae(chain_pair_pae_min) -> float:
    """Smallest off-diagonal entry of AF3's per-chain-pair PAE minima.

    AF3 reports a square matrix over chains; the diagonal is each chain against itself,
    so the interface value is the smallest entry off it. Returns NaN when the field is
    absent or malformed rather than a stand-in, per the metric contract.
    """
    try:
        rows = list(chain_pair_pae_min)
        off = [float(rows[i][j])
               for i in range(len(rows)) for j in range(len(rows[i])) if i != j]
        return min(off) if off else float("nan")
    except (TypeError, ValueError, IndexError):
        return float("nan")


def _binder_ptm(chain_ptm) -> float:
    """pTM of the binder chain, or NaN when the array is absent or too short."""
    try:
        return float(list(chain_ptm)[BINDER_INDEX])
    except (TypeError, ValueError, IndexError):
        return float("nan")


class AF3Oracle(StructureEvaluator):
    """Wrap :class:`AlphaFold3Interface` and translate its output into the contract."""

    def __init__(self, interface: AlphaFold3Interface, oracle_version: str = "alphafold3"):
        self.interface = interface
        self.oracle_version = oracle_version

    def predict(self, binder_sequence: str, **kwargs) -> Tuple[float, Dict]:
        confidence, native = self.interface.predict(binder_sequence=binder_sequence, **kwargs)
        native = dict(native)

        metrics = dict(native)
        metrics.update(
            i_ptm=native.get("iptm"),
            ptm_binder=_binder_ptm(native.get("chain_ptm")),
            min_pae_interaction=_min_cross_chain_pae(native.get("chain_pair_pae_min")),
        )
        return confidence, normalise(
            metrics,
            oracle="af3",
            oracle_version=self.oracle_version,
            structure_file=native.get("structure_file"),
        )

    def cleanup(self) -> None:
        self.interface.cleanup()


@register_oracle("af3")
def _build_af3(interface: AlphaFold3Interface = None, **kwargs) -> AF3Oracle:
    if interface is None:
        interface = AlphaFold3Interface(**kwargs)
    return AF3Oracle(interface)
