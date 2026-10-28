"""AlphaFold 3 oracle and confidence translation."""

from __future__ import annotations

from typing import Dict, Tuple

from ..af3_interface import AlphaFold3Interface, StructureEvaluator
from .metrics import normalise
from .registry import register_oracle

RECEPTOR_INDEX, BINDER_INDEX = 0, 1


def _min_cross_chain_pae(chain_pair_pae_min, binder_index=1):
    """Minimum PAE between the binder chain and other chains, in both directions."""
    try:
        rows = list(chain_pair_pae_min)
        values = [
            float(rows[binder_index][i]) for i in range(len(rows)) if i != binder_index
        ]
        values += [
            float(rows[i][binder_index]) for i in range(len(rows)) if i != binder_index
        ]
        return min(values) if values else float("nan")
    except (TypeError, ValueError, IndexError):
        return float("nan")


def _binder_ptm(chain_ptm, binder_index=1):
    """Binder-chain pTM, or NaN if unavailable."""
    try:
        return float(list(chain_ptm)[binder_index])
    except (TypeError, ValueError, IndexError):
        return float("nan")


def translate_confidences(native, binder_chain="B"):
    """Translate AF3 chain-specific confidences into the oracle metric names."""
    chain_ids = native.get("chain_ids")
    binder = chain_ids.index(binder_chain) if chain_ids is not None else BINDER_INDEX
    pair = native.get("chain_pair_iptm")
    iptm = native.get("iptm")
    if pair is not None:
        values = [float(pair[binder][i]) for i in range(len(pair)) if i != binder]
        values += [float(pair[i][binder]) for i in range(len(pair)) if i != binder]
        iptm = sum(values) / len(values) if values else None
    return dict(
        i_ptm=iptm,
        ptm_binder=_binder_ptm(native.get("chain_ptm"), binder),
        min_pae_interaction=_min_cross_chain_pae(
            native.get("chain_pair_pae_min"), binder
        ),
    )


class AF3Oracle(StructureEvaluator):
    """Wrap :class:`AlphaFold3Interface` and translate its output into the contract."""

    def __init__(
        self, interface: AlphaFold3Interface, oracle_version: str = "alphafold3"
    ):
        self.interface = interface
        self.oracle_version = oracle_version

    def predict(self, binder_sequence: str, **kwargs) -> Tuple[float, Dict]:
        confidence, native = self.interface.predict(
            binder_sequence=binder_sequence, **kwargs
        )
        native = dict(native)

        metrics = dict(native)
        binder_chain = getattr(self.interface, "binder_chain", "B")
        metrics.update(translate_confidences(native, binder_chain))
        if native.get("structure_file") and native.get("json_file"):
            from ..losses import ColabDesignLossCalculator

            residue = ColabDesignLossCalculator(
                native["structure_file"], native["json_file"], binder_chain=binder_chain
            ).get_residue_level_metrics()
            from .metrics import validate_residue_metrics

            metrics["residue_metrics"] = validate_residue_metrics(
                residue, expect_length=len(binder_sequence)
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
