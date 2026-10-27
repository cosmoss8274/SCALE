"""Oracle that replays evaluations already on disk.

Used for two things: reproducing a published run without a GPU (the demo notebook), and
regression-testing the loop's bookkeeping against a run whose outcome is known. It is
not a predictor - asked for a sequence it has never seen, it raises rather than guessing,
because a silent miss would let the loop report progress it never measured.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..af3_interface import StructureEvaluator
from .registry import register_oracle

#: Metric keys the SCALE loop consumes, in the AlphaFold 3 convention.
METRIC_KEYS = ("i_ptm", "ptm_binder", "min_pae_interaction")


class ReplayOracle(StructureEvaluator):
    """Look up recorded metrics by binder sequence.

    Args:
        records: iterable of dicts with a ``sequence`` key and a ``metrics`` mapping.
            Records whose ``status`` is present and not ``"complete"`` are ignored.
        target: which receptor's metrics to return when a record holds several, e.g.
            ``"TRKA"``. ``None`` means the record's ``metrics`` is already flat.
        confidence_key: which metric the loop's scalar confidence should mirror.
    """

    def __init__(self, records: Iterable[Dict], target: Optional[str] = None,
                 confidence_key: str = "i_ptm", oracle_version: str = "recorded"):
        self.target = target
        self.confidence_key = confidence_key
        self.oracle_version = oracle_version
        self._table: Dict[str, Dict] = {}
        for rec in records:
            if rec.get("status", "complete") != "complete":
                continue
            metrics = rec["metrics"][target] if target else rec["metrics"]
            self._table[rec["sequence"]] = dict(metrics)
        if not self._table:
            raise ValueError("no complete records supplied")

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "ReplayOracle":
        """Load a run record file (``{"records": [...]}`` or a bare list)."""
        payload = json.loads(Path(path).read_text())
        records = payload["records"] if isinstance(payload, dict) else payload
        return cls(records, **kwargs)

    def __len__(self) -> int:
        return len(self._table)

    @property
    def sequences(self) -> List[str]:
        return list(self._table)

    def predict(self, binder_sequence: str, **kwargs) -> Tuple[float, Dict]:
        try:
            metrics = dict(self._table[binder_sequence])
        except KeyError:
            raise KeyError(
                "sequence not in the replay table (%d entries). A replay oracle cannot "
                "evaluate new designs; point the loop at a real predictor instead."
                % len(self._table)
            ) from None
        metrics.update(oracle="replay", oracle_version=self.oracle_version,
                       target=self.target)
        return float(metrics.get(self.confidence_key, 0.0)), metrics


@register_oracle("replay")
def _build_replay(records=None, path=None, **kwargs) -> ReplayOracle:
    if (records is None) == (path is None):
        raise ValueError("pass exactly one of records= or path=")
    return ReplayOracle.from_json(path, **kwargs) if path else ReplayOracle(records, **kwargs)
