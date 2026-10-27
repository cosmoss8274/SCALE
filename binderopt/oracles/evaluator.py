"""Evaluate a batch of sequences with any oracle.

`optimizer.AF3Evaluator` re-derives metrics from AlphaFold 3's output files, which ties it to
that one predictor. This evaluator takes the metrics the oracle already returned, so the loop
works with any backend. A failed prediction is recorded and skipped rather than aborting the
round, because losing one design should not lose the round's other four.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import MetricContractError, validate

logger = logging.getLogger(__name__)


class OracleEvaluator:
    """Call ``oracle.predict`` over a list of sequences and collect normalised metrics."""

    def __init__(self, oracle, structure_dir: Optional[str] = None, retries: int = 1):
        self.oracle = oracle
        self.structure_dir = Path(structure_dir) if structure_dir else None
        self.retries = retries
        self.failures: List[Dict] = []
        self.current_iteration = 0

    def evaluate(self, sequences: List[str]) -> List[Dict]:
        self.current_iteration += 1
        results: List[Dict] = []
        for i, seq in enumerate(sequences):
            seq_id = "iter%03d_seq%03d" % (self.current_iteration, i)
            last = None
            for attempt in range(self.retries + 1):
                try:
                    kwargs = {"sequence_id": seq_id}
                    if self.structure_dir is not None:
                        kwargs.update(save_structure=True, structure_dir=str(self.structure_dir))
                    _, metrics = self.oracle.predict(binder_sequence=seq, **kwargs)
                    # Fail at the adapter that broke the contract, not 300 predictions
                    # later in a plotting call that cannot say which backend was wrong.
                    validate(metrics, where=type(self.oracle).__name__)
                    record = dict(metrics)
                    record.update(sequence=seq, sequence_id=seq_id,
                                  iteration=self.current_iteration)
                    results.append(record)
                    break
                except MetricContractError:
                    # Not a failed prediction: the adapter is wrong, and it will be wrong
                    # for every sequence. Retrying and recording it as a per-design
                    # failure would bury a programming error under N warnings.
                    raise
                except Exception as exc:              # noqa: BLE001 - reported, not hidden
                    last = exc
                    logger.warning("attempt %d failed for %s: %s", attempt + 1, seq_id, exc)
            else:
                self.failures.append({"sequence_id": seq_id, "sequence": seq, "error": repr(last)})
        if self.failures:
            logger.warning("%d of %d sequences failed this round",
                           len(self.failures), len(sequences))
        return results
