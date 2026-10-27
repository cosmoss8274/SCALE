"""Evaluate a batch of sequences with any oracle.

`optimizer.AF3Evaluator` re-derives metrics from AlphaFold 3's output files, which ties it to
that one predictor. This evaluator takes the metrics the oracle already returned, so the loop
works with any backend. A failed prediction is recorded and skipped rather than aborting the
round, because losing one design should not lose the round's other four.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import MetricContractError, validate

logger = logging.getLogger(__name__)


def sequence_id(sequence: str, iteration: Optional[int] = None) -> str:
    """A name for one prediction, derived from the sequence rather than its position.

    Counting calls instead would be enough within a single process, but these names become
    directory and file names that outlive it. A resumed run skips the rounds it has already
    done, so a counter starting from zero in the new process hands round 3 the names round 0
    used -- pointing the oracle at a populated output directory and overwriting the earlier
    structures. Hashing the sequence makes a name mean the same design in every session, so
    resuming is idempotent instead of destructive.
    """
    digest = hashlib.sha256(sequence.encode()).hexdigest()[:12]
    return digest if iteration is None else "r%02d_%s" % (iteration, digest)


class OracleEvaluator:
    """Call ``oracle.predict`` over a list of sequences and collect normalised metrics."""

    def __init__(self, oracle, structure_dir: Optional[str] = None, retries: int = 1):
        self.oracle = oracle
        self.structure_dir = Path(structure_dir) if structure_dir else None
        self.retries = retries
        self.failures: List[Dict] = []
        self.current_iteration = 0

    def evaluate(self, sequences: List[str], iteration: Optional[int] = None) -> List[Dict]:
        """Predict every sequence, skipping the ones that fail.

        ``iteration`` labels the round in the returned records and in each prediction's
        name. Pass the real round number when resuming a run; it defaults to counting
        calls, which is correct only for a run that starts from nothing.
        """
        self.current_iteration = (self.current_iteration + 1 if iteration is None
                                  else iteration)
        results: List[Dict] = []
        for seq in sequences:
            seq_id = sequence_id(seq, self.current_iteration)
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
