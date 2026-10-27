"""Plumbing test for the optimisation loop used in notebooks/SCALE_demo.ipynb.

A stub oracle stands in for a structure predictor so the wiring can be checked without a GPU:
evaluator -> surrogate training -> candidate generation -> surrogate ranking -> bookkeeping.

THE METRICS HERE ARE SYNTHETIC. They are a deterministic function of the sequence and mean
nothing biologically; they exist so that failures in the plumbing surface as failures. No
number produced by this file is a result, and none is reported anywhere. The notebook runs
the same code against a real oracle.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderopt.af3_interface import StructureEvaluator
from binderopt.models import MultiHeadSurrogate
from binderopt.objectives import describe, maximise
from binderopt.oracles.evaluator import OracleEvaluator
from binderopt.pools import read_csv
from binderopt.sampling import MultiMetricGuidedSampler

N_SEED, N_ROUNDS, N_PER_ROUND, N_CANDIDATES = 6, 2, 3, 40


class StubOracle(StructureEvaluator):
    """Deterministic pseudo-metrics from the sequence hash. Not a predictor."""

    def predict(self, binder_sequence, **kwargs):
        h = hashlib.sha256(binder_sequence.encode()).digest()
        i_ptm = 0.60 + 0.30 * (h[0] / 255.0)
        return i_ptm, {"i_ptm": round(i_ptm, 4),
                       "ptm_binder": round(0.5 + 0.4 * (h[1] / 255.0), 4),
                       "min_pae_interaction": round(0.8 + 4.0 * (h[2] / 255.0), 4),
                       "structure_file": None,
                       "oracle": "stub", "oracle_version": "test-only"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pool, _ = read_csv(root / "data" / "example_pool_trka.csv", expect_length=80)
    rng = np.random.default_rng(0)
    seeds = [pool[i] for i in rng.choice(len(pool), size=N_SEED, replace=False)]

    objective = maximise("i_ptm")
    evaluator = OracleEvaluator(StubOracle())
    measured = evaluator.evaluate(seeds)
    assert len(measured) == N_SEED, "evaluator lost sequences"
    assert not evaluator.failures
    print("round 0: measured %d designs | objective %s" % (len(measured), describe(objective)))

    surrogate = MultiHeadSurrogate(
        esm2_model="esm2_t33_650M_UR50D",
        rf_params={"n_estimators": 100, "random_state": 0, "n_jobs": -1},
        metric_weights={"i_ptm": 1.0}, calculate_rmsd=False,
    )
    sampler = MultiMetricGuidedSampler(top_n_for_sampling=5)
    history = [dict(round=0, **m) for m in measured]

    for rnd in range(1, N_ROUNDS + 1):
        seqs = [h["sequence"] for h in history]
        surrogate.train(seqs, {"i_ptm": [h["i_ptm"] for h in history]})
        best = sorted(history, key=objective, reverse=True)[:5]
        cands = sampler.generate_candidates([h["sequence"] for h in best], [],
                                            current_iteration=rnd, max_iterations=N_ROUNDS)
        cands = [c for c in dict.fromkeys(cands) if c not in set(seqs)][:N_CANDIDATES]
        assert cands, "sampler produced no new candidates"
        proposals, predicted = surrogate.top_n_selection(cands, n=N_PER_ROUND)
        assert len(proposals) == N_PER_ROUND == len(predicted)
        assert all(len(p) == len(seqs[0]) for p in proposals), "proposal changed length"
        results = evaluator.evaluate(proposals)
        history += [dict(round=rnd, **m) for m in results]
        print("round %d: %d candidates -> measured %d | predicted %.3f-%.3f"
              % (rnd, len(cands), len(results), predicted.min(), predicted.max()))

    assert len(history) == N_SEED + N_ROUNDS * N_PER_ROUND, "history bookkeeping wrong"
    assert len({h["sequence"] for h in history}) == len(history), "duplicate measured"
    rounds = sorted({h["round"] for h in history})
    running = [max(objective(h) for h in history if h["round"] <= r) for r in rounds]
    assert all(b >= a - 1e-12 for a, b in zip(running, running[1:])), "best-so-far decreased"
    assert all(m["oracle"] == "stub" for m in history), "provenance stamp lost"
    print("\nALL CHECKS PASSED | %d designs over %d rounds, best-so-far monotone"
          % (len(history), len(rounds)))
    return 0

@pytest.mark.slow
def test_loop_smoke():
    """pytest entry point. Downloads ESM-2 and runs three real rounds (~7 min on CPU).

    Deselect it with ``-m "not slow"``.
    """
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
