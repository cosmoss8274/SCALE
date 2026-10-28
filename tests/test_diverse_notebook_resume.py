"""Actual notebook search cells with synthetic scores and simulated interruption.

Checks bookkeeping/constraints only; these scores are not biological results.
"""

import csv, hashlib, json, sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderopt.diversity import DiversityPolicy, mismatches, nearest
from binderopt.oracles.evaluator import OracleEvaluator


class SyntheticOracle:
    def __init__(self):
        self.calls = 0

    def predict(self, binder_sequence, **kwargs):
        self.calls += 1
        v = 0.4 + hashlib.sha256(binder_sequence.encode()).digest()[0] / 512
        return v, dict(
            i_ptm=v,
            ptm_binder=0.8,
            min_pae_interaction=2.0,
            oracle="synthetic-test-only",
            oracle_version="test",
            residue_metrics={"plddt": [0.8] * len(binder_sequence)},
        )


class SyntheticSurrogate:
    def __init__(self, **kwargs):
        pass

    def train(self, *args):
        pass

    def predict(self, sequences):
        return {
            "i_ptm": np.array(
                [hashlib.sha256(s.encode()).digest()[1] / 256 for s in sequences]
            )
        }

    def predict_combined_score(self, predictions):
        return predictions["i_ptm"]


def test_actual_notebook_resumes_saved_batch_with_exact_budget(tmp_path, monkeypatch):
    module = ModuleType("binderopt.models")
    module.MultiHeadSurrogate = SyntheticSurrogate
    monkeypatch.setitem(sys.modules, "binderopt.models", module)
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/SCALE_demo_colab.ipynb").read_text())
    codes = [
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
        if c["cell_type"] == "code"
    ]
    initial = next(s for s in codes if s.startswith("#@title 5."))
    loop = next(s for s in codes if s.startswith("#@title 6."))
    pool = [
        r["sequence"] for r in csv.DictReader(open(root / "data/example_pool_trka.csv"))
    ]
    oracle = SyntheticOracle()
    evaluator = OracleEvaluator(oracle)
    ns = dict(
        RUN=tmp_path,
        json=json,
        np=np,
        policy=DiversityPolicy(),
        pool=pool,
        seed_designs=pool[:10],
        N_SEED=10,
        N_ROUNDS=5,
        N_PER_ROUND=5,
        N_CANDIDATES=100,
        evaluator=evaluator,
        objective=lambda h: h["i_ptm"],
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    exec(initial, ns)
    real = evaluator.evaluate

    def interrupt(sequences, **kwargs):
        if oracle.calls == 12:
            raise InterruptedError("simulated disconnect")
        return real(sequences, **kwargs)

    evaluator.evaluate = interrupt
    with pytest.raises(InterruptedError):
        exec(loop, ns)
    assert oracle.calls == 12
    selected = [
        p["sequence"]
        for p in json.loads((tmp_path / "round_01_plan.json").read_text())["selected"]
    ]
    evaluator.evaluate = real
    exec(initial, ns)
    exec(loop, ns)
    h = ns["history"]
    assert oracle.calls == len(h) == len({p["sequence"] for p in h}) == 35
    assert [p["sequence"] for p in h if p["round"] == 1] == selected
    for rnd in range(1, 6):
        group = [p for p in h if p["round"] == rnd]
        assert len(group) == 5 and len({p["founder_id"] for p in group}) >= 3
        for i, a in enumerate(group):
            assert nearest(a["sequence"], pool) >= 8
            prior = [p["sequence"] for p in h if 0 < p["round"] < rnd]
            if prior:
                assert nearest(a["sequence"], prior) >= 4
            parent = next(p for p in h if p["sequence_id"] == a["parent_id"])
            assert (
                parent["round"] < rnd
                and mismatches(a["sequence"], parent["sequence"])
                == a["mutations_from_parent"]
            )
            assert sum(p["founder_id"] == a["founder_id"] for p in group) <= 2
            for b in group[i + 1 :]:
                assert mismatches(a["sequence"], b["sequence"]) >= 8
    ns["policy"] = DiversityPolicy(min_pool_fraction=0.2)
    with pytest.raises(RuntimeError, match="settings changed"):
        exec(initial, ns)
    assert oracle.calls == 35
