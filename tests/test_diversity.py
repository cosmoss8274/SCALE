"""Hard novelty/diversity constraints and real parent provenance (no GPU required)."""

from collections import Counter
from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderopt.diversity import (
    DiversityPolicy,
    choose_parents,
    mismatches,
    nearest,
    propose_diverse,
    select_diverse,
)
from binderopt.sampling import MultiMetricGuidedSampler


def test_distance_rejects_truncated_and_empty_comparisons():
    with pytest.raises(ValueError):
        mismatches("AAA", "AA")
    with pytest.raises(ValueError):
        mismatches("", "")
    assert mismatches("AAAA", "ACDA") == 2
    assert DiversityPolicy().counts(81) == (9, 9, 5)


def test_higher_scoring_clones_cannot_fill_the_batch():
    parents = ["A" * 40, "C" * 40, "D" * 40]
    proposals = []
    for i, s in enumerate(parents):
        for n in range(5):
            proposals.append(
                {"sequence": s[:n] + "EFGH" + s[n + 4 :], "founder_id": str(i)}
            )
    # First founder has all the highest scores, but can receive at most two slots.
    selected = select_diverse(proposals, list(range(15, 0, -1)), 5, DiversityPolicy())
    counts = Counter(p["founder_id"] for p in selected)
    assert len(counts) == 3 and max(counts.values()) <= 2
    assert (
        min(
            mismatches(a["sequence"], b["sequence"])
            for a, b in combinations(selected, 2)
        )
        >= 4
    )


def test_infeasible_constraints_do_not_silently_relax():
    proposals = [{"sequence": "A" * 40, "founder_id": str(i)} for i in range(3)]
    with pytest.raises(RuntimeError, match="not relaxed"):
        select_diverse(proposals, [3, 2, 1], 3, DiversityPolicy())
    with pytest.raises(ValueError):
        select_diverse(proposals, [float("nan")] * 3, 3, DiversityPolicy())


def test_parents_keep_different_founders_and_their_own_metrics():
    history = [
        dict(sequence_id=str(i), founder_id=f, score=s, residue_metrics={"plddt": [i]})
        for i, (f, s) in enumerate([("a", 0.9), ("a", 0.95), ("b", 0.8), ("c", 0.7)])
    ]
    parents = choose_parents(history, lambda h: h["score"], 3)
    assert [p["sequence_id"] for p in parents] == ["1", "2", "3"]
    assert parents[0]["residue_metrics"]["plddt"] == [1]


def test_proposals_avoid_entire_pool_and_prior_rounds_and_record_true_parent():
    random.seed(4)
    np.random.seed(4)
    seqs = [
        "ACDEFGHIKLMNPQRSTVWY" * 4,
        "YWVTSRQPNMLKIHGFEDCA" * 4,
        "CVDEFGHIKLMNPQRSTVWY" * 4,
    ]
    parents = [
        dict(
            sequence=s,
            sequence_id=str(i),
            founder_id=str(i),
            round=0,
            residue_metrics={"plddt": [0.8] * 80},
        )
        for i, s in enumerate(seqs)
    ]
    # An unmeasured starting-pool entry is still protected by the novelty floor.
    reference = seqs + [seqs[0][:8] + "AAAAAAAA" + seqs[0][16:]]
    prior = [dict(parents[0], round=1, sequence="F" * 8 + seqs[0][8:])]
    sampler = MultiMetricGuidedSampler(
        residue_sampling={"min_mutations": 8, "max_mutations": 12}
    )
    policy = DiversityPolicy()
    candidates = propose_diverse(
        sampler, parents, reference, parents + prior, policy, 1, 5, 90
    )
    assert len(candidates) == 90
    for c in candidates:
        assert nearest(c["sequence"], reference) >= 8
        assert nearest(c["sequence"], [prior[0]["sequence"]]) >= 4
        p = next(p for p in parents if p["sequence_id"] == c["parent_id"])
        assert mismatches(c["sequence"], p["sequence"]) == c["mutations_from_parent"]
        assert 8 <= c["mutations_from_parent"] <= 12
    batch = select_diverse(candidates, np.linspace(1, 0, len(candidates)), 5, policy)
    assert len({p["founder_id"] for p in batch}) >= 3
    assert (
        min(mismatches(a["sequence"], b["sequence"]) for a, b in combinations(batch, 2))
        >= 8
    )


def test_novel_parents_are_not_displaced_by_high_scoring_initial_seeds():
    seed = "A" * 80
    novel = "C" * 5 + seed[5:]
    history = [
        dict(sequence=seed, sequence_id="seed", founder_id="seed", round=0, score=0.95),
        dict(sequence=novel, sequence_id="new", founder_id="seed", round=1, score=0.75),
        dict(
            sequence="D" * 80,
            sequence_id="other",
            founder_id="other",
            round=0,
            score=0.9,
        ),
    ]
    policy = DiversityPolicy(
        min_pool_fraction=0.0625, min_batch_fraction=0.0625, min_history_fraction=0.0375
    )
    parents = choose_parents(history, lambda h: h["score"], 2, [seed, "D" * 80], policy)
    assert [p["sequence_id"] for p in parents] == ["new", "other"]


def test_novel_parent_refines_locally_while_global_distance_stays_active():
    random.seed(7)
    np.random.seed(7)
    seed = "A" * 80
    novel = "C" * 5 + seed[5:]
    parent = dict(
        sequence=novel,
        sequence_id="new",
        founder_id="seed",
        round=1,
        residue_metrics={"plddt": [0.8] * 80},
    )
    initial = dict(sequence=seed, sequence_id="seed", founder_id="seed", round=0)
    policy = DiversityPolicy(
        min_pool_fraction=0.0625, min_batch_fraction=0.0625, min_history_fraction=0.0375
    )
    sampler = MultiMetricGuidedSampler(
        residue_sampling={"min_mutations": 5, "max_mutations": 9}
    )
    proposals = propose_diverse(
        sampler, [parent], [seed], [initial, parent], policy, 5, 5, 50
    )
    assert len(proposals) == 50
    assert min(p["mutations_from_parent"] for p in proposals) == 3
    assert all(
        p["mutations_from_parent"] >= 3 and p["mutations_from_pool"] >= 5
        for p in proposals
    )


def test_final_portfolio_preserves_diversity_and_actual_score_labels():
    from binderopt.demo_reporting import diverse_portfolio

    history = []
    for i, base in enumerate(["A" * 40, "C" * 40, "D" * 40]):
        for n in range(5):
            history.append(
                dict(
                    sequence=base[:n] + "EFGH" + base[n + 4 :],
                    sequence_id=f"{i}_{n}",
                    founder_id=str(i),
                    round=n + 1,
                    i_ptm=0.99 - i * 0.1 - n * 0.001,
                    mutations_from_pool=4,
                    surrogate_score=0.123,
                )
            )
    portfolio = diverse_portfolio(history, DiversityPolicy(), 5)
    assert len(portfolio) == 5 and len({x["founder_id"] for x in portfolio}) == 3
    assert max(Counter(x["founder_id"] for x in portfolio).values()) <= 2
    assert all(x["surrogate_score"] == 0.123 for x in portfolio)
    assert (
        min(
            mismatches(a["sequence"], b["sequence"])
            for a, b in combinations(portfolio, 2)
        )
        >= 4
    )


def test_disabled_novel_refinement_preserves_wide_search_parent_rule():
    seed = "A" * 80
    novel = "C" * 8 + seed[8:]
    h = [
        dict(sequence=seed, sequence_id="seed", founder_id="seed", round=0, score=0.95),
        dict(sequence=novel, sequence_id="new", founder_id="seed", round=1, score=0.75),
    ]
    p = choose_parents(
        h, lambda h: h["score"], 1, [seed], DiversityPolicy(adaptive_parent_steps=False)
    )
    assert p[0]["sequence_id"] == "seed"
