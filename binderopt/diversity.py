"""Sequence-distance constraints for the notebook's substitution-only search.

Distances are mismatches divided by length. Unequal lengths are rejected rather
than compared with a silently truncated zip. Constraints never relax to fill a batch.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import random


def mismatches(a: str, b: str) -> int:
    if not a or len(a) != len(b):
        raise ValueError("Substitution search requires nonempty, equal-length sequences")
    return sum(x != y for x, y in zip(a, b))


def nearest(sequence, references):
    if not references:
        raise ValueError("A nonempty reference pool is required")
    return min(mismatches(sequence, s) for s in references)


@dataclass(frozen=True)
class DiversityPolicy:
    min_pool_fraction: float = 0.10
    min_batch_fraction: float = 0.10
    min_history_fraction: float = 0.05
    max_per_founder: int = 2
    min_founders: int = 3
    adaptive_parent_steps: bool = True

    def __post_init__(self):
        if any(not 0 <= v <= 1 for v in (
                self.min_pool_fraction, self.min_batch_fraction, self.min_history_fraction)):
            raise ValueError("Distance fractions must lie in [0, 1]")
        if self.max_per_founder < 1 or self.min_founders < 1:
            raise ValueError("Founder limits must be positive integers")

    def counts(self, length):
        return tuple(math.ceil(length * f - 1e-12) for f in (
            self.min_pool_fraction, self.min_batch_fraction, self.min_history_fraction))

    def to_dict(self):
        return asdict(self)


def choose_parents(history, objective, n=5, reference_pool=None, policy=None):
    """Best eligible new representative per founder, with seed fallbacks.

    When novelty is a requirement, an initial seed cannot beat a feasible new
    parent merely by having a higher unconstrained score. Initial seeds remain
    fallbacks for founders without a measured novel descendant.
    """
    if reference_pool is not None and policy is not None and policy.adaptive_parent_steps:
        eligible, fallback = {}, {}
        for h in sorted(history, key=lambda h: (-objective(h), h["sequence_id"])):
            founder = h.get("founder_id", h["sequence_id"])
            fallback.setdefault(founder, h)
            minimum = policy.counts(len(h["sequence"]))[0]
            if h["round"] > 0 and nearest(h["sequence"], reference_pool) >= minimum:
                eligible.setdefault(founder, h)
        representatives = list(eligible.values())
        representatives.extend(h for f, h in fallback.items() if f not in eligible)
        return representatives[:n]
    best = {}
    for h in sorted(history, key=lambda h: (-objective(h), h["sequence_id"])):
        founder = h.get("founder_id", h["sequence_id"])
        best.setdefault(founder, h)
    return list(best.values())[:n]


def propose_diverse(sampler, parents, reference_pool, history, policy,
                    current_round, n_rounds, limit=600):
    """Generate balanced proposals, preserving the actual generating parent.

    Each parent gets the same proposal allowance. Invalid/previously measured
    proposals are discarded before ranking, including ones near unmeasured seeds.
    """
    if not parents or limit < len(parents):
        raise ValueError("Need parents and at least one proposal slot per parent")
    known = {h["sequence"] for h in history}
    prior_new = [h["sequence"] for h in history if h["round"] > 0]
    minimum, _, prior_minimum = policy.counts(len(parents[0]["sequence"]))
    per_parent = limit // len(parents)
    proposals, seen = [], set()
    for parent in parents:
        parent_sampler = sampler
        if policy.adaptive_parent_steps:
            # Novelty is global; the step size is local. Keep exploring from a
            # seed, then refine a novel design without stepping near the pool.
            parent_is_novel = nearest(parent["sequence"], reference_pool) >= minimum
            lower = max(1, prior_minimum if parent_is_novel else minimum)
            parent_sampler = type(sampler)(
                amino_acids="".join(sampler.amino_acids),
                residue_sampling={**sampler.residue_sampling,
                                  "min_mutations": lower, "max_mutations": lower + 4},
                top_n_for_sampling=sampler.top_n_for_sampling)
        accepted = []
        for _ in range(12):
            generated = sorted(parent_sampler.generate_candidates(
                [parent["sequence"]], [parent.get("residue_metrics") or None],
                current_iteration=current_round, max_iterations=n_rounds))
            random.shuffle(generated)
            for seq in generated:
                if seq in known or seq in seen:
                    continue
                distance = nearest(seq, reference_pool)
                if distance < minimum:
                    continue
                if prior_new and nearest(seq, prior_new) < prior_minimum:
                    continue
                accepted.append({
                    "sequence": seq, "parent_id": parent["sequence_id"],
                    "founder_id": parent.get("founder_id", parent["sequence_id"]),
                    "mutations_from_parent": mismatches(seq, parent["sequence"]),
                    "mutations_from_pool": distance,
                    "identity_to_pool": 1 - distance / len(seq),
                })
                seen.add(seq)
                if len(accepted) == per_parent:
                    break
            if len(accepted) == per_parent:
                break
        proposals.extend(accepted)
    if not proposals:
        raise RuntimeError("No proposals satisfy the sequence-distance constraints")
    return proposals


def select_diverse(proposals, scores, n, policy):
    """Score-ranked feasible batch with pairwise distance and founder limits.

    Reserve one slot per required founder before filling the remaining slots.
    This enforces multiple starting sequences even when every top score descends
    from one seed. No oracle information from the proposed batch is used.
    """
    if len(scores) != len(proposals) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("One finite surrogate score is required per proposal")
    required = min(policy.min_founders, n)
    if math.ceil(n / policy.max_per_founder) > len({p["founder_id"] for p in proposals}):
        raise RuntimeError("Too few founders for the requested batch cap")
    order = sorted(range(len(proposals)), key=lambda i: (-float(scores[i]), proposals[i]["sequence"]))
    selected, counts = [], Counter()
    batch_distance = policy.counts(len(proposals[0]["sequence"]))[1]

    def fits(i):
        p = proposals[i]
        return (counts[p["founder_id"]] < policy.max_per_founder and
                all(mismatches(p["sequence"], proposals[j]["sequence"]) >= batch_distance
                    for j in selected))

    for i in order:
        if not counts[proposals[i]["founder_id"]] and fits(i):
            selected.append(i)
            counts[proposals[i]["founder_id"]] += 1
            if len(counts) >= required:
                break
    if len(counts) < required:
        raise RuntimeError("Not enough distance-separated founders; constraints were not relaxed")
    for i in order:
        if len(selected) == n:
            break
        if i not in selected and fits(i):
            selected.append(i)
            counts[proposals[i]["founder_id"]] += 1
    if len(selected) != n:
        raise RuntimeError("Cannot fill the batch under the distance constraints; no predictions spent")
    return [{**proposals[i], "surrogate_score": float(scores[i])} for i in selected]
