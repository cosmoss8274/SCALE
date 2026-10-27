"""Check that structure-guided sampling is guided, and annealed.

Every property here replaces one that silently did not hold. The guided branch was never
reached, the annealing schedule resolved to a single step size, a 0-1 pLDDT was divided by
100 a second time, and prediction names were derived from a call counter that restarted on
resume. None of those failed loudly; each produced plausible output that simply carried no
information, so each gets a test that fails when the information disappears again.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderopt.oracles.evaluator import sequence_id
from binderopt.oracles.metrics import MetricContractError, validate_residue_metrics
from binderopt.sampling import MultiMetricGuidedSampler

LENGTH = 60
PARENTS = ["ACDEFGHIKLMNPQRSTVWY" * 3, "WYVTSRQPNMLKIHGFEDCA" * 3]


def residue_metrics(bad_positions, length=LENGTH):
    """Per-residue metrics in which only ``bad_positions`` look wrong."""
    bad = np.zeros(length)
    bad[list(bad_positions)] = 1.0
    return {
        "pae_inter": list(6.0 + 20.0 * bad),
        "pae_intra": list(3.0 + 10.0 * bad),
        "plddt": list(0.9 - 0.5 * bad),          # 0-1, as the contract requires
    }


def mutated_positions(candidates, parent):
    for candidate in candidates:
        for i, (a, b) in enumerate(zip(candidate, parent)):
            if a != b:
                yield i


def test_mutations_concentrate_on_the_worst_residues():
    """The point of the whole mechanism: bad residues are mutated more than good ones."""
    sampler = MultiMetricGuidedSampler()
    worst = set(range(0, LENGTH, 5))             # a fifth of the sequence
    candidates = sampler.combined_score_mutations(
        [PARENTS[0]], [residue_metrics(worst)], 400, 1.0)

    hit = [p for p in mutated_positions(candidates, PARENTS[0])]
    share = sum(1 for p in hit if p in worst) / len(hit)
    assert share > 0.5, (
        "only %.0f%% of mutations landed on the worst fifth of residues; uniform "
        "sampling would give 20%%, so the guidance is not biting" % (100 * share))


def test_position_choice_ignores_a_constant_scale():
    """Badness becomes a probability by dividing by its sum, so a multiplier cancels.

    This is why the annealing schedule cannot be implemented by scaling badness, which is
    what it did originally: the position distribution came out bit-for-bit identical at
    every decay value. Asserting the cancellation keeps anyone from reintroducing it as
    an annealing step.
    """
    sampler = MultiMetricGuidedSampler()
    scores = sampler._calculate_residue_badness_scores(residue_metrics({3, 17, 41}))
    assert scores.sum() > 0

    probabilities = scores ** sampler.sharpening
    probabilities = probabilities / probabilities.sum()
    scaled = (0.05 * scores) ** sampler.sharpening
    scaled = scaled / scaled.sum()
    assert np.allclose(probabilities, scaled), (
        "scaling badness changed the position distribution, so some code is relying on "
        "its absolute magnitude")


def test_step_size_anneals_across_rounds():
    """Early rounds take bigger steps than late ones, and the last round is the smallest."""
    sampler = MultiMetricGuidedSampler()
    metrics = [residue_metrics({1, 2, 3})] * len(PARENTS)

    means = []
    for rnd in range(1, 6):
        candidates = sampler.generate_candidates(
            PARENTS, metrics, current_iteration=rnd, max_iterations=5)
        counts = [min(sum(a != b for a, b in zip(c, p)) for p in PARENTS)
                  for c in candidates]
        means.append(float(np.mean(counts)))

    assert means[0] > means[-1], "step size did not shrink: %s" % means
    assert means[0] > 2.0, "first round should explore with several mutations: %.2f" % means[0]
    assert means[-1] < 1.5, "last round should refine one at a time: %.2f" % means[-1]


def test_anneal_endpoints():
    sampler = MultiMetricGuidedSampler()
    assert sampler._get_mutation_probability_decay(1, 5) == pytest.approx(1.0)
    assert sampler._get_mutation_probability_decay(5, 5) == pytest.approx(0.0)
    assert sampler._get_mutation_probability_decay(1, 1) == pytest.approx(1.0)


def test_missing_residue_metrics_fall_back_to_random():
    """A design the oracle could not profile still gets mutated, just not guided."""
    sampler = MultiMetricGuidedSampler()
    candidates = sampler.generate_candidates(PARENTS, [], current_iteration=1,
                                             max_iterations=3)
    assert candidates, "no candidates at all without residue metrics"

    partial = sampler.generate_candidates(
        PARENTS, [residue_metrics({1, 2}), None], current_iteration=1, max_iterations=3)
    assert len(partial) > len(candidates)


def test_mismatched_metrics_raise_rather_than_misalign():
    """Pairing is positional, so a short list would guide one design by another's structure."""
    sampler = MultiMetricGuidedSampler()
    with pytest.raises(ValueError, match="positional"):
        sampler.generate_candidates(PARENTS, [residue_metrics({1})],
                                    current_iteration=1, max_iterations=3)


def test_parents_are_not_proposed_as_their_own_offspring():
    sampler = MultiMetricGuidedSampler()
    candidates = sampler.generate_candidates(
        PARENTS, [residue_metrics({1, 2})] * len(PARENTS),
        current_iteration=1, max_iterations=3)
    assert not set(candidates) & set(PARENTS)


def test_plddt_on_the_wrong_scale_is_rejected():
    """A 0-100 pLDDT must fail at the adapter, not become a near-constant term."""
    with pytest.raises(MetricContractError, match="0-1"):
        validate_residue_metrics({"plddt": [92.0, 71.0, 44.0]}, where="test")
    validate_residue_metrics({"plddt": [0.92, 0.71, 0.44]}, where="test")


def test_plddt_discriminates_on_the_contract_scale():
    """The confidence term must vary across residues; dividing 0-1 by 100 flattened it."""
    sampler = MultiMetricGuidedSampler()
    scores = sampler._calculate_residue_badness_scores(
        {"plddt": [0.95, 0.60, 0.25]})
    assert scores.max() - scores.min() > 0.1, (
        "pLDDT contributed a near-constant %s; it is being rescaled somewhere" % scores)
    assert scores[2] > scores[0], "the least confident residue should be the worst"


def test_sequence_id_is_stable_across_sessions():
    """Resuming must not rename a design, or it overwrites an earlier one's output."""
    seq = "MKLVPQRSTACDEFGHIKLM"
    assert sequence_id(seq, 3) == sequence_id(seq, 3)
    assert sequence_id(seq, 0) != sequence_id(seq, 3)
    assert sequence_id(seq) != sequence_id(seq + "A")


def test_sequence_ids_do_not_collide_when_resuming():
    """The failure this replaced: a counter restarting at 0 reused round 0's names."""
    first_session = {sequence_id(s, 0) for s in ("AAAC", "AAAG")}
    resumed = {sequence_id(s, 3) for s in ("AAAT", "AACA")}
    assert not first_session & resumed


def main() -> int:
    import traceback
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("%-52s ok" % name)
        except Exception:                        # noqa: BLE001 - reported below
            failures.append(name)
            print("%-52s FAIL" % name)
            traceback.print_exc()
    print("\n%s" % ("ALL CHECKS PASSED" if not failures
                    else "FAILED: " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
