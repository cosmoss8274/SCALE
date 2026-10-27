"""Mutation candidate generation guided by residue-level structure metrics.

`generate_candidates` is the method actually used by the optimizer loop; the
old file's docstring already noted it had been simplified to pure random
mutation (the residue-badness-guided helpers below it were effectively
dead code in that file's call graph, but are kept here -- intact and
de-duplicated -- since they implement a real, documented alternative
sampling strategy that a future run could opt into).
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RESIDUE_SAMPLING = {
    #: Guided candidates per parent -- mutations placed by the structure.
    "combined_mutations_per_seq": 70,
    #: Random candidates per parent, for diversity the structure would not suggest.
    #: Guided must outnumber random, or the guidance is diluted below the point of
    #: having it: at 70 guided against 160 random, the worst fifth of residues received
    #: a *smaller* share of the pool's mutations than uniform sampling would give them.
    "random_mutations_per_seq": 25,
    #: A further handful of larger random jumps, spread over the parents.
    "random_mutations_total": 20,
    #: Kept so older configs still load. The four names below suggested four strategies;
    #: the code only ever summed them into one count of uniformly random mutations, which
    #: ``random_mutations_per_seq`` now states directly. Setting them has no effect.
    "interface_mutations_per_seq": None,
    "packing_mutations_per_seq": None,
    "plddt_mutations_per_seq": None,
    "helix_mutations_per_seq": None,
    "residue_weights": {
        "interface_score": 0.5,
        "packing_score": 0.15,
        "confidence_score": 0.3,
        "helix_score": 0.05,
    },
    #: Unused. The schedule is linear in the round number (see
    #: ``_get_mutation_probability_decay``); this key is kept only so that an existing
    #: config setting it still loads. Setting it has no effect.
    "progressive_decay_rate": 3.0,
    #: Mutations per guided candidate. The upper bound is annealed down to the lower one
    #: across the run; see ``combined_score_mutations``.
    "min_mutations": 1,
    "max_mutations": 4,
    #: Exponent applied to residue badness before it becomes a probability over positions.
    #: The weighted badness of a real complex spans only about 1.7x between its best and
    #: worst residue, and sampling proportionally to that is nearly uniform -- guidance in
    #: name only. Raising it to the fourth power gives the worst fifth of residues ~37% of
    #: mutations against 20% for uniform, a 9x probability ratio, while still leaving every
    #: position reachable. Higher exponents concentrate harder but start to collapse the
    #: pool onto a handful of sites.
    "badness_sharpening": 4.0,
}


class MultiMetricGuidedSampler:
    """Generates mutation candidates from top parent sequences."""

    def __init__(self, amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
                 residue_sampling: Dict = None, top_n_for_sampling: int = 30):
        self.amino_acids = list(amino_acids)
        self.residue_sampling = {**DEFAULT_RESIDUE_SAMPLING, **(residue_sampling or {})}
        self.residue_weights = self.residue_sampling["residue_weights"]
        self.min_mutations = self.residue_sampling["min_mutations"]
        self.max_mutations = self.residue_sampling["max_mutations"]
        self.sharpening = self.residue_sampling["badness_sharpening"]
        self.top_n_for_sampling = top_n_for_sampling

    def _get_mutation_probability_decay(self, current_iter: int, max_iter: int) -> float:
        """Annealing position: 1.0 on the first round, 0.0 on the last.

        The caller interpolates the mutation step size across this, so the schedule is
        linear in the round number. An exponential was used here originally, but it falls
        fastest exactly where the schedule should still be exploring: over five rounds
        ``exp(-3r/5)`` reaches the smallest step by round three and then repeats it, so
        three of the five rounds behave identically. A straight line spends each round
        somewhere different and lands on the minimum only at the end -- for five rounds,
        4, 3, 2, 2, 1 mutations.
        """
        if max_iter <= 1:
            return 1.0
        return float(np.clip(1.0 - (current_iter - 1) / (max_iter - 1), 0.0, 1.0))

    def generate_candidates(
        self,
        top_sequences: List[str],
        top_residue_metrics: List[Dict[str, np.ndarray]],
        current_iteration: int = 0,
        max_iterations: int = 50,
    ) -> List[str]:
        """Mutation candidates for one round: structure-guided where possible, plus random.

        Guided candidates concentrate mutations on residues the structure says are worst
        -- poorly placed against the receptor, poorly packed, or low-confidence. Random
        candidates keep the pool from collapsing onto whatever the current structure
        happens to dislike. Both are only proposals: the surrogate ranks them and the
        oracle measures a handful.

        ``top_residue_metrics`` is positional: entry *i* belongs to ``top_sequences[i]``,
        and ``None`` marks a sequence whose per-residue metrics are unavailable (an older
        checkpoint, or an oracle that does not provide them). Those sequences still get
        random mutations. A shorter list than ``top_sequences`` is an error rather than a
        silent truncation, because quietly pairing one design's structure with another's
        sequence would guide mutations to the wrong positions.
        """
        if top_residue_metrics and len(top_residue_metrics) != len(top_sequences):
            raise ValueError(
                "top_residue_metrics has %d entries for %d sequences; pass None for any "
                "sequence without per-residue metrics so the pairing stays positional"
                % (len(top_residue_metrics), len(top_sequences)))

        n_top = min(len(top_sequences), self.top_n_for_sampling)
        candidates = set()
        sampling_config = self.residue_sampling

        decay = self._get_mutation_probability_decay(current_iteration, max_iterations)
        lo, hi = self.min_mutations, self.max_mutations
        top_muts = max(lo, min(hi, int(round(lo + (hi - lo) * decay))))

        # --- Structure-guided mutations, for the parents that have residue metrics ---
        paired = [(s, m) for s, m in zip(top_sequences[:n_top], top_residue_metrics or [])
                  if m]
        if paired:
            guided = self.combined_score_mutations(
                [s for s, _ in paired], [m for _, m in paired],
                sampling_config["combined_mutations_per_seq"], decay)
            candidates.update(guided)
            logger.info("Structure-guided: %d candidates from %d of %d parents "
                        "(<=%d mutations each)",
                        len(guided), len(paired), n_top, top_muts)
        else:
            logger.info("No per-residue metrics available; mutating uniformly at random")

        # --- Random mutations from all top parents, annealed on the same schedule ---
        for seq in top_sequences[:n_top]:
            for _ in range(sampling_config["random_mutations_per_seq"]):
                n_muts = min(random.randint(lo, top_muts), len(seq))
                positions = random.sample(range(len(seq)), n_muts)
                mutant = list(seq)
                for pos in positions:
                    mutant[pos] = random.choice(self.amino_acids)
                candidates.add("".join(mutant))

        candidates.update(
            self._random_mutations(top_sequences[:n_top], n_total=sampling_config["random_mutations_total"])
        )
        candidates.difference_update(top_sequences)       # a parent is not a proposal

        logger.info("Generated %d unique mutation candidates total", len(candidates))
        return list(candidates)

    def _calculate_residue_badness_scores(
        self, metrics: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Combined per-residue 'badness': how much each position wants changing.

        Only relative values matter -- the caller normalises this into a probability over
        positions -- so the weights set how much each structural concern counts, not an
        absolute scale.
        """
        def array(key):
            return np.asarray(metrics[key], dtype=float)

        seq_len = len(next(iter(metrics.values())))
        combined_scores = np.zeros(seq_len)

        # Each concern contributes whatever data it has. Contact counts (con_inter,
        # con_intra) are AlphaFold-era extras that Boltz-2 does not write, so they refine
        # a term when present rather than gating it: requiring them would silence the
        # interface term -- half the total weight -- for the default oracle.
        if "pae_inter" in metrics:
            interface_score = array("pae_inter") / 31.0
            if "con_inter" in metrics:
                con_badness = np.exp(-array("con_inter") / 2.5)
                interface_score = interface_score * 0.7 + con_badness * 0.3
            combined_scores += interface_score * self.residue_weights["interface_score"]

        if "plddt" in metrics:
            # 0-1 by the metric contract (binderopt/oracles/metrics.py), normalised by the
            # adapter. Dividing a 0-1 pLDDT by 100 again would leave every residue at
            # ~0.99 -- a term that looks active and guides nothing.
            confidence_score = 1 - array("plddt")
            combined_scores += confidence_score * self.residue_weights["confidence_score"]

        if "pae_intra" in metrics:
            packing_score = array("pae_intra") / 31.0
            if "con_intra" in metrics and array("con_intra").max() > 0:
                con_norm = (8 - array("con_intra")) / 8.0
                packing_score = packing_score * 0.6 + con_norm * 0.4
            combined_scores += packing_score * self.residue_weights["packing_score"]

        if "helix" in metrics:
            helix_score = 1.0 - array("helix")
            combined_scores += helix_score * self.residue_weights["helix_score"]

        return np.clip(combined_scores, 0, None)

    def _identify_dominant_problem(self, metrics: Dict[str, np.ndarray], pos: int) -> str:
        """Identify which structural concern is worst at a given residue position."""
        problems = {}
        if "pae_inter" in metrics:
            problems["interface"] = metrics["pae_inter"][pos] / 31.0
        if "pae_intra" in metrics:
            problems["packing"] = metrics["pae_intra"][pos] / 31.0
        if "plddt" in metrics:
            problems["confidence"] = 1.0 - float(metrics["plddt"][pos])
        if "helix" in metrics:
            problems["helix"] = 1.0 - metrics["helix"][pos]
        return max(problems.items(), key=lambda x: x[1])[0] if problems else "random"

    def combined_score_mutations(
        self,
        sequences: List[str],
        metrics_list: List[Dict[str, np.ndarray]],
        n_per_seq: int,
        decay_multiplier: float,
    ) -> List[str]:
        """Mutate the residues the structure says are worst, annealing the step size.

        Two things are decided per candidate, and ``decay_multiplier`` governs only the
        second:

        *Where* comes from the per-residue badness, raised to ``badness_sharpening`` and
        normalised into a probability over positions. The exponent is what makes the
        guidance bite: raw weighted badness spans only ~1.7x between a complex's best and
        worst residue, and sampling proportionally to that is within a few percent of
        uniform.

        The decay deliberately does not enter here: ``p = badness / badness.sum()`` is
        scale-free, so multiplying badness by any constant leaves ``p`` bit-for-bit
        identical. Applying the decay before normalising -- as this function did until the
        schedule was measured -- is a no-op dressed as an annealing step.

        *How many* is the schedule. Early rounds take several mutations at once to cover
        ground; later rounds take one or two, because by then the parents are good and a
        large jump is more likely to break a design than improve it. The step size is
        interpolated by ``decay_multiplier`` (1 at the start, →0 at the end) rather than
        compared against fixed thresholds: thresholds on a decaying quantity collapse to
        their lowest branch almost immediately, which is how the three documented regimes
        turned out to be one in practice.
        """
        candidates = []
        lo, hi = self.min_mutations, self.max_mutations
        # Width of the mutation-count window for this round, annealed hi -> lo.
        top = max(lo, min(hi, int(round(lo + (hi - lo) * decay_multiplier))))

        for seq, metrics in zip(sequences, metrics_list):
            badness_scores = self._calculate_residue_badness_scores(metrics) ** self.sharpening
            total = badness_scores.sum()
            mutation_probs = (badness_scores / total) if total > 0 else None

            for _ in range(n_per_seq):
                n_muts = min(random.randint(lo, top), len(seq))
                if mutation_probs is None:
                    # No usable structure signal: uniform positions, still annealed.
                    positions = random.sample(range(len(seq)), n_muts)
                else:
                    positions = np.random.choice(len(seq), size=n_muts, replace=False,
                                                 p=mutation_probs)
                mutant = list(seq)
                for pos in positions:
                    mutant[pos] = random.choice(self.amino_acids)
                candidates.append("".join(mutant))

        return candidates

    def _random_mutations(self, sequences: List[str], n_total: int) -> List[str]:
        """Pure random exploration mutations, spread evenly across `sequences`."""
        candidates = []
        n_per_seq = n_total // max(len(sequences), 1)
        for seq in sequences:
            for _ in range(n_per_seq):
                n_muts = random.randint(2, 4)
                positions = random.sample(range(len(seq)), n_muts)
                mutant = list(seq)
                for pos in positions:
                    mutant[pos] = random.choice(self.amino_acids)
                candidates.append("".join(mutant))
        return candidates
