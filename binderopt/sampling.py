"""Sequence mutation guided by residue-level structural confidence."""

from __future__ import annotations

import logging
import random
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RESIDUE_SAMPLING = {
    #: Guided candidates per parent -- mutations placed by the structure.
    "combined_mutations_per_seq": 70,
    "random_mutations_per_seq": 25,
    #: A further handful of larger random jumps, spread over the parents.
    "random_mutations_total": 20,
    "residue_weights": {
        "interface_score": 0.5,
        "packing_score": 0.15,
        "confidence_score": 0.3,
        "helix_score": 0.05,
    },
    #: Mutations per guided candidate. The upper bound is annealed down to the lower one
    #: across the run; see ``combined_score_mutations``.
    "min_mutations": 1,
    "max_mutations": 4,
    "badness_sharpening": 4.0,
}


class MultiMetricGuidedSampler:
    """Generates mutation candidates from top parent sequences."""

    def __init__(
        self,
        amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
        residue_sampling: Dict = None,
        top_n_for_sampling: int = 30,
    ):
        self.amino_acids = list(amino_acids)
        self.residue_sampling = {**DEFAULT_RESIDUE_SAMPLING, **(residue_sampling or {})}
        self.residue_weights = self.residue_sampling["residue_weights"]
        self.min_mutations = self.residue_sampling["min_mutations"]
        self.max_mutations = self.residue_sampling["max_mutations"]
        self.sharpening = self.residue_sampling["badness_sharpening"]
        self.top_n_for_sampling = top_n_for_sampling

    def _get_mutation_probability_decay(
        self, current_iter: int, max_iter: int
    ) -> float:
        """Annealing position: 1.0 on the first round, 0.0 on the last."""
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
        """Mutation candidates for one round: structure-guided where possible, plus random."""
        if top_residue_metrics and len(top_residue_metrics) != len(top_sequences):
            raise ValueError(
                "top_residue_metrics has %d entries for %d sequences; pass None for any "
                "sequence without per-residue metrics so the pairing stays positional"
                % (len(top_residue_metrics), len(top_sequences))
            )

        n_top = min(len(top_sequences), self.top_n_for_sampling)
        candidates = set()
        sampling_config = self.residue_sampling

        decay = self._get_mutation_probability_decay(current_iteration, max_iterations)
        lo, hi = self.min_mutations, self.max_mutations
        top_muts = max(lo, min(hi, int(round(lo + (hi - lo) * decay))))

        # --- Structure-guided mutations, for the parents that have residue metrics ---
        paired = [
            (s, m)
            for s, m in zip(top_sequences[:n_top], top_residue_metrics or [])
            if m
        ]
        if paired:
            guided = self.combined_score_mutations(
                [s for s, _ in paired],
                [m for _, m in paired],
                sampling_config["combined_mutations_per_seq"],
                decay,
            )
            candidates.update(guided)
            logger.info(
                "Structure-guided: %d candidates from %d of %d parents "
                "(<=%d mutations each)",
                len(guided),
                len(paired),
                n_top,
                top_muts,
            )
        else:
            logger.info(
                "No per-residue metrics available; mutating uniformly at random"
            )

        # --- Random mutations from all top parents, annealed on the same schedule ---
        for seq in top_sequences[:n_top]:
            for _ in range(sampling_config["random_mutations_per_seq"]):
                n_muts = min(random.randint(lo, top_muts), len(seq))
                positions = random.sample(range(len(seq)), n_muts)
                mutant = list(seq)
                for pos in positions:
                    mutant[pos] = random.choice(
                        [aa for aa in self.amino_acids if aa != seq[pos]]
                    )
                candidates.add("".join(mutant))

        candidates.update(
            self._random_mutations(
                top_sequences[:n_top], n_total=sampling_config["random_mutations_total"]
            )
        )
        candidates.difference_update(top_sequences)  # a parent is not a proposal

        logger.info("Generated %d unique mutation candidates total", len(candidates))
        return sorted(candidates)

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

        if "pae_inter" in metrics:
            interface_score = array("pae_inter") / 31.0
            if "con_inter" in metrics:
                con_badness = np.exp(-array("con_inter") / 2.5)
                interface_score = interface_score * 0.7 + con_badness * 0.3
            combined_scores += interface_score * self.residue_weights["interface_score"]

        if "plddt" in metrics:
            confidence_score = 1 - array("plddt")
            combined_scores += (
                confidence_score * self.residue_weights["confidence_score"]
            )

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

    def combined_score_mutations(
        self,
        sequences: List[str],
        metrics_list: List[Dict[str, np.ndarray]],
        n_per_seq: int,
        decay_multiplier: float,
    ) -> List[str]:
        """Mutate the residues the structure says are worst, annealing the step size."""
        candidates = []
        lo, hi = self.min_mutations, self.max_mutations
        # Width of the mutation-count window for this round, annealed hi -> lo.
        top = max(lo, min(hi, int(round(lo + (hi - lo) * decay_multiplier))))

        for seq, metrics in zip(sequences, metrics_list):
            badness_scores = (
                self._calculate_residue_badness_scores(metrics) ** self.sharpening
            )
            total = badness_scores.sum()
            mutation_probs = (badness_scores / total) if total > 0 else None

            for _ in range(n_per_seq):
                n_muts = min(random.randint(lo, top), len(seq))
                if mutation_probs is None:
                    # No usable structure signal: uniform positions, still annealed.
                    positions = random.sample(range(len(seq)), n_muts)
                else:
                    positions = np.random.choice(
                        len(seq), size=n_muts, replace=False, p=mutation_probs
                    )
                mutant = list(seq)
                for pos in positions:
                    mutant[pos] = random.choice(
                        [aa for aa in self.amino_acids if aa != seq[pos]]
                    )
                candidates.append("".join(mutant))

        return candidates

    def _random_mutations(self, sequences: List[str], n_total: int) -> List[str]:
        """Pure random exploration mutations, spread evenly across `sequences`."""
        candidates = []
        n_per_seq = n_total // max(len(sequences), 1)
        for seq in sequences:
            for _ in range(n_per_seq):
                n_muts = min(
                    random.randint(
                        self.min_mutations, max(self.min_mutations, self.max_mutations)
                    ),
                    len(seq),
                )
                positions = random.sample(range(len(seq)), n_muts)
                mutant = list(seq)
                for pos in positions:
                    mutant[pos] = random.choice(
                        [aa for aa in self.amino_acids if aa != seq[pos]]
                    )
                candidates.append("".join(mutant))
        return candidates
