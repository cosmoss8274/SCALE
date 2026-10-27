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
    "combined_mutations_per_seq": 70,
    "interface_mutations_per_seq": 15,
    "packing_mutations_per_seq": 10,
    "plddt_mutations_per_seq": 10,
    "helix_mutations_per_seq": 5,
    "random_mutations_total": 20,
    "residue_weights": {
        "interface_score": 0.5,
        "packing_score": 0.15,
        "confidence_score": 0.3,
        "helix_score": 0.05,
    },
    "progressive_decay_rate": 3.0,
}


class MultiMetricGuidedSampler:
    """Generates mutation candidates from top parent sequences."""

    def __init__(self, amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
                 residue_sampling: Dict = None, top_n_for_sampling: int = 30):
        self.amino_acids = list(amino_acids)
        self.residue_sampling = {**DEFAULT_RESIDUE_SAMPLING, **(residue_sampling or {})}
        self.residue_weights = self.residue_sampling["residue_weights"]
        self.decay_rate = self.residue_sampling["progressive_decay_rate"]
        self.top_n_for_sampling = top_n_for_sampling

    def _get_mutation_probability_decay(self, current_iter: int, max_iter: int) -> float:
        """Exponential decay multiplier (1.0 at start, decaying toward 0 across iterations)."""
        if max_iter <= 1:
            return 1.0
        progress = current_iter / max_iter
        return float(np.exp(-self.decay_rate * progress))

    def generate_candidates(
        self,
        top_sequences: List[str],
        top_residue_metrics: List[Dict[str, np.ndarray]],
        current_iteration: int = 0,
        max_iterations: int = 50,
    ) -> List[str]:
        """Generate mutation candidates: structure-guided (when residue metrics available) + random.

        Structure-guided mutations bias toward residues with poor PAE/pLDDT/contacts.
        Random mutations provide diversity across the full sequence.
        Both pools are combined and later filtered by the surrogate.
        """
        n_top = min(len(top_sequences), self.top_n_for_sampling)
        candidates = set()
        sampling_config = self.residue_sampling

        decay = self._get_mutation_probability_decay(current_iteration, max_iterations)

        # --- Structure-guided mutations (sequences that have resolved structures) ---
        guided_seqs = top_sequences[:len(top_residue_metrics)]
        guided_metrics = top_residue_metrics
        if guided_seqs:
            n_guided = sampling_config["combined_mutations_per_seq"]
            guided = self.combined_score_mutations(guided_seqs, guided_metrics, n_guided, decay)
            candidates.update(guided)
            logger.info("Structure-guided: %d candidates from %d seqs", len(guided), len(guided_seqs))

        # --- Random mutations from all top parents ---
        total_random_per_seq = (
            sampling_config["interface_mutations_per_seq"]
            + sampling_config["packing_mutations_per_seq"]
            + sampling_config["plddt_mutations_per_seq"]
            + sampling_config["helix_mutations_per_seq"]
        )
        for seq in top_sequences[:n_top]:
            for _ in range(total_random_per_seq):
                n_muts = random.randint(1, 3)
                positions = random.sample(range(len(seq)), n_muts)
                mutant = list(seq)
                for pos in positions:
                    mutant[pos] = random.choice(self.amino_acids)
                candidates.add("".join(mutant))

        candidates.update(
            self._random_mutations(top_sequences[:n_top], n_total=sampling_config["random_mutations_total"])
        )

        logger.info("Generated %d unique mutation candidates total", len(candidates))
        return list(candidates)

    def _calculate_residue_badness_scores(
        self, metrics: Dict[str, np.ndarray], decay_multiplier: float = 1.0
    ) -> np.ndarray:
        """Combined per-residue 'badness' score, used by the residue-guided sampling strategy."""
        seq_len = len(next(iter(metrics.values())))
        combined_scores = np.zeros(seq_len)

        if "pae_inter" in metrics and "con_inter" in metrics:
            pae_norm = metrics["pae_inter"] / 31.0
            con_badness = np.exp(-metrics["con_inter"] / 2.5)
            interface_score = pae_norm * 0.7 + con_badness * 0.3
            combined_scores += interface_score * self.residue_weights["interface_score"]

        if "plddt" in metrics:
            confidence_score = 1 - (metrics["plddt"] / 100.0)
            combined_scores += confidence_score * self.residue_weights["confidence_score"]

        if "pae_intra" in metrics and "con_intra" in metrics:
            pae_norm = metrics["pae_intra"] / 31.0
            if metrics["con_intra"].max() == 0:
                packing_score = pae_norm
            else:
                con_norm = (8 - metrics["con_intra"]) / 8.0
                packing_score = pae_norm * 0.6 + con_norm * 0.4
            combined_scores += packing_score * self.residue_weights["packing_score"]

        if "helix" in metrics:
            helix_score = 1.0 - metrics["helix"]
            combined_scores += helix_score * self.residue_weights["helix_score"]

        combined_scores *= decay_multiplier
        return np.clip(combined_scores, 0, None)

    def _identify_dominant_problem(self, metrics: Dict[str, np.ndarray], pos: int) -> str:
        """Identify which structural concern is worst at a given residue position."""
        problems = {}
        if "pae_inter" in metrics:
            problems["interface"] = metrics["pae_inter"][pos] / 31.0
        if "pae_intra" in metrics:
            problems["packing"] = metrics["pae_intra"][pos] / 31.0
        if "plddt" in metrics:
            problems["confidence"] = (100 - metrics["plddt"][pos]) / 100.0
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
        """Residue-badness-guided mutation strategy (alternative to pure random sampling)."""
        candidates = []
        for seq, metrics in zip(sequences, metrics_list):
            badness_scores = self._calculate_residue_badness_scores(metrics, decay_multiplier)

            if badness_scores.sum() == 0:
                for _ in range(n_per_seq):
                    n_muts = random.randint(1, 3)
                    positions = random.sample(range(len(seq)), n_muts)
                    mutant = list(seq)
                    for pos in positions:
                        mutant[pos] = random.choice(self.amino_acids)
                    candidates.append("".join(mutant))
                continue

            mutation_probs = badness_scores / badness_scores.sum()
            avg_badness = badness_scores.mean()

            for _ in range(n_per_seq):
                if avg_badness > 0.7:
                    n_muts = random.randint(2, 4)
                elif avg_badness > 0.4:
                    n_muts = random.randint(1, 3)
                else:
                    n_muts = random.randint(1, 2)

                positions = np.random.choice(len(seq), size=min(n_muts, len(seq)), replace=False, p=mutation_probs)
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
