"""ColabDesign-style structure loss calculation from AlphaFold3 outputs.

Logic is unchanged from the original; cleaned up to use logging instead of
print, and the dead disabled `calculate_rmsd_loss` implementation (the one
wrapped in a triple-quoted string in the original) was dropped, keeping only
the live "global L-RMSD" version.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from Bio.PDB import MMCIFParser

logger = logging.getLogger(__name__)


class ColabDesignLossCalculator:
    """Calculate ColabDesign-style losses from a single AF3 structure prediction."""

    def __init__(self, structure_file: str, json_file: str):
        """
        Args:
            structure_file: path to .cif structure file.
            json_file: path to *_confidences.json (full, not summary).
        """
        self.structure_file = structure_file
        self.json_file = json_file

        with open(json_file) as f:
            self.full_conf = json.load(f)

        summary_file = json_file.replace("_confidences.json", "_summary_confidences.json")
        with open(summary_file) as f:
            self.summary_conf = json.load(f)

        self._parse_structure()

    def _parse_structure(self) -> None:
        """Parse the CIF structure; the shortest chain is treated as the binder."""
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("pred", self.structure_file)

        residues_info = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):
                        residues_info.append({"chain_id": chain.id, "ca_coord": residue["CA"].coord})

        self.chain_ids = np.array([r["chain_id"] for r in residues_info])
        self.ca_coords = np.array([r["ca_coord"] for r in residues_info])

        unique_chains = np.unique(self.chain_ids)
        if len(unique_chains) < 2:
            raise ValueError(f"Need at least 2 chains, found {len(unique_chains)}: {unique_chains}")

        chain_lengths = {chain: (self.chain_ids == chain).sum() for chain in unique_chains}
        sorted_chains = sorted(chain_lengths.items(), key=lambda x: x[1])

        # Shortest chain is always the binder; everything else is target(s).
        self.binder_chain = sorted_chains[0][0]
        target_chains = [c for c, _ in sorted_chains[1:]]

        self.binder_mask = self.chain_ids == self.binder_chain
        self.target_mask = np.isin(self.chain_ids, target_chains)

        self.target_indices = np.where(self.target_mask)[0]
        self.binder_indices = np.where(self.binder_mask)[0]

        self.target_len = len(self.target_indices)
        self.binder_len = len(self.binder_indices)

        logger.debug(
            "Detected %d chains: %s; binder=%s (%d res); target(s)=%s (%d res)",
            len(unique_chains), list(unique_chains), self.binder_chain, self.binder_len,
            target_chains, self.target_len,
        )

    def calculate_plddt_loss(self) -> float:
        """ColabDesign-style pLDDT loss for the binder: 1 - plddt/100. Lower is better."""
        if "atom_plddts" not in self.full_conf:
            return None

        atom_plddts = np.array(self.full_conf["atom_plddts"])
        atoms_per_residue = 4
        binder_start_atom = self.binder_indices[0] * atoms_per_residue
        binder_end_atom = (self.binder_indices[-1] + 1) * atoms_per_residue
        binder_plddts = atom_plddts[binder_start_atom:binder_end_atom]

        expected_atoms = self.binder_len * atoms_per_residue
        if len(binder_plddts) != expected_atoms:
            num_residues = len(binder_plddts) // atoms_per_residue
            if num_residues == 0:
                return None
            residue_plddts = binder_plddts[: num_residues * atoms_per_residue].reshape(
                num_residues, atoms_per_residue
            ).mean(axis=1)
        else:
            residue_plddts = binder_plddts.reshape(self.binder_len, atoms_per_residue).mean(axis=1)

        return float((1 - residue_plddts / 100.0).mean())

    def calculate_pae_losses(self) -> Tuple[float, float]:
        """PAE-intra (local window) and PAE-inter (binder-target), both normalized 0-1."""
        if "pae" not in self.full_conf:
            return None, None

        pae = np.array(self.full_conf["pae"])
        pae_norm = pae / 31.0
        pae_norm = (pae_norm + pae_norm.T) / 2

        pae_intra_matrix = pae_norm[np.ix_(self.binder_indices, self.binder_indices)]
        pae_intra_local = self._calculate_local_pae_intra(pae_intra_matrix)
        pae_intra_loss = float(pae_intra_local.mean())

        pae_inter = pae_norm[np.ix_(self.binder_indices, self.target_indices)]
        pae_inter_loss = float(pae_inter.mean())

        return pae_intra_loss, pae_inter_loss

    def calculate_contact_losses(
        self, cutoff: float = 8.0, k_intra: int = 12, k_inter: int = 25
    ) -> Tuple[float, float]:
        """Long-range intra-binder contact loss and binder-target contact loss."""
        dist = np.linalg.norm(self.ca_coords[:, None] - self.ca_coords[None, :], axis=-1)
        prob_contact = np.where(dist < cutoff, 0.95, 0.05)
        contact_loss_matrix = -np.log(prob_contact + 1e-8)

        loss_intra = contact_loss_matrix[np.ix_(self.binder_indices, self.binder_indices)]
        longrange_pairs = [
            loss_intra[i, j]
            for i in range(self.binder_len)
            for j in range(i + 6, self.binder_len)
        ]

        if longrange_pairs:
            k_intra = min(k_intra, len(longrange_pairs))
            if k_intra > 0:
                top_k_intra = np.partition(longrange_pairs, k_intra - 1)[:k_intra]
                con_loss = float(np.array(top_k_intra).mean())
            else:
                con_loss = 0.0
        else:
            con_loss = 0.0

        loss_inter = contact_loss_matrix[np.ix_(self.binder_indices, self.target_indices)]
        loss_inter_flat = loss_inter.flatten()
        k_inter = min(k_inter, len(loss_inter_flat))
        if k_inter > 0:
            top_k_inter = np.partition(loss_inter_flat, k_inter - 1)[:k_inter]
            i_con_loss = float(top_k_inter.mean())
        else:
            i_con_loss = 0.0

        return con_loss, i_con_loss

    def calculate_rmsd_loss(self, rmsd_calculator=None) -> Dict[str, float]:
        """Global L-RMSD (binder, target-aligned) using all residues. Empty dict if unavailable."""
        if rmsd_calculator is None:
            return {}

        rmsd_results: Dict[str, float] = {}
        if len(self.binder_indices) > 0 and len(self.target_indices) > 0:
            try:
                l_rmsd = rmsd_calculator.calculate_minimal_rmsd(
                    self.structure_file,
                    residue_indices=self.binder_indices,
                    alignment_indices=self.target_indices,
                )
                if np.isfinite(l_rmsd):
                    rmsd_results["l_rmsd"] = float(l_rmsd)
                else:
                    # IMPUTED, not measured. Note this is 5.0 while optimizer.py imputes
                    # 10.0 for the same metric; the two were never reconciled and the
                    # published runs used both. See _METRIC_DEFAULTS in optimizer.py.
                    logger.warning("IMPUTED l_rmsd=5.0: RMSD was non-finite")
                    rmsd_results["l_rmsd"] = 5.0
            except Exception as exc:
                logger.warning("IMPUTED l_rmsd=5.0: RMSD calculation failed (%s)", exc)
                rmsd_results["l_rmsd"] = 5.0

        return rmsd_results

    def calculate_helix_loss(self, cutoff: float = 6.0) -> float:
        """Helicity loss (0=perfectly helical, 1=not helical at all) via i,i+3/i,i+4 geometry."""
        binder_coords = self.ca_coords[self.binder_mask]
        if len(binder_coords) < 4:
            return None
        helix_qualities = self._calculate_helix_quality_standard(binder_coords)
        return float((1.0 - helix_qualities).mean())

    def calculate_radius_of_gyration(self) -> float:
        """Radius of gyration of the binder; lower = more compact."""
        binder_coords = self.ca_coords[self.binder_mask]
        center = binder_coords.mean(axis=0)
        return float(np.sqrt(((binder_coords - center) ** 2).sum(axis=1).mean()))

    def calculate_termini_distance(self) -> float:
        """N-to-C terminus CA-CA distance of the binder."""
        binder_coords = self.ca_coords[self.binder_mask]
        if len(binder_coords) < 2:
            return None
        return float(np.linalg.norm(binder_coords[0] - binder_coords[-1]))

    def get_iptm(self) -> float:
        """Interface pTM from the AF3 summary confidences file. Higher is better."""
        return float(self.summary_conf.get("iptm", 0.0))

    def get_residue_level_metrics(self) -> Dict[str, np.ndarray]:
        """Per-residue arrays (shape (binder_len,)) used to guide mutation sampling."""
        metrics: Dict[str, np.ndarray] = {}

        if "atom_plddts" in self.full_conf:
            atom_plddts = np.array(self.full_conf["atom_plddts"])
            atoms_per_residue = 4
            binder_start_atom = self.binder_indices[0] * atoms_per_residue
            binder_end_atom = (self.binder_indices[-1] + 1) * atoms_per_residue
            binder_plddts = atom_plddts[binder_start_atom:binder_end_atom]
            expected_atoms = self.binder_len * atoms_per_residue
            if len(binder_plddts) == expected_atoms:
                metrics["plddt"] = binder_plddts.reshape(self.binder_len, atoms_per_residue).mean(axis=1)
            else:
                metrics["plddt"] = np.zeros(self.binder_len)

        if "pae" in self.full_conf:
            pae = np.array(self.full_conf["pae"])
            pae_inter = pae[np.ix_(self.binder_indices, self.target_indices)]
            metrics["pae_inter"] = pae_inter.mean(axis=1)

            pae_intra = pae[np.ix_(self.binder_indices, self.binder_indices)]
            metrics["pae_intra"] = self._calculate_local_pae_intra(pae_intra)

        dist = np.linalg.norm(self.ca_coords[:, None] - self.ca_coords[None, :], axis=-1)
        dist_inter = dist[np.ix_(self.binder_indices, self.target_indices)]
        metrics["con_inter"] = (dist_inter < 8.0).sum(axis=1)

        dist_intra = dist[np.ix_(self.binder_indices, self.binder_indices)]
        metrics["con_intra"] = self._calculate_longrange_contacts(dist_intra)

        binder_coords = self.ca_coords[self.binder_mask]
        metrics["helix"] = self._calculate_helix_quality_standard(binder_coords)

        return metrics

    def _calculate_local_pae_intra(self, pae_intra: np.ndarray, window: int = 5) -> np.ndarray:
        """Average PAE within a +/-window sequence neighborhood, per residue."""
        pae_intra_local = np.zeros(self.binder_len)
        for i in range(self.binder_len):
            start, end = max(0, i - window), min(self.binder_len, i + window + 1)
            neighbors = [j for j in range(start, end) if j != i]
            pae_intra_local[i] = pae_intra[i, neighbors].mean() if neighbors else 0.0
        return pae_intra_local

    def _calculate_longrange_contacts(
        self, dist_intra: np.ndarray, contact_cutoff: float = 8.0, sequence_separation: int = 5
    ) -> np.ndarray:
        """Count of long-range (sequence separation > threshold) spatial contacts per residue."""
        con_longrange = np.zeros(self.binder_len)
        for i in range(self.binder_len):
            for j in range(self.binder_len):
                if i != j and dist_intra[i, j] < contact_cutoff and abs(i - j) > sequence_separation:
                    con_longrange[i] += 1
        return con_longrange

    def _calculate_helix_quality_standard(self, coords: np.ndarray) -> np.ndarray:
        """Per-residue helix quality (0-1, higher=more helical) from i,i+3/i,i+4 CA distances."""
        helix_quality = np.zeros(len(coords))
        ideal_i_i3_dist = 5.1
        tolerance = 1.0

        for i in range(len(coords) - 3):
            dist_i_i3 = np.linalg.norm(coords[i] - coords[i + 3])
            helix_score = np.exp(-((dist_i_i3 - ideal_i_i3_dist) / tolerance) ** 2)
            helix_quality[i] = helix_score
            for offset in range(1, 4):
                if i + offset < len(coords):
                    helix_quality[i + offset] = max(helix_quality[i + offset], helix_score)

        if len(coords) >= 5:
            ideal_i_i4_dist = 6.2
            for i in range(len(coords) - 4):
                dist_i_i4 = np.linalg.norm(coords[i] - coords[i + 4])
                helix_score_i4 = np.exp(-((dist_i_i4 - ideal_i_i4_dist) / tolerance) ** 2)
                for offset in range(5):
                    if i + offset < len(coords):
                        helix_quality[i + offset] = max(helix_quality[i + offset], helix_score_i4 * 0.8)

        return helix_quality

    def calculate_all_losses(self, rmsd_calculator=None) -> Dict[str, float]:
        """Compute all losses + raw scores + residue-level metrics for this structure."""
        losses: Dict[str, float] = {}
        raw_scores: Dict[str, float] = {}

        if "atom_plddts" in self.full_conf:
            atom_plddts = np.array(self.full_conf["atom_plddts"])
            atoms_per_residue = 4
            binder_start_atom = self.binder_indices[0] * atoms_per_residue
            binder_end_atom = (self.binder_indices[-1] + 1) * atoms_per_residue
            binder_plddts = atom_plddts[binder_start_atom:binder_end_atom]
            expected_atoms = self.binder_len * atoms_per_residue
            if len(binder_plddts) == expected_atoms:
                residue_plddts = binder_plddts.reshape(self.binder_len, atoms_per_residue).mean(axis=1)
                raw_scores["plddt"] = float(residue_plddts.mean())
                losses["plddt"] = self.calculate_plddt_loss()

        if "pae" in self.full_conf:
            pae = np.array(self.full_conf["pae"])
            pae_inter_matrix = pae[np.ix_(self.binder_indices, self.target_indices)]
            raw_scores["pae_inter"] = float(pae_inter_matrix.mean())

            pae_intra_matrix = pae[np.ix_(self.binder_indices, self.binder_indices)]
            mask_no_diag = ~np.eye(self.binder_len, dtype=bool)
            raw_scores["pae_intra"] = float(pae_intra_matrix[mask_no_diag].mean())

            pae_intra_loss, pae_inter_loss = self.calculate_pae_losses()
            losses["pae_intra"] = pae_intra_loss
            losses["pae_inter"] = pae_inter_loss

        dist = np.linalg.norm(self.ca_coords[:, None] - self.ca_coords[None, :], axis=-1)
        dist_intra = dist[np.ix_(self.binder_indices, self.binder_indices)]
        longrange_contacts = sum(
            1 for i in range(self.binder_len) for j in range(i + 6, self.binder_len) if dist_intra[i, j] < 8.0
        )
        raw_scores["con_intra"] = float(longrange_contacts)

        dist_inter = dist[np.ix_(self.binder_indices, self.target_indices)]
        raw_scores["con_inter"] = float((dist_inter < 8.0).sum())

        con_loss, i_con_loss = self.calculate_contact_losses()
        losses["con_intra"] = con_loss
        losses["con_inter"] = i_con_loss

        binder_coords = self.ca_coords[self.binder_mask]
        if len(binder_coords) >= 4:
            ideal_dist, tolerance = 5.1, 1.0
            helix_contacts = sum(
                1 for i in range(len(binder_coords) - 3)
                if abs(np.linalg.norm(binder_coords[i] - binder_coords[i + 3]) - ideal_dist) < tolerance
            )
            raw_scores["helix"] = float(helix_contacts)
            losses["helix"] = self.calculate_helix_loss()

        if rmsd_calculator is not None:
            for rmsd_key, rmsd_value in self.calculate_rmsd_loss(rmsd_calculator).items():
                losses[rmsd_key] = rmsd_value
                raw_scores[rmsd_key] = rmsd_value

        losses["i_ptm"] = self.get_iptm()
        raw_scores["i_ptm"] = losses["i_ptm"]
        losses["rg"] = self.calculate_radius_of_gyration()
        raw_scores["rg"] = losses["rg"]
        losses["termini"] = self.calculate_termini_distance()
        raw_scores["termini"] = losses["termini"]

        losses = {k: v for k, v in losses.items() if v is not None}
        raw_scores = {k: v for k, v in raw_scores.items() if v is not None}

        losses["raw_scores"] = raw_scores
        losses["residue_level"] = self.get_residue_level_metrics()
        return losses


def calculate_structure_losses_from_af3(
    structure_file: str, json_file: str, rmsd_calculator=None
) -> Dict[str, float]:
    """Convenience wrapper: build a ColabDesignLossCalculator and return calculate_all_losses()."""
    calculator = ColabDesignLossCalculator(structure_file, json_file)
    return calculator.calculate_all_losses(rmsd_calculator=rmsd_calculator)
