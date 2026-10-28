"""Protein-complex confidence losses and coordinate descriptors from AF3 outputs."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


class ColabDesignLossCalculator:
    """Score one AF3 protein complex; chain B is the default binder."""

    def __init__(
        self, structure_file, json_file, *, binder_chain="B", target_chains=None
    ):
        self.structure_file, self.json_file = str(structure_file), str(json_file)
        self.full_conf = json.loads(Path(json_file).read_text())
        summary = Path(json_file).with_name(
            Path(json_file).name.replace(
                "_confidences.json", "_summary_confidences.json"
            )
        )
        if summary == Path(json_file):
            raise ValueError("Expected an AF3 *_confidences.json file")
        self.summary_conf = json.loads(summary.read_text()) if summary.exists() else {}
        self.binder_chain = binder_chain
        self._parse_structure(target_chains)

    def _parse_structure(self, target_chains):
        cif = MMCIF2Dict(self.structure_file)
        chains = np.asarray(cif["_atom_site.label_asym_id"])
        names = np.asarray(cif["_atom_site.label_atom_id"])
        seqids = np.asarray(cif["_atom_site.label_seq_id"])
        models = cif.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(chains))
        xyz = np.column_stack(
            [np.asarray(cif["_atom_site.Cartn_" + k], dtype=float) for k in "xyz"]
        )
        if len(set(models)) != 1 or not np.isfinite(xyz).all():
            raise ValueError("Expected one model with finite atom coordinates")
        ca = np.flatnonzero((names == "CA") & (seqids != ".") & (seqids != "?"))
        keys = [(chains[i], seqids[i]) for i in ca]
        if len(set(keys)) != len(keys):
            raise ValueError("Resolve alternate C-alpha locations before scoring")
        self.chain_ids, self.ca_coords = chains[ca], xyz[ca]
        proteins = list(dict.fromkeys(self.chain_ids))
        self.target_chains = (
            list(target_chains)
            if target_chains is not None
            else [c for c in proteins if c != self.binder_chain]
        )
        if (
            self.binder_chain not in proteins
            or not self.target_chains
            or self.binder_chain in self.target_chains
            or not set(self.target_chains) <= set(proteins)
        ):
            raise ValueError("Choose distinct protein binder and target chains")
        self.binder_mask = self.chain_ids == self.binder_chain
        self.target_mask = np.isin(self.chain_ids, self.target_chains)
        self.binder_indices, self.target_indices = np.flatnonzero(
            self.binder_mask
        ), np.flatnonzero(self.target_mask)
        self.binder_len, self.target_len = len(self.binder_indices), len(
            self.target_indices
        )
        tokens = self.full_conf.get("token_chain_ids")
        if tokens is None and any(
            k in self.full_conf for k in ("pae", "contact_probs")
        ):
            raise ValueError(
                "token_chain_ids are required for AF3 pairwise confidence arrays"
            )
        mapping = np.full(len(ca), -1, dtype=int)
        for chain in [self.binder_chain] + self.target_chains:
            rows = np.flatnonzero(self.chain_ids == chain)
            ids = [i for i, c in enumerate(tokens or []) if c == chain]
            if tokens is not None and len(ids) != len(rows):
                raise ValueError(
                    f"Chain {chain}: expected one token per protein residue"
                )
            if ids:
                mapping[rows] = ids
        self.binder_tokens, self.target_tokens = (
            mapping[self.binder_indices],
            mapping[self.target_indices],
        )
        self.residue_plddt = None
        if "atom_plddts" in self.full_conf:
            v = np.asarray(self.full_conf["atom_plddts"], dtype=float)
            if (
                v.shape != (len(chains),)
                or not np.isfinite(v).all()
                or (v < 0).any()
                or (v > 100).any()
            ):
                raise ValueError(
                    "atom_plddts must match the CIF atoms and lie in [0, 100]"
                )
            if "atom_chain_ids" in self.full_conf and list(
                self.full_conf["atom_chain_ids"]
            ) != list(chains):
                raise ValueError("AF3 atom-chain order differs from the CIF")
            self.residue_plddt = np.asarray(
                [v[(chains == c) & (seqids == r)].mean() for c, r in keys]
            )
        self._pairwise = {}
        for key in ("pae", "contact_probs"):
            if key in self.full_conf:
                v = np.asarray(self.full_conf[key], dtype=float)
                if (
                    v.shape != (len(tokens), len(tokens))
                    or not np.isfinite(v).all()
                    or (v < 0).any()
                ):
                    raise ValueError(f"Invalid token-by-token {key} matrix")
                if key == "contact_probs" and (v > 1).any():
                    raise ValueError("Contact probabilities must lie in [0, 1]")
                self._pairwise[key] = v

    def calculate_plddt_loss(self):
        """Mean binder loss 1 - pLDDT/100, averaged over residues."""
        return (
            None
            if self.residue_plddt is None
            else float(1 - self.residue_plddt[self.binder_indices].mean() / 100)
        )

    def calculate_pae_losses(self):
        """Symmetric off-diagonal intra-binder and interface mean PAE divided by 31."""
        if "pae" not in self._pairwise:
            return None, None
        p = self._pairwise["pae"]
        p = (p + p.T) / 2
        intra = p[np.ix_(self.binder_tokens, self.binder_tokens)]
        mask = ~np.eye(self.binder_len, dtype=bool)
        return (
            float(intra[mask].mean() / 31) if mask.any() else None,
            float(p[np.ix_(self.binder_tokens, self.target_tokens)].mean() / 31),
        )

    @staticmethod
    def _contact_loss(probabilities, mask, k):
        if not isinstance(k, int) or k < 1:
            raise ValueError("Contact neighbour count must be a positive integer")
        values = []
        for row, valid in zip(probabilities, mask):
            p = row[valid]
            if p.size:
                values.append(
                    -np.log(np.clip(np.sort(p)[-min(k, p.size) :], 1e-8, 1)).mean()
                )
        return float(np.mean(values)) if values else None

    def calculate_contact_losses(self, cutoff=8.0, k_intra=12, k_inter=1):
        """Per-residue top-k contact cross-entropy from AF3 contact probabilities."""
        if cutoff != 8.0:
            raise ValueError("AF3 contact_probs use an 8-Angstrom contact definition")
        if "contact_probs" not in self._pairwise:
            return None, None
        p = self._pairwise["contact_probs"]
        intra = p[np.ix_(self.binder_tokens, self.binder_tokens)]
        ids = np.arange(self.binder_len)
        inter = p[np.ix_(self.binder_tokens, self.target_tokens)]
        return (
            self._contact_loss(intra, np.abs(ids[:, None] - ids[None, :]) > 5, k_intra),
            self._contact_loss(inter, np.ones(inter.shape, dtype=bool), k_inter),
        )

    def calculate_rmsd_loss(self, rmsd_calculator=None):
        """Binder C-alpha RMSD after target alignment to a supplied reference."""
        if rmsd_calculator is None:
            return {}
        value = rmsd_calculator.calculate_minimal_rmsd(
            self.structure_file,
            residue_indices=self.binder_indices,
            alignment_indices=self.target_indices,
        )
        if not np.isfinite(value) or value < 0:
            raise ValueError("Reference-aligned binder RMSD is unavailable")
        return {"l_rmsd": float(value)}

    def _helix_preference(self, coords):
        """Optional C-alpha i,i+3 geometric helix preference in [0, 1]."""
        q, counts = np.zeros(len(coords)), np.zeros(len(coords))
        if len(coords) >= 4:
            scores = np.exp(
                -(((np.linalg.norm(coords[3:] - coords[:-3], axis=1) - 5.1) / 1.0) ** 2)
            )
            for offset in range(4):
                q[offset : offset + len(scores)] += scores
                counts[offset : offset + len(scores)] += 1
        return q / np.maximum(counts, 1)

    def calculate_helix_loss(self, cutoff=6.0):
        """Coordinate helix-bias loss; lower favours helical geometry."""
        if cutoff != 6.0:
            raise ValueError("This descriptor uses fixed i,i+3 geometry")
        return (
            None
            if self.binder_len < 4
            else float(
                1 - self._helix_preference(self.ca_coords[self.binder_indices]).mean()
            )
        )

    def calculate_radius_of_gyration(self):
        """Unweighted binder C-alpha radius of gyration in Angstroms."""
        xyz = self.ca_coords[self.binder_indices]
        return float(np.sqrt(np.mean(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1))))

    def calculate_termini_distance(self):
        """Binder N-to-C C-alpha distance in Angstroms."""
        xyz = self.ca_coords[self.binder_indices]
        return float(np.linalg.norm(xyz[0] - xyz[-1])) if len(xyz) > 1 else None

    def get_iptm(self):
        """AF3 interface pTM for the selected binder and target chains."""
        ids = self.summary_conf.get(
            "chain_ids",
            list(dict.fromkeys(self.full_conf.get("token_chain_ids", self.chain_ids))),
        )
        pairs = self.summary_conf.get("chain_pair_iptm")
        if pairs is not None:
            b = ids.index(self.binder_chain)
            targets = [ids.index(c) for c in self.target_chains]
            value = float(
                np.mean(
                    [float(pairs[b][t]) for t in targets]
                    + [float(pairs[t][b]) for t in targets]
                )
            )
        else:
            value = self.summary_conf.get("iptm") if len(ids) == 2 else None
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Invalid AF3 interface pTM")
        return value

    def get_residue_level_metrics(self):
        """Per-binder-residue confidence arrays and coordinate contact counts."""
        metrics = {}
        if self.residue_plddt is not None:
            metrics["plddt"] = self.residue_plddt[self.binder_indices] / 100
        if "pae" in self._pairwise:
            p = self._pairwise["pae"]
            metrics["pae_inter"] = p[
                np.ix_(self.binder_tokens, self.target_tokens)
            ].mean(axis=1)
            if self.binder_len > 1:
                intra = p[np.ix_(self.binder_tokens, self.binder_tokens)].copy()
                np.fill_diagonal(intra, np.nan)
                metrics["pae_intra"] = np.nanmean(intra, axis=1)
        xyz = self.ca_coords[self.binder_indices]
        target = self.ca_coords[self.target_indices]
        metrics["con_inter"] = (
            np.linalg.norm(xyz[:, None] - target[None, :], axis=-1) < 8
        ).sum(axis=1)
        ids = np.arange(self.binder_len)
        metrics["con_intra"] = (
            (np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1) < 8)
            & (np.abs(ids[:, None] - ids[None, :]) > 5)
        ).sum(axis=1)
        if self.binder_len >= 4:
            metrics["helix"] = self._helix_preference(xyz)
        return metrics

    def calculate_all_losses(self, rmsd_calculator=None):
        """Return scalar losses, raw measurements and per-residue sampling metrics."""
        pi, pe = self.calculate_pae_losses()
        ci, ce = self.calculate_contact_losses()
        losses = dict(
            plddt=self.calculate_plddt_loss(),
            pae_intra=pi,
            pae_inter=pe,
            con_intra=ci,
            con_inter=ce,
            helix=self.calculate_helix_loss(),
            i_ptm=self.get_iptm(),
            rg=self.calculate_radius_of_gyration(),
            termini=self.calculate_termini_distance(),
        )
        losses.update(self.calculate_rmsd_loss(rmsd_calculator))
        raw = {
            k: losses[k] for k in ("i_ptm", "rg", "termini", "l_rmsd") if k in losses
        }
        if self.residue_plddt is not None:
            raw["plddt"] = float(self.residue_plddt[self.binder_indices].mean())
        if pi is not None:
            raw["pae_intra"] = pi * 31
        if pe is not None:
            raw["pae_inter"] = pe * 31
        result = {k: v for k, v in losses.items() if v is not None}
        result["raw_scores"] = {k: v for k, v in raw.items() if v is not None}
        result["residue_level"] = self.get_residue_level_metrics()
        return result


def calculate_structure_losses_from_af3(
    structure_file, json_file, rmsd_calculator=None, **kwargs
):
    """Calculate metrics for one AF3 model and its confidence JSON."""
    return ColabDesignLossCalculator(
        structure_file, json_file, **kwargs
    ).calculate_all_losses(rmsd_calculator)
