# Metrics and objectives

Every objective returns a scalar to maximise. `minimise(name)` negates a metric;
`weighted(weights)` combines metrics with explicit signed coefficients.

## Predictor metrics

| Name | Definition | Units | Preferred direction |
|---|---|---|---|
| `i_ptm` | Binder–receptor interface pTM | 0–1 | Higher |
| `ptm_binder` | Binder-chain pTM | 0–1 | Higher |
| `min_pae_interaction` | Minimum cross-chain PAE over both directions | Å | Lower |
| `residue_metrics.plddt` | Per-binder-residue confidence | 0–1 | Higher |
| `residue_metrics.pae_inter` | Mean PAE from each binder residue to the receptor | Å | Lower |

Records include the predictor name and version. Confidence thresholds are specific to the
predictor. Missing optional measurements are represented by NaN; objectives require finite
inputs. Surrogate heads train on available finite labels for each metric. Cluster weighted scores
are divided by the sum of absolute active weights; entirely unmeasured columns are inactive,
and rows missing an otherwise active metric are excluded from selection.

## AF3 structural scoring

The AF3 reader uses `token_chain_ids` and CIF residue identities for token mapping,
and averages atom pLDDT within each residue. Specify `binder_chain` when it differs from B.

| Term | Definition | Direction when used in a weighted score |
|---|---|---|
| `plddt` | Mean binder `1 − pLDDT/100` | Negative weight |
| `pae_intra` | Symmetrised off-diagonal binder PAE, divided by 31 Å | Negative weight |
| `pae_inter` | Symmetrised mean binder–target PAE, divided by 31 Å | Negative weight |
| `con_intra` | Mean top-k `−log(p_contact)` per binder residue; sequence separation >5 | Negative weight |
| `con_inter` | Mean top-k `−log(p_contact)` per binder residue against the target | Negative weight |
| `l_rmsd` | Binder Cα RMSD after target alignment to a reference | Negative weight |
| `rg` | Binder Cα radius of gyration | Task-specific |
| `termini` | Binder N-to-C Cα distance | Task-specific |
| `helix` | Cα i,i+3 geometric helix preference, expressed as a loss | Task-specific |
| `nll` | ProteinMPNN sequence negative log-likelihood | Negative weight |

Contact losses use AF3's predicted 8 Å contact probabilities, with per-residue neighbour
counts of 12 internally and 1 at the interface by default. They are unavailable when
contact probabilities are absent. Radius of gyration, terminal distance and helix bias are
optional descriptors, disabled in the shared default scoring weights.

The helix descriptor averages `exp(−((d(i,i+3)−5.1 Å)/1 Å)²)` over windows covering each
residue; it is a geometric preference rather than a secondary-structure assignment.
The reference-aligned RMSD uses one target alignment transform for the binder comparison.

## Selectivity

`selectivity(on_target, off_targets, penalty)` scores on-target utility minus the penalty
times the strongest off-target utility. Its default metric is interface pTM. For a metric
where lower values are preferred, set `higher_is_better=False`.
Cluster selectivity runs evaluate and model on-target and off-target predictions separately.

Definitions follow the [AF3 output format](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md)
and the confidence/contact reductions in [ColabDesign](https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py).
