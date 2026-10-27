# SCALE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoss8274/SCALE/blob/main/notebooks/SCALE_demo_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Structure-conditioned active learning for binder optimisation. SCALE is an optimisation loop
placed *after* a binder generator: a generator (RFdiffusion + ProteinMPNN, BoltzGen, ODesign,
or template redesign) supplies a pool of candidates, and SCALE decides where to spend
structure predictions, learns a sequence-to-metric model from what it measures, and proposes
the next designs.

```
 pool of designs ──▶ structure oracle ──▶ metrics ──▶ surrogate (ESM-2 + random forest)
       ▲                                                        │
       └──────────── guided mutation ◀── ranked proposals ◀──────┘
```

## Layout

| Path | Contents |
|---|---|
| `binderopt/` | the loop: surrogate, sampler, losses, optimiser |
| `binderopt/oracles/` | structure-oracle backends and the registry that selects them |
| `binderopt/objectives.py` | what "better" means — a callable the loop maximises |
| `binderopt/pools.py` | read a starting pool from any generator (CSV or FASTA) |
| `tests/` | plumbing tests that need no GPU |
| `notebooks/SCALE_demo.ipynb` | local example with a configurable target, starting pool and objective |
| `notebooks/build_notebook*.py` | generate the notebook from source (keeps it diffable) |
| `configs/` | `defaults.yaml` + one YAML per target, loaded by `binderopt/targets.py` |
| `design.py` | CLI driver for a production run |
| `analysis/` | measurement scripts — interface biophysics, paralogue baselines, lineages |
| `figures/` | figure scripts for the manuscript panels |
| `data/` | the records, metrics and structures the notebook and some scripts read |

## Quick start

In Colab, open the badge above and run the setup cell. It keeps Colab's CUDA-matched
PyTorch for the notebook and installs Boltz 2.2.1 in a separate Python 3.12 environment.
Setup checks the dependencies again on every run, including after an interrupted install.
The installed versions are saved in `scale_run/environment.json`.

Locally:

```bash
git clone https://github.com/cosmoss8274/SCALE
cd SCALE
pip install -e ".[boltz,notebook]"
jupyter notebook notebooks/SCALE_demo.ipynb
```

There are two notebooks and they are generated from one source, so they cannot drift apart:

| | |
|---|---|
| `notebooks/SCALE_demo.ipynb` | local, assumes an environment you have already set up |
| `notebooks/SCALE_demo_colab.ipynb` | Colab: install cell, GPU check, downloads the results |

Regenerate both with `python notebooks/build_notebook.py`; check one runs in a real kernel with
`python notebooks/verify_notebook.py`.

See `VERIFICATION.md` for exactly what has been executed and what has not.

The notebook is designed to evaluate new proposals with its selected predictor. It needs
a GPU. The default budget is 10 initial evaluations followed by 5 rounds of 5 (35 predictions); runtime
depends on the hardware and input sizes. Four things are the user's to supply, one cell each:

| | what | default |
|---|---|---|
| 1 | the target | TrkA ligand-binding domain |
| 2 | the starting pool, from any generator | 100 designs in `data/` |
| 3 | the oracle | Boltz-2 |
| 4 | the objective | maximise interface pTM |

Note that AlphaFold 3 cannot be used from a notebook: its weights require a signed licence
and are not redistributable. Use Boltz-2 there and AF3 on a cluster you control — the loop
code is identical, only `get_oracle(...)` changes.

## Sequence exploration in the demo

The bundled 100-sequence example is highly redundant: 92 entries have another pool
sequence only one residue away. More labels from this pool do not necessarily add broad
sequence coverage. Supply a broader candidate pool to explore different starting families.

The notebook exposes three minimum distances, expressed as fractions of binder length:
10% from **every sequence in the complete starting pool**, 10% between designs in the
same new batch, and 5% from previously measured new designs. For 80-aa binders these are
8, 8 and 4 substitutions. This substitution search requires equal-length sequences;
it does not truncate unequal sequences to compute a distance.

The default proposes from the best measured representative of five founding seeds,
using steps of 8–12 substitutions. The optional novel-refinement setting prioritises
novel representatives, with initial-seed fallbacks and smaller local steps.
All three global distance floors remain enforced. A
five-design batch must contain at least three founders, with at most two descendants of
any one founder. Proposal metadata records the actual generating parent and founder;
founders are starting-sequence identifiers, not inferred structural families. Pairwise
sequence distance is enforced independently of these identifiers. The surrogate ranks
feasible proposals, and a constrained selection chooses the measured batch. If the
constraints cannot be met, the run stops before predicting that batch, without relaxing
them. These configurable exploration settings are not experimentally calibrated.

The two evaluation configurations use `(MIN_POOL_DISTANCE, MIN_BATCH_DISTANCE,
MIN_HISTORY_DISTANCE, USE_NOVEL_REFINEMENT)` values `(0.10, 0.10, 0.05, False)` for
wide exploration and `(0.0625, 0.0625, 0.0375, True)` for moderate novelty with local
refinement. The former retains score-ranked parents and steps of 8–12 substitutions;
the latter prioritises novelty-feasible parents and permits local refinement. This
compares two configuration bundles rather than isolating a single parameter.

Both configurations completed 35 evaluations on the same ten initial labels and
structures, each with 25 new designs. Wider exploration is the demo default: its final
five-design diverse portfolio scored 0.8198–0.8709, with a minimum pairwise distance of
12 positions and four founding seeds. The refinement setting reached a similar best
new score (0.8725), a higher raw top-five median and fewer low-scoring designs, but its
diverse portfolio scored 0.7236–0.8725. Each setting was run once. The default favours
the requested quality of a diverse final set; it does not dominate every score summary.
See [VERIFICATION.md](VERIFICATION.md) for the complete comparison.

An optional `SEED_BATCH_PATH` CSV supplies an explicit initial batch from the pool.
Score-based screening changes the starting condition and should be recorded in a sibling
`<seed_batch>.selection.json` file; the notebook includes it in the exported configuration.
Do not interpret runs with different starting batches as an isolated algorithm comparison.

For the disclosed moderate-start demonstration, set `SEED_BATCH_PATH` to
`data/example_start_trka.csv`. This fixed batch retains seven previously measured seeds
with Boltz-2 interface pTM below 0.85, excludes three known seeds at or above 0.85, and
adds three previously unmeasured entries selected with random seed 0. The accompanying
selection manifest records the exclusion rule and the selected entries. Newly observed
high scores are retained, not filtered after prediction. The default blank field still
samples ten entries from the complete pool.

`history.json` and `trajectory.csv` include parent and founder identifiers, substitutions
from the parent, founder and nearest pool sequence, and nearest-pool identity. The plot
shows measured scores (including separate cumulative best overall and best novel design),
score versus novelty, batch pairwise distance, and contributions
from each founding seed. Per-round medians can rise or fall; only best-so-far is
cumulative. Per-round plans are saved before predictions, so interrupted rounds resume
the same selected sequences. Changed search settings require a fresh output directory.

These constraints are used by the two demo notebooks. The production `design.py` driver
retains its existing selection policy.

## Oracles

The loop never calls a predictor directly — it calls an object implementing
`predict(binder_sequence, **kwargs) -> (confidence, metrics)`. Backends are selected by name:

```python
from binderopt.oracles import get_oracle

oracle = get_oracle("af3",    target_name="TRKA", target_sequence=seq, ...)
oracle = get_oracle("boltz2", target_sequence=seq, msa_path="trka.a3m")
oracle = get_oracle("replay", path="data/trk_multi_joint.json", target="TRKA")
```

- **`af3`** — AlphaFold 3, the oracle used for every number in the manuscript.
- **`boltz2`** — Boltz-2 via its command line. Its confidence JSON carries no summarised
  interface PAE, so `min_pae_interaction` is computed here from `pae_*.npz` (requires
  `--write_full_pae`); without that file the metric is returned as NaN rather than silently
  replaced.
- **`replay`** — recorded evaluations, for reproducing a run without a GPU. Asked for a
  sequence it has not measured it raises, rather than inventing a value.

**Metrics from different oracles are not interchangeable.** An interface pTM of 0.80 from
Boltz-2 does not denote what 0.80 from AlphaFold 3 denotes. The hit criteria used in the
manuscript are calibrated on AlphaFold 3; every oracle stamps `oracle` and `oracle_version`
onto its metrics, the cache key includes the oracle name, and results from different oracles
must not be pooled into one figure or filtered with one threshold.

For running several predictors over a shared MSA on a cluster, OmniFold (Jing et al.,
bioRxiv 2025.07.18.665594) already does this and can be wrapped as a backend rather than
reimplemented.

## Objectives

An objective is any callable `f(metrics) -> float` that the loop maximises. Keeping it
outside the surrogate matters: the surrogate predicts *metrics*, the objective combines them,
so changing what you optimise for does not invalidate a trained surrogate and two runs stay
comparable on the underlying metrics.

```python
from binderopt.objectives import maximise, minimise, weighted, selectivity

objective = maximise("i_ptm")
objective = minimise("min_pae_interaction")
objective = weighted({"i_ptm": 1.0, "min_pae_interaction": -0.1})
objective = selectivity("TRKA", ["TRKB", "TRKC"], penalty=2.5)   # per-receptor metrics
```

`selectivity` penalises the **worst** off-target rather than their sum, so a design cannot
score well by shedding one paralogue while keeping another, and the penalty is linear rather
than hinged, because a threshold leaves most candidates with zero gradient.

## Tests

```bash
python tests/test_boltz_parsing.py   # Boltz-2 output -> metric mapping, no GPU needed
python tests/test_colab_setup.py    # setup recovery and Python checks, no downloads
python tests/test_loop_smoke.py      # evaluator -> surrogate -> sampler -> bookkeeping
```

`test_loop_smoke.py` drives the loop with a stub oracle whose metrics are a deterministic
function of the sequence. Those numbers are synthetic and are never reported as results; the
test exists so that breakage in the wiring fails loudly.

## What runs from a fresh clone, and what does not

`binderopt/`, `notebooks/` and `tests/` are self-contained. The scripts in `analysis/` and
`figures/` are the ones that produced the manuscript panels, and most of them read the output
of a cluster run rather than anything that can be shipped — a pack of 300 or 2,400 predicted
complexes, a run directory with checkpoints, or crystal structures fetched from the PDB. They
are included for provenance: they show exactly how each number was computed. Do not expect
them all to execute here.

Runs from the shipped `data/`:

```
figures/plot_si_biophys.py  plot_trk_full.py  plot_trk_multi.py  plot_trk_mut.py
        plot_trkabc_baseline.py
analysis/calc_lineage.py    calc_native.py    make_gxg_ref.py
```

Needs inputs that are not shipped:

| script | needs |
|---|---|
| `figures/plot_ladder_boxes.py` | `ladder_percandidate.csv` — 23,625 rows of pool-ladder evaluations |
| `figures/plot_si_biophys_traj.py`, `analysis/calc_traj_rosetta_mp.py` | `bio_traj/` — 2,400 predicted complexes |
| `analysis/calc_biophys.py`, `calc_rosetta.py`, `calc_ddg.py` | `bio_pack/` — 300 predicted complexes |
| `figures/render_trk9.py` | PyMOL, plus the nine complexes under `trkstr/` |
| `analysis/sup_trk.py` | `native/1WWW.pdb`, `1WWB.cif`, `1WWC.cif` — fetch from the PDB |
| `analysis/build_trk_msa.py`, `run_trk_multi.py`, `trkabc_baseline.py` | a run directory from the cluster (MSAs, checkpoints, per-target inputs) |

## Analysis scripts

`analysis/` holds the measurements reported alongside the loop. Interface biophysics
(`calc_rosetta.py`) uses Rosetta `InterfaceAnalyzer` on unrelaxed predicted coordinates and
reports shape complementarity, buried area, hydrogen bonds and unsatisfied donors — but **no
energy term**: on unrelaxed coordinates `dG_separated` is dominated by clashes. `calc_ddg.py`
is the pilot that establishes this; with the unbound state repacked and the bound state left
as predicted, 58 of 60 complexes score positive (median +115.6 REU), which measures residual
clash rather than binding. Reporting an energy would require relaxing the bound state, at
which point the coordinates are no longer the oracle's.

## Imputed values

When a metric is absent from an evaluation result the loop substitutes a fixed poor value so
that the design ranks low instead of winning by omission. Those substitutions are
**imputations, not measurements**, and once written to a record they are indistinguishable
from observed values — they enter surrogate training and hit-criteria filtering alike.

| metric | imputed value | where |
|---|---|---|
| `min_pae_interaction` | 7.0 Å | `optimizer.py` (`_METRIC_DEFAULTS`) |
| `l_rmsd` | 10.0 Å | `optimizer.py` (`_METRIC_DEFAULTS`) |
| `l_rmsd` | **5.0 Å** | `losses.py`, when RMSD is non-finite or fails |
| `nll` | NaN | `optimizer.py`, absence is routine and stays missing |

The two `l_rmsd` values differ and were never reconciled; the published runs used both.
Neither is changed here, because changing them would alter reproduction of those runs. Every
substitution now logs at WARNING with an `IMPUTED` prefix.

In the records shipped under `data/`, five `min_pae_interaction` values are exactly 7.0
(2 of 478 in `trk_multi_ctrl.json`, 3 of 525 in `trk_multi_joint.json`) and so cannot be
distinguished from imputations. `analysis/flag_imputed.py` lists them:

```bash
python analysis/flag_imputed.py data/trk_multi_joint.json data/trk_multi_ctrl.json
```

## Caveats carried in the code

- The oracle is also the evaluator; optimising a quantity later used to judge success is
  circular. Independent structural measures in `analysis/` exist for that reason.
- `binderopt/losses.py` falls back to `l_rmsd = 5.0` when RMSD cannot be computed; a fallback
  that silently enters a scored metric should raise instead.
- The starting pool buys the first round, not the last: across pool sizes from 25 to 500 the
  round-15 interface pTM medians agree to within 0.03.

## Licence

The code is MIT; see [LICENSE](LICENSE).

The nine predicted structures in `data/structures/` are **not** MIT. They are AlphaFold 3
Output, carry Google's Output Terms of Use (non-commercial use only), and ship with a copy of
those terms and a citation in [`data/structures/README.md`](data/structures/README.md).

SCALE does not need AlphaFold 3. The default oracle is Boltz-2, which is open source; the AF3
adapter is optional and the notebook never uses it.

## Citing

If SCALE is useful to you, please cite this repository. Please also cite the tools it drives:

- **Boltz-2** — Passaro, S. et al. Boltz-2: Towards Accurate and Efficient Binding Affinity
  Prediction (2025). https://github.com/jwohlwend/boltz
- **ESM-2** — Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure
  with a language model. *Science* **379**, 1123–1130 (2023).
- **AlphaFold 3**, if you use the structures in `data/structures/` — Abramson, J. et al.
  Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*
  **630**, 493–500 (2024).
