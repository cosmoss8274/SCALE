# Verification status

What in this directory has been executed, what has not, and what the next person will hit.
Written so that nothing here has to be taken on trust.

## Colab installation check — 28 October 2026

The setup and GPU cells were executed in a real Colab T4 runtime. The original setup
used Python 3.13.16, outside Boltz-2's supported Python range, and skipped dependency
installation on subsequent runs merely because the checkout directory existed.

The revised setup keeps the notebook on Colab's Python 3.13.16 and PyTorch
2.11.0+cu130. It installs Boltz 2.2.1 into a separate Python 3.12.3 environment,
checks dependencies on every run, and records both environments in
`scale_run/environment.json`.

The installation, required notebook imports, CLI startup, kernel GPU check and a tiny
CUDA tensor operation in the isolated backend passed. The backend reported PyTorch
2.14.1, NumPy 1.26.4, CUDA 13.0 and Tesla T4. Initial setup took 143.8 seconds;
a repeat setup with the added CUDA check took 11.5 seconds on this runtime.

`python tests/test_colab_setup.py` passed three offline checks: recovery after a failed
package install, rejection of an incompatible backend Python, and failure to report
success when backend CUDA is unavailable. Package-manager calls are mocked in these tests.

This installation-only check was followed by the complete real-prediction run recorded below.

## Executed and passing

| Check | Command | Result |
|---|---|---|
| Boltz-2 output parsing | `python tests/test_boltz_parsing.py` | 7/7 checks pass |
| Loop wiring | `python tests/test_loop_smoke.py` | passes — 12 designs over 3 rounds |
| Objectives | inline | `maximise`, `minimise`, `weighted`, `selectivity` return the expected values |
| Pool loader | inline | reads `data/example_pool_trka.csv`, 100 unique 80-aa sequences |
| Package imports | inline | all of `binderopt.*` import with no GPU and no network |
| Syntax | inline | 43 Python files parse; all notebook code cells parse |
| Surrogate on real data | earlier build of the notebook, executed cell by cell | ESM-2 650M embedding, random-forest fit, held-out prediction, sampler, ranking, figures and py3Dmol all ran to completion |

`test_boltz_parsing.py` builds the output tree documented in the Boltz repository and checks
the mapping onto the loop's metric names — including that `min_pae_interaction` comes from
`pae_*.npz` and is returned as NaN, not substituted, when `--write_full_pae` was not passed.

`test_loop_smoke.py` drives the loop with a stub oracle whose metrics are a deterministic
function of the sequence. **Those numbers are synthetic and are not results**; the test exists
so that breakage in evaluator → surrogate → sampler → ranking → bookkeeping fails loudly.

## Defects found by review, and fixed

An external review of the first version of this directory found real defects. All were
introduced by copying files without checking that they still worked in the new layout.

| Defect | Status |
|---|---|
| `configs/` not copied, though `binderopt/targets.py:16` requires `<root>/configs` | fixed — `defaults.yaml` and five target YAMLs restored; `list_targets()` now returns `bhrf1, her2, myb, sc2rbd, trka` and `load_target("trka")` builds a `RunConfig` |
| `design.py`, the production CLI driver, not copied | fixed — restored at the root |
| `pyyaml` missing from `requirements.txt` while `targets.py` and `config.py` import it | fixed — a clean install now loads the configs |
| README quoted "58 of 60 complexes score positive (median +115.6 REU)" with no supporting file | fixed — `data/biophys_ddg_pilot.json` now ships |
| Scripts in `analysis/` and `figures/` presented as if runnable: 7 of 8 figure scripts and 11 of 11 analysis scripts could not open their inputs | partly fixed — 13 small input files shipped, bringing 8 scripts to runnable; the remaining 11 need bulk run outputs and are now listed as such in the README with what each requires |

The review's count of "5 of 8" figure scripts was conservative; the true figure was 7 of 8.
Nothing in `binderopt/` was dead code — the missing half was the configuration it reads.

## Second review round, and what was done

| Finding | Status |
|---|---|
| Undisclosed magic number `min_pae_interaction = 7.0` | fixed as far as is safe — the constant is now documented in `optimizer.py` and in the README as an *imputation, not a measurement*; every substitution logs at WARNING with an `IMPUTED` prefix; the duplicated copy at the batch-padding site now reads the same dict instead of repeating the literals |
| Same metric imputed inconsistently | documented, not changed — `optimizer.py` imputes `l_rmsd = 10.0`, `losses.py` imputes `5.0`. The published runs used both. Changing either would alter reproduction of those runs, so the discrepancy is stated in the README table instead |
| Imputed values indistinguishable from measurements in shipped records | quantified and tooled — `analysis/flag_imputed.py` lists them. Five `min_pae_interaction` values are exactly 7.0: records 17 and 85 at round 0 (TrkB) in both arms, and record 155 at round 12 (TrkC) in the joint arm |
| 7 of 8 figure scripts unrunnable | partly fixed in round one — 8 scripts now run from shipped data; the remaining 11 need bulk run outputs and the README lists what each requires |
| README numbers unsupported by repo contents | fixed — "58 of 60 complexes positive, median +115.6 REU" is now backed by `data/biophys_ddg_pilot.json` and re-verified from it (n=60, 58 positive, median +115.6, range −4.9 to +790.4). The two remaining external numbers (0.50–0.97 GPU-minutes; round-15 medians within 0.03 interface pTM) are now labelled as manuscript results whose source tables are not shipped |

**Direction of the imputation's effect.** All five suspect values sit in an off-target
column. Measured TrkB interface PAE across the paralogue baseline ran 4.66–15.24 Å with a
median of 9.21, so imputing 7.0 places those designs *below* the typical off-target value —
it makes the paralogue look more engaged than it may be. The selectivity claim is therefore
conservative with respect to this imputation, not inflated.

## Live notebook verification — 28 October 2026

The earlier local-mutation demo completed all 14 code cells in a fresh Colab Tesla T4
runtime: 10 initial predictions and five rounds of five new predictions (35 total).
Boltz 2.2.1 used three recycling steps, one diffusion sample, and the public MSA server.
The initial best interface pTM was 0.9046 and the run best was 0.9198. The latter differed
from the nearest starting-pool sequence at only 2 of 80 positions. This run verifies
execution and parsing, but also motivates explicit sequence exploration constraints.

The downloaded notebook's executed source matched its input snapshot, with no error
outputs. The result archive passed 19 integrity/content checks, including all 35 metric
records and CIF structures, complete round counts, metric/CSV consistency, exact receptor
and binder sequence correspondence in every CIF, and finite atom coordinates. All 14 code
cells were executed. Source snapshots and generated results are retained privately.

## Distance-constrained comparison — 28 October 2026

Both frozen configurations completed all 14 code cells on real Colab Tesla T4 runtimes.
Each evaluated the same ten initial designs and 25 new designs, for 35 evaluations per
setting. The second run imported the first run's ten initial measured labels and CIFs
exactly; no new-design labels from the first setting entered its surrogate. There were
60 actual prediction calls across these two runs. Both used Boltz 2.2.1, three recycling
steps, one diffusion sample, and the public MSA server.

| Quantity | Wider exploration (default) | Moderate novelty + local refinement |
|---|---:|---:|
| Pool / batch / previous-new distance floors (80 aa) | 8 / 8 / 4 | 5 / 5 / 3 |
| Novel-refinement flag | False | True |
| Best new interface pTM | 0.870861 | 0.872471 |
| Best new distance to nearest of all 100 pool sequences | 8 (90% identity) | 9 (88.75% identity) |
| Raw top-five score median | 0.837592 | 0.862364 |
| All 25 new scores: median / mean | 0.753901 / 0.629556 | 0.741014 / 0.708718 |
| New designs with interface pTM ≥0.8 | 6/25 | 5/25 |
| New designs with interface pTM <0.5 | 6/25 | 2/25 |
| Final diverse portfolio: median / minimum | 0.837592 / 0.819846 | 0.810894 / 0.723616 |
| Final diverse portfolio: minimum pairwise distance | 12 | 5 |
| Final diverse portfolio: founding seeds | 4 | 4 |

The final portfolio contains five measured new designs under the same batch distance
and founder limits, with at most two from one founder. The raw top five in the refinement
run represented only two founders; they cannot be presented as a qualifying diverse
portfolio. Founder identifiers describe starting sequences, not structural families.

Wider exploration is the default because the requested final diverse set retained higher
minimum and median scores and wider separation. The refinement setting had a higher
mean and raw top-five median. These are two configuration bundles, each run once, rather
than a distance-only ablation or evidence of a globally optimal setting.

Both runs used a disclosed initial batch that excluded three *previously known* seeds
with interface pTM ≥0.85, kept seven previously measured lower-scoring seeds, and added
three previously unmeasured seeds selected with random seed 0. One added seed scored
0.886401 and was retained. Neither setting exceeded that shared initial best. The earlier
local-mutation run had a different initial batch and is not a matched performance control.
All low scores and round-to-round declines remain in the downloaded data and figures.

The wider archive passed 44 independent checks; the refinement archive passed 51,
including exact equality of the ten shared labels/CIFs and the backend settings.
Checks covered all 35 metric records, actual CIF sequences and finite coordinates,
complete round counts, saved batch plans, parent/founder lineage, distances to the full
pool and prior new designs, batch separation and founder caps. Both executed notebooks
passed eight checks, including all 14 executed code cells, exact source correspondence
to their respective frozen inputs, no error outputs and five complete guided rounds.

The release's combined sampler reproduced the wider run's five saved rounds of parents,
all 600 proposals per round and selected batches when novel refinement was disabled.
The final plotting change moves legends outside the data areas. Frozen run archives and
input snapshots are retained privately; public notebooks are regenerated without outputs
or embedded private run data.

Offline tests exercise the actual notebook search cells through a simulated interruption,
exact total prediction count, fixed-batch resumption, all three distance floors, actual
parent provenance, founding-seed caps, and rejection of incompatible checkpoint settings.
Synthetic test scores are used only for these code checks, not as prediction results.

The Boltz adapter passes `--use_msa_server` when requested; live runs use it. Parsed
interface PAE comes from the emitted full PAE arrays. The isolated backend version is
recorded in `environment.json`; records may report the generic `boltz2-cli` version label
because the notebook kernel and predictor use different environments.

## Known blockers

**Installing Boltz-2 on a cluster with a slow PyPI link.** Four attempts on the project's HPC
failed, none of them a code problem:

1. `python3` there is 3.9; Boltz requires >=3.10,<3.13. Use a 3.12 interpreter.
2. pip downloads ran at ~10 kB/s with repeated interruptions; a 13 MB wheel failed after six
   resume attempts (`error: incomplete-download`).
3. `--resume-retries` exists only in newer pip, so it cannot be passed to the command that
   upgrades pip itself.
4. Building a local wheelhouse to copy over hits two upstream pins with no current binary
   wheel: `fairscale==0.4.13` (sdist only, and `pip download --no-binary` tries to build it)
   and `modelcif==1.2` (PyPI now serves wheels only for 1.6+).

Colab also needs a compatible Python version. The current setup isolates Boltz-2 in
Python 3.12 instead of relying on the notebook kernel's Python version.

**Metrics are oracle-specific.** The hit criteria in `data/` and the manuscript are calibrated
on AlphaFold 3. A Boltz-2 interface pTM of 0.80 does not denote what 0.80 from AlphaFold 3
denotes. Every oracle stamps `oracle` and `oracle_version` onto its metrics; do not pool
results from two oracles in one figure or filter them with one threshold.

## Numbers quoted in the notebook

Every figure the notebook draws comes from the run it just performed. The only external
numbers are in the closing markdown — AlphaFold 3 cost per prediction (0.50–0.97 GPU-minutes)
and the pool-size result (round-15 medians agree within 0.03 interface pTM across pools of
25–500) — both from the manuscript runs, and both labelled as such.
