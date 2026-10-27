# Verification status

What in this directory has been executed, what has not, and what the next person will hit.
Written so that nothing here has to be taken on trust.

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
`pae_*.npz` and is returned as NaN, not substituted, when `--write_pae` was not passed.

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

## Not executed

**Boltz-2 has never been run from this code.** The adapter is validated against the documented
output schema, not against a live prediction. Running it is the first thing to do in Colab,
where `pip install boltz` is routine.

**The notebook has not been executed in a Jupyter kernel.** The sandbox this was written in
forbids socket binding, so `nbconvert --execute` cannot start a kernel; cells were executed as
a script instead, in one namespace, which exercises the same code but not the rendering. The
py3Dmol views in particular have never been rendered. Run the notebook top to bottom in Colab
before trusting its output.

**`--use_msa_server` is an open question.** `binderopt/oracles/boltz.py` does not pass it, so
the receptor runs in single-sequence mode unless you supply `msa_path`. Boltz may support
fetching an MSA from the ColabFold server; check `boltz predict --help` and wire it up if the
flag exists. Do not assume it does.

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

In Colab none of this applies — the network is fast and pip resolves sdists normally.

**Metrics are oracle-specific.** The hit criteria in `data/` and the manuscript are calibrated
on AlphaFold 3. A Boltz-2 interface pTM of 0.80 does not denote what 0.80 from AlphaFold 3
denotes. Every oracle stamps `oracle` and `oracle_version` onto its metrics; do not pool
results from two oracles in one figure or filter them with one threshold.

## Numbers quoted in the notebook

Every figure the notebook draws comes from the run it just performed. The only external
numbers are in the closing markdown — AlphaFold 3 cost per prediction (0.50–0.97 GPU-minutes)
and the pool-size result (round-15 medians agree within 0.03 interface pTM across pools of
25–500) — both from the manuscript runs, and both labelled as such.
