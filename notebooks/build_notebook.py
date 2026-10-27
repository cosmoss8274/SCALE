"""Generate the SCALE notebooks -- both of them, from this one source.

    python notebooks/build_notebook.py

writes two files:

    notebooks/SCALE_demo.ipynb         run it where you already have an environment
    notebooks/SCALE_demo_colab.ipynb   run it on a fresh Colab T4, nothing installed

They are the same pipeline. The Colab variant adds setup, a GPU assertion, weight
downloads and a results download, and rebases every path under ``/content``; the local
variant assumes a working checkout and writes beside it. Generating both from one source
is deliberate: hand-maintained pairs drift, and the drift is silent -- Protein-Hunter's
own pair diverged in how it aggregates ipTM, with only the Colab copy updated.

The notebook runs a real optimisation: a real structure oracle on real new sequences.
Nothing is replayed. The four things a user supplies -- target, starting pool, oracle,
objective -- each get their own cell.

Committed unexecuted, which is the field convention: BindCraft and Protein-Hunter both
ship their Colab notebooks with no outputs. py3Dmol views cannot survive a save anyway.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

#: Fill this in once the repository has a public home; the badge and the clone both need
#: it. Kept as one constant so there is exactly one place to edit.
REPO_URL = "https://github.com/cosmoss8274/SCALE"
COLAB_NOTEBOOK = "notebooks/SCALE_demo_colab.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def form(source: str):
    """A code cell shown as a Colab form: widgets visible, code hidden."""
    cell = code(source)
    cell.metadata["cellView"] = "form"
    return cell


# ----------------------------------------------------------------------------------
# Cells shared by both variants are functions of `colab` so the few differences stay
# next to the code they affect rather than in a patch applied afterwards.
# ----------------------------------------------------------------------------------

def title_cells(colab: bool):
    cells = []
    if colab:
        cells.append(md(
            '<a href="https://colab.research.google.com/github/%s/blob/main/%s" '
            'target="_parent"><img src="https://colab.research.google.com/assets/'
            'colab-badge.svg" alt="Open In Colab"/></a>'
            % (REPO_URL.removeprefix("https://github.com/"), COLAB_NOTEBOOK)))

    cells.append(md(r"""# SCALE — run the loop on your own target

**SCALE** (Structure-Conditioned Active Learning for protein Evolution) is an optimisation
loop placed *after* a binder generator. The generator supplies a pool of candidates; SCALE
decides where to spend structure predictions, learns a sequence model from what it measures,
and proposes the next designs.

```
 pool of designs ──▶ structure oracle ──▶ metrics ──▶ surrogate (ESM-2 + random forest)
       ▲                                                        │
       └──────────── guided mutation ◀── ranked proposals ◀──────┘
```

**This notebook runs that loop for real.** Every structure in it is predicted while it
runs, on sequences that did not exist before you ran the cell. Nothing is replayed.

Four things are yours to supply, and each has its own cell:

| | what | default here |
|---|---|---|
| **1** | the **target** | TrkA ligand-binding domain (101 aa) |
| **2** | the **starting pool** — from any generator | 100 designs shipped in `data/` |
| **3** | the **oracle** — any structure predictor | Boltz-2 |
| **4** | the **objective** — what "better" means | maximise interface pTM |"""))

    cells.append(md(r"""> **⚠️ Read before you run.**
>
> - This pipeline has **not been experimentally validated**. It optimises a structure
>   predictor's confidence, which is not the same as optimising binding. Treat the output
>   as candidates to test, never as results.
> - The oracle is also the scorer, so the loop is optimising the quantity it is judged by.
>   `analysis/` computes independent structural measures for exactly that reason.
> - **Metrics are not comparable across predictors.** An interface pTM of 0.80 from Boltz-2
>   does not denote what 0.80 from AlphaFold 3 denotes. The manuscript's hit criteria are
>   calibrated on AlphaFold 3 and do not transfer. Every record is stamped with the oracle
>   that produced it, and the code refuses to pool two oracles in one comparison.
> - A short run shows the loop **working**, not that the method works. That claim needs the
>   full-scale runs in the manuscript (15 rounds, 225 predictions per target)."""))
    return cells


def setup_cells(colab: bool):
    """Install, clone, GPU check. Only the Colab variant installs anything."""
    cells = [md("## 0 Setup")]

    if colab:
        cells.append(form(r'''#@title 🛠️ Install SCALE and Boltz-2 (~5 min, mostly downloads) { display-mode: "form" }
%%time
#@markdown Repository to clone. Change it if you are running your own fork.
REPO_URL = "''' + REPO_URL + r'''"  #@param {type:"string"}

# Deliberately NOT installed, and why:
#   AlphaFold 3  - the weights need a signed licence from DeepMind and are not
#                  redistributable, so no notebook can fetch them. Boltz-2 is used here;
#                  run AF3 on a cluster you control. Only get_oracle(...) changes.
#   MSA tools    - jackhmmer/hhblits need sequence databases of several hundred GB.
#                  Boltz can query a remote MSA server instead; see the target cell.
#   torch        - Colab ships a build matched to its own driver. Reinstalling it breaks
#                  CUDA, so boltz is installed without touching it.
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/content/SCALE")

# Guarded so that re-running after a disconnect is cheap, and still repairs sys.path.
if not ROOT.exists():
    print("cloning SCALE ...", flush=True)
    subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)

    print("installing boltz and friends (this is the slow part) ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "boltz", "fair-esm", "py3Dmol", "scikit-learn", "biopython"],
                   check=True)
    print("done", flush=True)
else:
    print("already set up at", ROOT)

CODE = ROOT / "code" if (ROOT / "code").exists() else ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
DATA = CODE / "data"
RUN = Path("/content/scale_run"); RUN.mkdir(parents=True, exist_ok=True)
print("code:", CODE)
print("run :", RUN)'''))

        cells.append(form(r'''#@title 🖥️ Check the GPU { display-mode: "form" }
#@markdown Boltz-2 needs a CUDA GPU. This cell **stops the notebook** if there is not one,
#@markdown rather than letting a CPU run start that would never finish.
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU. Runtime > Change runtime type > Hardware accelerator: T4 GPU, "
        "then run this cell again. Boltz-2 on CPU takes hours per prediction.")

name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print("GPU: %s  (%.1f of %.1f GB free)" % (name, free / 1e9, total / 1e9))
if total / 1e9 < 14:
    print("\nNote: under ~15 GB you may hit OOM on long targets. Shorten the receptor,\n"
          "or use a larger runtime.")'''))
    else:
        cells.append(code(r'''# Local run: assumes the environment of requirements.txt and a checkout you are in.
#     pip install -r requirements.txt && pip install boltz
import sys
from pathlib import Path

CODE = (Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()).resolve()
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
DATA = CODE / "data"
RUN = (CODE / "run"); RUN.mkdir(exist_ok=True)

import torch
print("code:", CODE)
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available()
       else "none — Boltz-2 needs CUDA; the rest of the notebook still runs")'''))

    budget = r'''#@title ⚙️ Budget and plot style { display-mode: "form" }
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
})
BLUE, GREY, AMBER = "#1F5FBF", "#9aa0a6", "#E08A1E"

#@markdown Every unit below is one structure prediction, so these set the GPU budget.
#@markdown On a T4 a 100-aa receptor with an 80-aa binder takes roughly 1–2 minutes each.
#@markdown
#@markdown `N_SEED` is the one to think about. Seeds are spent measuring designs the
#@markdown generator already made, and the best of them is where the loop starts, so a
#@markdown large sample starts the run near the pool's ceiling and leaves the loop
#@markdown nothing visible to do. Ten is enough to train on and leaves roughly twice the
#@markdown headroom of sixteen.
N_SEED       = 10   #@param {type:"integer"}
N_ROUNDS     = 5    #@param {type:"integer"}
N_PER_ROUND  = 5    #@param {type:"integer"}
#@markdown Mutants ranked by the surrogate each round. Costs no oracle calls, so this can
#@markdown be large; the sampler proposes roughly 500 per round.
N_CANDIDATES = 600  #@param {type:"integer"}

_total = N_SEED + N_ROUNDS * N_PER_ROUND
print("%d structure predictions in total" % _total)
print("at 1-2 min each that is roughly %d-%d minutes of GPU time" % (_total, 2 * _total))
print("checkpointed after every prediction, so a disconnect costs at most one")'''
    cells.append(form(budget))
    return cells


def target_cells(colab: bool):
    structure_default = "data/structures/TRKA_r00_0.cif"
    return [
        md(r"""## 1 Your target

One receptor sequence. The default is the fifth extracellular domain of TrkA (UniProt
P04629, residues 282–382; the segment modelled in PDB 1WWW)."""),

        form(r'''#@title 1. The receptor { display-mode: "form" }
#@markdown Paste your own receptor sequence to retarget the whole notebook.
TARGET_NAME = "TRKA"  #@param {type:"string"}
TARGET_SEQUENCE = "VSFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAFMDNP"  #@param {type:"string"}

#@markdown An MSA makes the receptor fold markedly more accurate. Leave the server option
#@markdown on unless you have your own A3M — but note it sends the receptor sequence to a
#@markdown public third-party service.
USE_MSA_SERVER = True  #@param {type:"boolean"}
TARGET_MSA = ""  #@param {type:"string"}

TARGET_SEQUENCE = "".join(TARGET_SEQUENCE.split()).upper()
TARGET_MSA = TARGET_MSA or None
if TARGET_MSA and USE_MSA_SERVER:
    raise ValueError("Give either TARGET_MSA or USE_MSA_SERVER, not both.")

print("%s: %d residues" % (TARGET_NAME, len(TARGET_SEQUENCE)))
if TARGET_MSA:
    print("receptor MSA:", TARGET_MSA)
elif USE_MSA_SERVER:
    print("receptor MSA: fetched by Boltz from the public MMseqs2 server")
else:
    print("receptor MSA: none — single-sequence mode is markedly less accurate")'''),

        md("Before spending GPU time, look at what you are designing against."),

        form(r'''#@title 1.1 Look at the target { display-mode: "form" }
#@markdown Orientation only — the loop needs just the sequence. Leave blank to skip.
TARGET_STRUCTURE = "''' + structure_default + r'''"  #@param {type:"string"}
TARGET_CHAIN     = "A"  #@param {type:"string"}

import py3Dmol

path = (CODE / TARGET_STRUCTURE) if TARGET_STRUCTURE else None
if path and path.exists():
    view = py3Dmol.view(width=640, height=380, js="https://3dmol.org/build/3Dmol.js")
    view.addModel(path.read_text(), "cif")
    view.setStyle({"chain": TARGET_CHAIN}, {"cartoon": {"color": "#8fa3bf"}})
    view.zoomTo({"chain": TARGET_CHAIN})
    print("%s chain %s" % (path.name, TARGET_CHAIN))
    view.show()
else:
    print("no structure supplied — skipping. The loop does not need one.")'''),
    ]


def pool_cells(colab: bool):
    return [
        md(r"""## 2 Your starting pool

SCALE does not generate the first designs; it improves designs something else produced.
Generators are selected by name and registered like oracles, so a new one plugs in without
touching the loop:

```python
from binderopt.generators import get_generator, GENERATORS

get_generator("file", path="my_designs.csv")        # anything, from any tool
get_generator("rfdiffusion_mpnn", config=cfg)       # run the pipeline here
```

The pool shipped here is 100 RFdiffusion + ProteinMPNN designs for TrkA. **Point
`POOL_PATH` at your own CSV or FASTA and the rest of the notebook is unchanged.**"""),

        form(r'''#@title 2. The starting pool { display-mode: "form" }
from binderopt.generators import get_generator, GENERATORS

#@markdown A CSV (with a sequence column) or a FASTA. Yours goes here.
POOL_PATH    = "data/example_pool_trka.csv"  #@param {type:"string"}
SEQ_COLUMN   = "sequence"  #@param {type:"string"}
#@markdown Require every binder to be this long; 0 accepts any length.
BINDER_LENGTH = 80  #@param {type:"integer"}

print("available generators:", ", ".join(sorted(GENERATORS)))

pool_path = Path(POOL_PATH)
if not pool_path.is_absolute():
    pool_path = CODE / pool_path
generator = get_generator("file", path=pool_path, sequence_column=SEQ_COLUMN,
                          expect_length=BINDER_LENGTH or None)
pool = generator.generate(10_000)

rng = np.random.default_rng(0)
seed_designs = [pool[i] for i in rng.choice(len(pool), size=min(N_SEED, len(pool)),
                                            replace=False)]
print("%s: %d designs of %d aa | %d sampled to measure"
      % (generator.describe(), len(pool), len(pool[0]), len(seed_designs)))'''),
    ]


def oracle_cells(colab: bool):
    return [
        md(r"""## 3 Your oracle

The loop never calls a predictor directly. It calls an object with
`predict(binder_sequence) -> (confidence, metrics)`, and every backend translates its own
field names into one vocabulary — `i_ptm`, `ptm_binder`, `min_pae_interaction` — defined in
`binderopt/oracles/metrics.py`. Switching predictor is one line:

```python
get_oracle("boltz2", target_sequence=seq, ...)   # open source, runs here
get_oracle("af3",    interface=af3_interface)    # on a cluster you control
```

A new predictor is a `@register_oracle("name")` decorator in your own code — no edit to the
package. What the contract does **not** do is make two predictors comparable: a shared name
is not a shared scale, so each record is stamped with the oracle that produced it and the
code refuses to pool them."""),

        form(r'''#@title 3. The oracle { display-mode: "form" }
from binderopt.oracles import get_oracle, ORACLES
from binderopt.oracles.evaluator import OracleEvaluator

print("available oracles:", ", ".join(sorted(ORACLES)))

#@markdown Samples per prediction. More is slower and a little better.
DIFFUSION_SAMPLES = 1  #@param {type:"integer"}
RECYCLING_STEPS   = 3  #@param {type:"integer"}

oracle = get_oracle(
    "boltz2",
    target_sequence=TARGET_SEQUENCE,
    msa_path=TARGET_MSA,
    use_msa_server=USE_MSA_SERVER,
    work_dir=str(RUN / "boltz"),
    diffusion_samples=DIFFUSION_SAMPLES,
    recycling_steps=RECYCLING_STEPS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
)
evaluator = OracleEvaluator(oracle, structure_dir=str(RUN / "structures"))
print("oracle: %s (%s)" % (type(oracle).__name__, oracle.oracle_version))'''),
    ]


def objective_cells(colab: bool):
    return [
        md(r"""## 4 Your objective

Any callable `f(metrics) -> float` that the loop maximises:

```python
maximise("i_ptm")                                  # this notebook
minimise("min_pae_interaction")                    # tighter interface instead
weighted({"i_ptm": 1.0, "min_pae_interaction": -0.1})
selectivity("TRKA", ["TRKB", "TRKC"], penalty=2.5) # needs per-receptor metrics
```

The objective sits outside the surrogate on purpose: the surrogate predicts *metrics*, the
objective combines them. Changing what you optimise for therefore does not invalidate a
trained surrogate, and two runs stay comparable on the underlying metrics."""),

        form(r'''#@title 4. The objective { display-mode: "form" }
from binderopt.objectives import maximise, minimise, weighted, selectivity, describe

OBJECTIVE = "maximise i_ptm"  #@param ["maximise i_ptm", "minimise min_pae_interaction", "weighted i_ptm - 0.1*pae"]

objective = {
    "maximise i_ptm": lambda: maximise("i_ptm"),
    "minimise min_pae_interaction": lambda: minimise("min_pae_interaction"),
    "weighted i_ptm - 0.1*pae": lambda: weighted({"i_ptm": 1.0,
                                                  "min_pae_interaction": -0.1}),
}[OBJECTIVE]()
print("objective:", describe(objective))'''),
    ]


def loop_cells(colab: bool):
    return [
        md(r"""## 5 Round 0 — measure the pool

The first predictions. These are the labels the surrogate learns from, so they are spent on
the starting pool rather than on proposals.

Each round is checkpointed to `RUN/history.json`, and re-running the loop cell picks up
where it stopped. Colab disconnects; an hour of GPU time should not die with the tab."""),

        form(r'''#@title 5. Measure the starting pool { display-mode: "form" }
import time
from binderopt.oracles.metrics import describe as describe_metrics

HISTORY_PATH = RUN / "history.json"
history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []


def save_history():
    HISTORY_PATH.write_text(json.dumps(history, indent=1))


def measure(sequences, rnd):
    """Predict each sequence, checkpointing after every one.

    Per design rather than per round: a round is 15-30 minutes of GPU time, and Colab
    disconnects. Saving only at the end of a round throws away everything already paid
    for. Sequences already in the history are skipped, so re-running this cell after a
    disconnect costs nothing for the work that survived.
    """
    done = {h["sequence"] for h in history}
    todo = [s for s in sequences if s not in done]
    if todo and len(todo) < len(sequences):
        print("resuming: %d of %d already measured" % (len(sequences) - len(todo),
                                                       len(sequences)))
    for n, seq in enumerate(todo, 1):
        t0 = time.time()
        for record in evaluator.evaluate([seq], iteration=rnd):
            history.append(dict(round=rnd, **record))
            save_history()
            print("  [%d/%d] %s  %.1f min" % (n, len(todo), describe_metrics(record),
                                              (time.time() - t0) / 60), flush=True)
    return [h for h in history if h["round"] == rnd]


t0 = time.time()
measured = measure(seed_designs, 0)
mins = (time.time() - t0) / 60
print("round 0: %d of %d designs measured, %.1f min"
      % (len(measured), len(seed_designs), mins))
if evaluator.failures:
    print("failed:", [f["sequence_id"] for f in evaluator.failures])

if not measured:
    raise RuntimeError(
        "Every prediction failed, so there is nothing to learn from. The error for the "
        "first one is in evaluator.failures[0]['error'].")

scores0 = [objective(m) for m in measured]
print("objective: median %.3f | best %.3f" % (np.median(scores0), max(scores0)))
print("best design:", describe_metrics(max(measured, key=objective)))'''),

        md(r"""## 6 The loop

Each round, in full, with nothing hidden:

1. **train** the surrogate on everything measured so far
2. **propose** mutants of the current best designs
3. **rank** them with the surrogate — thousands of candidates, no oracle calls
4. **measure** only the top `N_PER_ROUND` with the oracle
5. **keep** the results, checkpoint, repeat"""),

        form(r'''#@title 6. Run the loop { display-mode: "form" }
import random
from binderopt.models import MultiHeadSurrogate
from binderopt.sampling import MultiMetricGuidedSampler

random.seed(0); np.random.seed(0)
surrogate = MultiHeadSurrogate(
    esm2_model="esm2_t33_650M_UR50D",
    rf_params={"n_estimators": 300, "random_state": 0, "n_jobs": -1},
    metric_weights={"i_ptm": 1.0},
    calculate_rmsd=False,
)
sampler = MultiMetricGuidedSampler(top_n_for_sampling=5)

for rnd in range(1, N_ROUNDS + 1):
    if any(h["round"] == rnd for h in history):
        print("round %d: already done, skipping" % rnd)
        continue
    t0 = time.time()

    # 1 train on everything measured so far
    seqs = [h["sequence"] for h in history]
    surrogate.train(seqs, {"i_ptm": [h["i_ptm"] for h in history]})

    # 2 propose mutants of the current best, guided by where their structures are worst.
    #    residue_metrics is per binder residue, supplied by the oracle; a design without
    #    it contributes None and gets random mutations instead of being dropped.
    best = sorted(history, key=objective, reverse=True)[:5]
    candidates = sampler.generate_candidates(
        top_sequences=[h["sequence"] for h in best],
        top_residue_metrics=[h.get("residue_metrics") or None for h in best],
        current_iteration=rnd, max_iterations=N_ROUNDS,
    )
    known = set(seqs)
    candidates = [c for c in dict.fromkeys(candidates) if c not in known][:N_CANDIDATES]
    if not candidates:
        print("round %d: the sampler proposed nothing new; stopping" % rnd)
        break

    # 3 rank with the surrogate, 4 measure only the top few
    proposals, predicted = surrogate.top_n_selection(candidates, n=N_PER_ROUND)
    results = measure(proposals, rnd)

    got = [objective(m) for m in results]
    guided = sum(1 for h in best if h.get("residue_metrics"))
    print("round %d | %4d ranked (%d of %d parents structure-guided) | measured %d in "
          "%.1f min | surrogate %.3f-%.3f | best this round %.3f (run best %.3f)"
          % (rnd, len(candidates), guided, len(best), len(results),
             (time.time() - t0) / 60, predicted.min(), predicted.max(),
             max(got) if got else float("nan"),
             max(objective(h) for h in history)))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n%d designs measured over %d rounds" % (len(history), len({h["round"] for h in history})))'''),
    ]


def results_cells(colab: bool):
    cells = [
        md(r"""## 7 The trajectory

What the run just produced. Each point is one design the oracle measured; the line follows
the best objective value reached so far."""),

        form(r'''#@title 7. Trajectory { display-mode: "form" }
rounds = sorted({h["round"] for h in history})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

ax = axes[0]
ax.scatter([h["round"] for h in history], [objective(h) for h in history],
           s=22, color=GREY, edgecolor="white", linewidths=0.3, zorder=2,
           label="each design")
running = [max(objective(h) for h in history if h["round"] <= r) for r in rounds]
ax.plot(rounds, running, "-o", color=BLUE, lw=1.3, ms=3.5, zorder=3, label="best so far")
ax.set(xlabel="Round", ylabel=describe(objective).capitalize(), xticks=rounds)
ax.legend(fontsize=7.5, loc="lower right")

ax = axes[1]
for key, colour, label in (("i_ptm", BLUE, "interface pTM"),
                           ("min_pae_interaction", AMBER, "interface PAE (Å)")):
    per_round = [[h[key] for h in history if h["round"] == r] for r in rounds]
    if all(np.isnan(v).all() for v in per_round):
        continue            # NaN throughout, e.g. no PAE written: draw nothing
    target_ax = ax if key == "i_ptm" else ax.twinx()
    target_ax.plot(rounds, [np.nanmedian(v) for v in per_round],
                   "-o", color=colour, lw=1.2, ms=3.2)
    target_ax.set_ylabel(label, color=colour)
    target_ax.tick_params(axis="y", colors=colour)
ax.set(xlabel="Round", xticks=rounds)
fig.tight_layout()
fig.savefig(RUN / "trajectory.png", dpi=200, bbox_inches="tight")
plt.show()

first = [objective(h) for h in history if h["round"] == 0]
last = [objective(h) for h in history if h["round"] == rounds[-1]]
print("objective: round 0 median %.3f → round %d median %.3f (run best %.3f)"
      % (np.median(first), rounds[-1], np.median(last),
         max(objective(h) for h in history)))'''),

        md(r"""With three rounds of five designs this shows the loop **running**, not that the
method works. Raise `N_ROUNDS` if you want the trend rather than the mechanics."""),

        md("## 8 The best design"),

        form(r'''#@title 8. The best design { display-mode: "form" }
import py3Dmol

top = max(history, key=objective)
pae = top.get("min_pae_interaction", float("nan"))
print("objective %.3f | interface pTM %.3f | interface PAE %s | round %d"
      % (objective(top), top["i_ptm"],
         "n/a" if np.isnan(pae) else "%.2f Å" % pae, top["round"]))
print("oracle: %s %s" % (top.get("oracle"), top.get("oracle_version")))
print(top["sequence"])

structure = top.get("structure_file")
if structure and Path(structure).exists():
    view = py3Dmol.view(width=720, height=420, js="https://3dmol.org/build/3Dmol.js")
    view.addModel(Path(structure).read_text(), "cif")
    view.setStyle({"chain": "A"}, {"cartoon": {"color": "#d4d4d4"}})   # receptor
    view.setStyle({"chain": "B"}, {"cartoon": {"color": BLUE}})        # binder
    view.zoomTo()
    view.show()
else:
    print("\n(no structure file kept for this design)")'''),

        form(r'''#@title 8.1 Save the run { display-mode: "form" }
import csv

FIELDS = ["round", "sequence_id", "sequence", "objective", "i_ptm", "ptm_binder",
          "min_pae_interaction", "oracle", "oracle_version", "structure_file"]
with open(RUN / "trajectory.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for h in history:
        writer.writerow({**h, "objective": objective(h)})

(RUN / "run_config.json").write_text(json.dumps({
    "target_name": TARGET_NAME, "target_sequence": TARGET_SEQUENCE,
    "objective": describe(objective),
    "oracle": type(oracle).__name__, "oracle_version": oracle.oracle_version,
    "n_seed": N_SEED, "n_rounds": N_ROUNDS, "n_per_round": N_PER_ROUND,
    "n_candidates": N_CANDIDATES, "pool": str(pool_path),
    "use_msa_server": USE_MSA_SERVER,
}, indent=2))

print("wrote %s" % (RUN / "trajectory.csv"))
print("     %d designs, %d columns" % (len(history), len(FIELDS)))'''),
    ]

    if colab:
        cells.append(form(r'''#@title 📥 Download the results { display-mode: "form" }
#@markdown Zips the predicted structures, the per-design metrics, the trajectory plot and
#@markdown the run configuration, then downloads it — a Colab session is temporary.
#@markdown
#@markdown If the download does not start, disable your ad blocker and run this cell
#@markdown again, or open the folder icon in the left sidebar and download it by hand.
import shutil

archive = shutil.make_archive("/content/scale_run", "zip", RUN)
print("%s (%.1f MB)" % (archive, Path(archive).stat().st_size / 1e6))

try:
    from google.colab import files
    files.download(archive)
except ImportError:
    print("not on Colab — the archive is at the path above")'''))

    return cells


def closing_cells(colab: bool):
    return [md(r"""## 9 Beyond this notebook

**A different predictor.** Register it and select it by name; the loop does not change.

```python
from binderopt.oracles import register_oracle
from binderopt.oracles.metrics import normalise

@register_oracle("my_predictor")
def _build(**kw):
    return MyAdapter(**kw)          # predict() -> (confidence, normalise(metrics, ...))
```

**A different generator.** Same shape — `@register_generator("name")`, and
`generate(n) -> list[str]`.

**Selectivity against a paralogue.** Give the loop one oracle per receptor and a
`selectivity` objective; nothing else changes. The penalty takes the **maximum** over
off-targets rather than their sum, so a design cannot score well by shedding one paralogue
while keeping another, and it is linear rather than hinged, because a threshold leaves most
candidates with zero gradient.

**Running at scale.** `design.py` with a config in `configs/targets/` is the production
driver: checkpointing, resumption, sequence injection, adaptive weights. It needs the
external tools configured — see `binderopt/external.py`.

**Things worth knowing.**

- One AlphaFold 3 prediction of a 100-residue binder against a 100-residue receptor cost
  0.50–0.97 GPU-minutes in the manuscript runs. Boltz-2 on a T4 is in the same range but
  has not been benchmarked here.
- The oracle is also the evaluator, so the loop optimises the quantity it is judged by.
  `analysis/` computes independent structural measures for that reason.
- Hit criteria are oracle-specific. Re-calibrate when you change backend.
- A bigger starting pool buys a better *first* round, not a better last one.

**Citing the pieces.** Boltz-2 (Passaro et al.), ESM-2 (Lin et al.), and, if you used them
to make the pool, RFdiffusion (Watson et al.) and ProteinMPNN (Dauparas et al.). Their
weights carry their own licences, which differ from this repository's.""")]


def build(colab: bool):
    cells = []
    for section in (title_cells, setup_cells, target_cells, pool_cells, oracle_cells,
                    objective_cells, loop_cells, results_cells, closing_cells):
        cells.extend(section(colab))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    if colab:
        # What makes Colab pick a GPU runtime by itself. Without it the user silently
        # lands on CPU, where Boltz-2 is effectively infinite.
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["colab"] = {"gpuType": "T4", "provenance": [],
                                "include_colab_link": True}
    else:
        nb.metadata["colab"] = {"provenance": []}
    return nb


def main() -> int:
    for colab, name in ((False, "SCALE_demo.ipynb"), (True, "SCALE_demo_colab.ipynb")):
        nb = build(colab)
        path = HERE / name
        nbf.write(nb, str(path))
        n_code = sum(1 for c in nb.cells if c.cell_type == "code")
        print("wrote %-24s %2d cells (%d code)" % (path.name, len(nb.cells), n_code))
    return 0


if __name__ == "__main__":
    sys.exit(main())
