"""Generate the local and Colab demonstration notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

#: Fill this in once the repository has a public home; the badge and the clone both need
#: it. Kept as one constant so there is exactly one place to edit.
REPO_URL = "https://github.com/cosmoss8274/SCALE"
COLAB_NOTEBOOK = "notebooks/SCALE_demo_colab.ipynb"

COLAB_SETUP = r"""
#@title Install SCALE and Boltz-2 { display-mode: "form" }
#@markdown Repository to clone. Change this only to use your own fork.
REPO_URL = "__REPO_URL__"  #@param {type:"string"}

import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

started = time.perf_counter()
ROOT = Path("/content/SCALE")
RUN = Path("/content/scale_run")
BOLTZ_ENV = Path("/content/.scale-boltz2-py312")
BOLTZ_VERSION = "2.2.1"

if not ROOT.exists():
    print("Cloning SCALE ...", flush=True)
    subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)
elif not (ROOT / ".git").is_dir():
    raise RuntimeError(f"{ROOT} exists but is not a Git checkout. Choose a different directory.")

CODE = ROOT / "code" if (ROOT / "code").exists() else ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
DATA = CODE / "data"
RUN.mkdir(parents=True, exist_ok=True)

print("Checking notebook dependencies ...", flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "uv", "fair-esm",
     "py3Dmol", "scikit-learn", "biopython", "PyYAML"],
    check=True,
)
uv = [sys.executable, "-m", "uv"]
boltz_python = BOLTZ_ENV / "bin" / "python"
boltz_executable = BOLTZ_ENV / "bin" / "boltz"
if not boltz_python.exists():
    print("Creating the Boltz-2 Python 3.12 environment ...", flush=True)
    subprocess.run(uv + ["venv", "--python", "3.12", str(BOLTZ_ENV)], check=True)

environment_python = json.loads(subprocess.check_output(
    [str(boltz_python), "-c",
     "import json,sys; print(json.dumps(list(sys.version_info[:3])))"],
    text=True,
))
if environment_python[:2] != [3, 12]:
    raise RuntimeError(f"Expected Python 3.12 in {BOLTZ_ENV}; found {environment_python}.")

print(f"Checking Boltz-{BOLTZ_VERSION} dependencies ...", flush=True)
subprocess.run(
    uv + ["pip", "install", "--python", str(boltz_python), f"boltz=={BOLTZ_VERSION}"],
    check=True,
)
subprocess.run(uv + ["pip", "check", "--python", str(boltz_python)], check=True)
subprocess.run([str(boltz_executable), "--help"], stdout=subprocess.DEVNULL, check=True)
backend_gpu = json.loads(subprocess.check_output(
    [str(boltz_python), "-c",
     "import json,torch; "
     "assert torch.cuda.is_available(), 'CUDA is unavailable in the Boltz environment'; "
     "torch.zeros(1, device='cuda').cpu(); "
     "print(json.dumps({'name': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}))"],
    text=True,
))

# Make the isolated CLI available to the existing adapter; no predictor is run here.
env_bin = str(BOLTZ_ENV / "bin")
path_parts = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join([env_bin] + [p for p in path_parts if p != env_bin])
for module in ("numpy", "scipy", "sklearn", "esm", "Bio", "yaml", "py3Dmol"):
    importlib.import_module(module)

backend_versions = json.loads(subprocess.check_output(
    [str(boltz_python), "-c",
     "import json; from importlib.metadata import version; "
     "print(json.dumps({p: version(p) for p in ('boltz','torch','numpy')}))"],
    text=True,
))
if backend_versions["boltz"] != BOLTZ_VERSION:
    raise RuntimeError(f"Expected Boltz {BOLTZ_VERSION}; found {backend_versions['boltz']}.")

environment = {
    "kernel_python": sys.version.split()[0],
    "kernel_torch": metadata.version("torch"),
    "backend_python": ".".join(map(str, environment_python)),
    "backend_packages": backend_versions,
    "backend_gpu": backend_gpu,
    "backend_executable": str(boltz_executable),
    "code_commit": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "setup_seconds": round(time.perf_counter() - started, 1),
}
(RUN / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print("Setup checks passed. No prediction has been run.")
print(json.dumps(environment, indent=2))
print("Code:", CODE)
print("Results:", RUN)
"""


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def form(source: str):
    """A code cell shown as a Colab form: widgets visible, code hidden."""
    cell = code(source)
    cell.metadata["cellView"] = "form"
    return cell


def title_cells(colab: bool):
    badge = f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosmoss8274/SCALE/blob/main/{COLAB_NOTEBOOK})"
    return [
        md(badge),
        md(
            "# SCALE\n\nOptimise a candidate binder pool with structure predictions, an ESM-2 surrogate and guided mutation. Supply a receptor sequence, a CSV or FASTA pool, a predictor and an objective."
        ),
    ]


def setup_cells(colab: bool):
    """Install, clone, GPU check. Only the Colab variant installs anything."""
    cells = [md("## 0 Setup")]

    if colab:
        cells.append(form(COLAB_SETUP.replace("__REPO_URL__", REPO_URL).strip()))

        cells.append(form(r"""#@title 🖥️ Check the GPU { display-mode: "form" }
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
          "or use a larger runtime.")"""))
    else:
        cells.append(
            code(
                r"""# Local run: assumes the environment of requirements.txt and a checkout you are in.
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
       else "none — Boltz-2 needs CUDA; the rest of the notebook still runs")"""
            )
        )

    budget = r"""#@title ⚙️ Budget and plot style { display-mode: "form" }
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
#@markdown Ten initial labels make this a small demonstration. More labels can improve
#@markdown coverage, especially when the starting sequences are genuinely diverse.
N_SEED       = 10   #@param {type:"integer"}
N_ROUNDS     = 5    #@param {type:"integer"}
N_PER_ROUND  = 5    #@param {type:"integer"}
#@markdown Mutants ranked by the surrogate each round. Costs no oracle calls, so this can
#@markdown be large; the default ranks 600 proposals per round.
N_CANDIDATES = 600  #@param {type:"integer"}
#@markdown Default: wider exploration (8/8/4 positions for an 80-aa binder).
#@markdown Sequence-distance fractions and local refinement are configurable.
MIN_POOL_DISTANCE = 0.10  #@param {type:"number"}
MIN_BATCH_DISTANCE = 0.10  #@param {type:"number"}
MIN_HISTORY_DISTANCE = 0.05  #@param {type:"number"}
MAX_PER_FOUNDER = 2  #@param {type:"integer"}
MIN_FOUNDERS = 3  #@param {type:"integer"}
#@markdown Optional: prioritise measured novel parents and refine locally.
USE_NOVEL_REFINEMENT = False  #@param {type:"boolean"}
from binderopt.diversity import DiversityPolicy
policy = DiversityPolicy(MIN_POOL_DISTANCE, MIN_BATCH_DISTANCE,
                         MIN_HISTORY_DISTANCE, MAX_PER_FOUNDER, MIN_FOUNDERS,
                         USE_NOVEL_REFINEMENT)

_total = N_SEED + N_ROUNDS * N_PER_ROUND
print("%d structure predictions in total" % _total)
print("at 1-2 min each that is roughly %d-%d minutes of GPU time" % (_total, 2 * _total))
print("checkpointed after every prediction, so a disconnect costs at most one")"""
    cells.append(form(budget))
    return cells


def target_cells(colab: bool):
    structure_default = "data/structures/TRKA_r00_0.cif"
    return [
        md(r"""## 1 Your target

One receptor sequence. The default is the fifth extracellular domain of TrkA (UniProt
P04629, residues 282–382; the segment modelled in PDB 1WWW)."""),
        form(r"""#@title 1. The receptor { display-mode: "form" }
#@markdown Paste your own receptor sequence to retarget the whole notebook.
TARGET_NAME = "TRKA"  #@param {type:"string"}
TARGET_SEQUENCE = "VSFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAFMDNP"  #@param {type:"string"}

#@markdown Optional remote MSA: sends the receptor sequence to the public MMseqs2 service.
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
    print("receptor MSA: single-sequence mode")"""),
        md("Before spending GPU time, look at what you are designing against."),
        form(
            r'''#@title 1.1 Look at the target { display-mode: "form" }
#@markdown Orientation only — the loop needs just the sequence. Leave blank to skip.
TARGET_STRUCTURE = "'''
            + structure_default
            + r""""  #@param {type:"string"}
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
    print("no structure supplied — skipping. The loop does not need one.")"""
        ),
    ]


def pool_cells(colab: bool):
    return [
        md(
            "## 2 Your starting pool\n\nSupply a CSV with a sequence column, or a FASTA file."
        ),
        form(r"""#@title 2. The starting pool { display-mode: "form" }
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

#@markdown Optional CSV listing an explicit starting batch (same sequence column).
#@markdown This changes the starting condition; record any score-based exclusions.
SEED_BATCH_PATH = ""  #@param {type:"string"}
seed_selection = {"mode": "random", "random_seed": 0}
rng = np.random.default_rng(0)
if SEED_BATCH_PATH:
    seed_path = Path(SEED_BATCH_PATH)
    if not seed_path.is_absolute():
        seed_path = CODE / seed_path
    seed_generator = get_generator("file", path=seed_path, sequence_column=SEQ_COLUMN,
                                   expect_length=BINDER_LENGTH or None)
    seed_designs = seed_generator.generate(10_000)
    if len(seed_designs) != N_SEED or not set(seed_designs) <= set(pool):
        raise ValueError("Explicit starting batch must contain N_SEED unique sequences from the pool")
    seed_selection = {"mode": "explicit", "path": str(seed_path)}
    manifest_path = seed_path.with_suffix(".selection.json")
    if manifest_path.exists():
        seed_selection["selection_manifest"] = json.loads(manifest_path.read_text())
else:
    if len(pool) < N_SEED:
        raise ValueError("Starting pool is smaller than N_SEED")
    seed_designs = [pool[i] for i in rng.choice(len(pool), size=N_SEED, replace=False)]
if len({len(s) for s in pool}) != 1:
    raise ValueError("Distance-constrained substitution search requires equal-length binders")
print("starting-batch selection:", json.dumps(seed_selection))
print("%s: %d designs of %d aa | %d sampled to measure"
      % (generator.describe(), len(pool), len(pool[0]), len(seed_designs)))"""),
    ]


def oracle_cells(colab: bool):
    return [
        md(
            "## 3 Your oracle\n\nSelect a structure predictor through the oracle interface. This example uses Boltz-2."
        ),
        form(r"""#@title 3. The oracle { display-mode: "form" }
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
print("oracle: %s (%s)" % (type(oracle).__name__, oracle.oracle_version))"""),
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
        form(r"""#@title 4. The objective { display-mode: "form" }
from binderopt.objectives import maximise, minimise, weighted, selectivity, describe

OBJECTIVE = "maximise i_ptm"  #@param ["maximise i_ptm", "minimise min_pae_interaction", "weighted i_ptm - 0.1*pae"]

objective = {
    "maximise i_ptm": lambda: maximise("i_ptm"),
    "minimise min_pae_interaction": lambda: minimise("min_pae_interaction"),
    "weighted i_ptm - 0.1*pae": lambda: weighted({"i_ptm": 1.0,
                                                  "min_pae_interaction": -0.1}),
}[OBJECTIVE]()
print("objective:", describe(objective))"""),
    ]


def loop_cells(colab: bool):
    return [
        md(
            r"""## 5 Round 0 — measure the pool

The first predictions. These are the labels the surrogate learns from, so they are spent on
the starting pool rather than on proposals.

Each round is checkpointed to `RUN/history.json`, and re-running the loop cell picks up
where it stopped. Colab disconnects; an hour of GPU time should not die with the tab."""
        ),
        form(r'''#@title 5. Measure the starting pool { display-mode: "form" }
import time
from binderopt.oracles.metrics import describe as describe_metrics

HISTORY_PATH = RUN / "history.json"
history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
# Check run identity before spending initial predictions or reusing a checkpoint.
policy_path = RUN / "search_policy.json"
policy_record = {"policy": policy.to_dict(), "seed_sequences": seed_designs,
                 "reference_pool": pool, "n_per_round": N_PER_ROUND,
                 "n_rounds": N_ROUNDS, "n_candidates": N_CANDIDATES}
if history and not policy_path.exists():
    raise RuntimeError("This RUN contains results from another search; choose a fresh directory")
if policy_path.exists() and json.loads(policy_path.read_text()) != policy_record:
    raise RuntimeError("Search settings changed: choose a fresh RUN directory")
policy_path.write_text(json.dumps(policy_record, indent=2))


def save_history():
    HISTORY_PATH.write_text(json.dumps(history, indent=1))


def measure(sequences, rnd, metadata=None):
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
            details = (metadata or {}).get(seq, {})
            history.append(dict(round=rnd, **record, **details))
            save_history()
            print("  [%d/%d] %s  %.1f min" % (n, len(todo), describe_metrics(record),
                                              (time.time() - t0) / 60), flush=True)
    return [h for h in history if h["round"] == rnd]


t0 = time.time()
from binderopt.oracles.evaluator import sequence_id
seed_metadata = {s: {"founder_id": sequence_id(s, 0), "parent_id": None,
                     "mutations_from_parent": 0} for s in seed_designs}
measured = measure(seed_designs, 0, seed_metadata)
if len(measured) != N_SEED:
    raise RuntimeError("Initial batch is incomplete; resolve prediction failures before continuing")
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
        md(
            "## 6 Optimise\n\nFit the surrogate, propose sequence mutations, rank candidates and evaluate the next batch."
        ),
        form(
            r"""#@title 6. Run the loop { display-mode: "form" }
import random
from binderopt.models import MultiHeadSurrogate
from binderopt.sampling import MultiMetricGuidedSampler
from binderopt.diversity import choose_parents, propose_diverse, select_diverse

random.seed(0); np.random.seed(0)
surrogate = MultiHeadSurrogate(
    esm2_model="esm2_t33_650M_UR50D",
    rf_params={"n_estimators": 300, "random_state": 0, "n_jobs": -1},
    metric_weights={"i_ptm": 1.0},
    calculate_rmsd=False,
)
# With optional novel refinement, a novel parent can take a smaller local step.
# The three global distance constraints stay active for every proposed sequence.
minimum_step = max(1, policy.counts(len(pool[0]))[0])
sampler = MultiMetricGuidedSampler(top_n_for_sampling=5, residue_sampling={
    "min_mutations": minimum_step, "max_mutations": max(minimum_step, minimum_step + 4),
})
for rnd in range(1, N_ROUNDS + 1):
    # Round-specific randomness makes an interrupted round reproducible.
    random.seed(rnd); np.random.seed(rnd)
    existing = [h for h in history if h["round"] == rnd]
    if len(existing) == N_PER_ROUND:
        print("round %d: already complete, skipping" % rnd)
        continue
    t0 = time.time()
    # A persisted full batch allows resuming a partly measured round without reranking.
    plan_path = RUN / ("round_%02d_plan.json" % rnd)
    previous = [h for h in history if h["round"] < rnd]
    parents = choose_parents(previous, objective, n=5, reference_pool=pool, policy=policy)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        selected, candidate_count = plan["selected"], plan["candidate_count"]
    else:
        seqs = [h["sequence"] for h in previous]
        surrogate.train(seqs, {"i_ptm": [h["i_ptm"] for h in previous]})
        proposals = propose_diverse(sampler, parents, pool, previous, policy,
                                    rnd, N_ROUNDS, N_CANDIDATES)
        candidates = [p["sequence"] for p in proposals]
        predictions = surrogate.predict(candidates)
        predicted = surrogate.predict_combined_score(predictions=predictions)
        selected = select_diverse(proposals, predicted, N_PER_ROUND, policy)
        candidate_count = len(candidates)
        plan_path.write_text(json.dumps({"candidate_count": candidate_count,
                                        "selected": selected}, indent=2))
    metadata = {p["sequence"]: {k: v for k, v in p.items() if k != "sequence"}
                for p in selected}
    results = measure([p["sequence"] for p in selected], rnd, metadata)
    if len(results) != N_PER_ROUND:
        raise RuntimeError("Round incomplete; rerun this cell to retry the saved batch")
    predicted = np.array([p["surrogate_score"] for p in selected])
    guided = sum(bool(h.get("residue_metrics")) for h in parents)
    print("round %d | %4d ranked (%d of %d parents structure-guided) | measured %d in "
          "%.1f min | surrogate %.3f-%.3f | best this round %.3f (run best %.3f)"
          % (rnd, candidate_count, guided, len(parents), len(results),
             (time.time() - t0) / 60, predicted.min(), predicted.max(),
             max(objective(h) for h in results), max(objective(h) for h in history)))
    print("  nearest-pool substitutions:", [h["mutations_from_pool"] for h in results],
          "| founding seeds:", len({h["founder_id"] for h in results}))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n%d designs measured over %d rounds" % (len(history), len({h["round"] for h in history})))"""
        ),
    ]


def results_cells(colab: bool):
    cells = [
        md(r"""## 7 The trajectory

Measured objective values and structure-confidence metrics from this run. The
best-so-far curve is cumulative; metric curves show each round's median. Round 0 is
the selected initial batch. Sequence distances remain in the saved candidate table."""),
        form(r"""#@title 7. Trajectory { display-mode: "form" }
from binderopt.demo_reporting import annotate_distances, plot_trajectory
annotate_distances(history, pool)
save_history()
rounds = sorted({h["round"] for h in history})
fig = plot_trajectory(history, objective, policy, RUN / "trajectory.png")
plt.show()

first = [objective(h) for h in history if h["round"] == 0]
last = [objective(h) for h in history if h["round"] == rounds[-1]]
print("objective: round 0 median %.3f → round %d median %.3f (run best %.3f)"
      % (np.median(first), rounds[-1], np.median(last),
         max(objective(h) for h in history)))"""),
        md("## 8 The best design"),
        form(r"""#@title 8. The best design { display-mode: "form" }
import py3Dmol

new_designs = [h for h in history if h["round"] > 0]
top = max(new_designs or history, key=objective)
print("Best new design; overall best including seeds: %.3f" % max(objective(h) for h in history))
print("Nearest starting-pool sequence: %d substitutions, %.1f%% identity"
      % (top["mutations_from_pool"], 100 * top["identity_to_pool"]))
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
    print("\n(no structure file kept for this design)")"""),
        form(
            r"""#@title 8.1 Save the run { display-mode: "form" }
import csv

FIELDS = ["round", "sequence_id", "sequence", "objective", "i_ptm", "ptm_binder",
          "min_pae_interaction", "oracle", "oracle_version", "structure_file",
          "parent_id", "founder_id", "mutations_from_parent", "mutations_from_founder",
          "mutations_from_pool", "identity_to_pool", "surrogate_score"]
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
    "use_msa_server": USE_MSA_SERVER, "seed_selection": seed_selection,
    "diversity_policy": policy.to_dict(), "diffusion_samples": DIFFUSION_SAMPLES,
    "recycling_steps": RECYCLING_STEPS,
}, indent=2))

print("wrote %s" % (RUN / "trajectory.csv"))
print("     %d designs, %d columns" % (len(history), len(FIELDS)))

# A final high-scoring list must also retain sequence and founder diversity.
from binderopt.demo_reporting import diverse_portfolio
portfolio = diverse_portfolio(history, policy, n=N_PER_ROUND)
with open(RUN / "diverse_portfolio.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for h in portfolio:
        writer.writerow({**h, "objective": objective(h)})
print("Diverse measured portfolio (iPTM | substitutions from nearest starting sequence):")
for h in portfolio:
    print("  %s | %.3f | %d | founder %s" %
          (h["sequence_id"], h["i_ptm"], h["mutations_from_pool"], h["founder_id"]))"""
        ),
    ]

    if colab:
        cells.append(form(r"""#@title 📥 Download the results { display-mode: "form" }
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
    print("not on Colab — the archive is at the path above")"""))

    return cells


def closing_cells(colab: bool):
    return [
        md(
            "## Further configuration\n\nRegister custom predictors and generators using the [extension guide](https://github.com/cosmoss8274/SCALE/blob/main/docs/extending.md). Cluster settings are described in [configuration](https://github.com/cosmoss8274/SCALE/blob/main/docs/configuration.md)."
        )
    ]


def build(colab: bool):
    cells = []
    for section in (
        title_cells,
        setup_cells,
        target_cells,
        pool_cells,
        oracle_cells,
        objective_cells,
        loop_cells,
        results_cells,
        closing_cells,
    ):
        cells.extend(section(colab))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
    )
    if colab:
        # What makes Colab pick a GPU runtime by itself. Without it the user silently
        # lands on CPU, where Boltz-2 is effectively infinite.
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["colab"] = {
            "gpuType": "T4",
            "provenance": [],
            "include_colab_link": True,
        }
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
