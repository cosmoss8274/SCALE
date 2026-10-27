#!/usr/bin/env python3
"""Multi-objective selectivity loop: one on-target, several counter-targets.

Objective, per candidate binder:

    score = i-pTM(TrkA) - penalty * max( i-pTM(TrkB), i-pTM(TrkC) )

The counter-target term is the WORST case over the paralogues, not their sum: with
a sum the optimiser can be paid for suppressing one paralogue while the other stays
high, which is exactly the failure mode pan-Trk selectivity has to avoid. The
penalty is linear with no threshold, the form shown to work for HER2/EGFR; the
hinge form was tested there and left 97% of candidates with zero gradient.

Two arms share everything except the selection rule:
  TrkA-only   select on i-pTM(TrkA) alone; the paralogues are still folded and
              recorded every round, so the control has measured off-target values
  joint       select on the score above

This script does NOT modify the HER2/EGFR code path. It reuses the target-agnostic
components of that pipeline unchanged - StrictAF3Interface, CorrectedAF3Evaluator,
MultiHeadSurrogate with the archived ESM2 3B encoder and shared embedding cache,
and MultiMetricGuidedSampler with DEFAULT_RESIDUE_SAMPLING - so round-for-round the
machinery is the same as Figure 6's.

Deviation from the HER2/EGFR runs, recorded in the manifest: ProteinMPNN pose NLL
is not scored here. It is not part of the objective and PoseNLLScorer is keyed by
target name, so wiring new targets into it is a separate change.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

POSITIVE = "TRKA"
NEGATIVES = ("TRKB", "TRKC")
ARMS = ("TrkA-only", "joint")


def sha_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name,
                                     suffix=".tmp", delete=False) as handle:
        json.dump(obj, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
        tmp = Path(handle.name)
    tmp.replace(path)


def score_of(record: dict, arm: str, penalty: float) -> float:
    pos = record["metrics"][POSITIVE]["i_ptm"]
    if arm == "TrkA-only":
        return float(pos)
    worst = max(record["metrics"][n]["i_ptm"] for n in NEGATIVES)
    return float(pos) - penalty * float(worst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--targets-dir", type=Path, required=True,
                   help="directory holding TRK*_only_data.json and targets.json")
    p.add_argument("--seed-json", type=Path, required=True,
                   help="JSON list of starting binder sequences")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--per-round", type=int, required=True)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--penalty", type=float, default=2.5)
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--esm2-model", default="esm2_t36_3B_UR50D")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if not os.environ.get("PBS_JOBID"):
        p.error("this run requires a PBS compute-node allocation")
    if min(a.rounds, a.per_round, a.top_n) <= 0:
        p.error("rounds, per-round and top-n must be positive")
    if not 0 < a.penalty <= 100:
        p.error("penalty must be positive")

    root = a.project_root.resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "repo"))

    # binderopt.models decides at IMPORT time whether ESM2 is importable, so the
    # local loader has to be installed into sys.modules before that import, not
    # after it - the same order run_three_arms.py uses.
    from experiments.new_selectivity.esm_local import install_local_esm_loader
    esm_hashes = install_local_esm_loader(root)

    from binderopt.models import MultiHeadSurrogate
    from binderopt.sampling import MultiMetricGuidedSampler, DEFAULT_RESIDUE_SAMPLING
    from experiments.new_selectivity.embedding_cache import CachedEmbeddingExtractor
    from experiments.new_selectivity.metrics_corrected import (CorrectedAF3Evaluator,
                                                              CorrectedLossCalculator)
    from experiments.new_selectivity.strict_af3 import StrictAF3Interface
    if a.esm2_model != "esm2_t36_3B_UR50D":
        raise ValueError("this run is pinned to the archived ESM2 3B model")

    spec = json.loads((a.targets_dir / "targets.json").read_text())
    tag_of = {"TRKA": "TrkA", "TRKB": "TrkB", "TRKC": "TrkC"}
    sequences = {t: spec[tag_of[t]]["sequence"] for t in (POSITIVE,) + NEGATIVES}
    assert len({*sequences.values()}) == 3, "target sequences must be distinct"

    seeds = json.loads(a.seed_json.read_text())
    seeds = [s["sequence"] if isinstance(s, dict) else s for s in seeds]
    seeds = list(dict.fromkeys(seeds))
    if a.max_seeds:
        seeds = seeds[:a.max_seeds]
    if not seeds:
        p.error("no seed sequences")

    shared_cache = root / "runs/trkabc_af3_cache"
    evaluators = {}
    for t in (POSITIVE,) + NEGATIVES:
        iface = StrictAF3Interface(
            target_name=t, target_sequence=sequences[t],
            msa_json=a.targets_dir / ("%s_only_data.json" % t),
            expected_msa_sha256=None, project_root=root, shared_cache=shared_cache,
            baseline={}, af3_results_dir=out / "raw_af3" / t,
            msa_cache_dir=root / "cache/af3_msa" / t)
        evaluators[t] = CorrectedAF3Evaluator(iface, str(out / "structures" / t),
                                              pose_nll_scorer=None, target_name=t)

    cpus = int(os.environ.get("PBS_NCPUS") or len(os.sched_getaffinity(0)))
    rf = {"n_estimators": 100, "criterion": "friedman_mse", "max_depth": None,
          "min_samples_split": 2, "min_samples_leaf": 1, "bootstrap": True,
          "n_jobs": max(1, min(24, cpus)), "random_state": 1}
    base = MultiHeadSurrogate(esm2_model=a.esm2_model, rf_params=rf,
                              metric_weights={"i_ptm": 1.0}, calculate_rmsd=False,
                              device=a.device)
    base.embedding_extractor = CachedEmbeddingExtractor(
        base.embedding_extractor, root / "cache/esm2_embeddings",
        esm_hashes["esm2_t36_3B_UR50D.pt"])
    surrogates = {POSITIVE: base}
    for t in NEGATIVES:                       # one RF head set per target, one encoder
        s = copy.copy(base)
        s.heads = copy.deepcopy(base.heads)
        s.metrics = list(base.metrics)
        s.training_history = []
        s.last_train_sequences = None
        s.last_train_embeddings = None
        surrogates[t] = s

    ckpt_path = out / "latest_checkpoint.json"
    if ckpt_path.is_file():
        state = json.loads(ckpt_path.read_text())
        records = state["records"]
        last_round = state["last_completed_round"]
        print("resuming at round %d with %d records" % (last_round, len(records)), flush=True)
    else:
        records, last_round = [], -1
        atomic_json(out / "manifest.json", {
            "arm": a.arm, "rounds": a.rounds, "per_round": a.per_round,
            "top_n": a.top_n, "seed": a.seed, "penalty": a.penalty,
            "objective": "i_ptm(%s) - %.3f * max(%s)" % (POSITIVE, a.penalty,
                                                         ", ".join(NEGATIVES)),
            "positive": POSITIVE, "negatives": list(NEGATIVES),
            "target_sequence_sha256": {t: sha_sequence(s) for t, s in sequences.items()},
            "msa_sha256": {t: hashlib.sha256(
                (a.targets_dir / ("%s_only_data.json" % t)).read_bytes()).hexdigest()
                for t in (POSITIVE,) + NEGATIVES},
            "seed_count": len(seeds),
            "seed_sequences_sha256": sha_sequence("".join(sorted(seeds))),
            "sampling": DEFAULT_RESIDUE_SAMPLING,
            "esm2_model": a.esm2_model, "rf_params": rf,
            "pose_nll_scored": False,
            "af3_shared_cache": str(shared_cache)})


    def evaluate(seqs: list[str], iteration: int) -> list[dict]:
        # Only COMPLETE records count as done. A record left behind by a transient
        # AF3 crash must be retried on resume, not skipped - skipping it would
        # leave the round permanently short and fail again immediately.
        done = {r["sequence"] for r in records if r["status"] == "complete"}
        stale = [r for r in records
                 if r["status"] != "complete" and r["sequence"] in set(seqs)]
        for r in stale:
            records.remove(r)
        if stale:
            print("retrying %d previously failed sequence(s)" % len(stale), flush=True)
        todo = [s for s in seqs if s not in done]
        made = []
        for s in todo:
            rec = {"iteration": iteration, "sequence": s,
                   "sequence_sha256": sha_sequence(s), "metrics": {}, "files": {},
                   "status": "incomplete"}
            try:
                for t in (POSITIVE,) + NEGATIVES:
                    # AF3 occasionally dies with SIGSEGV on a single complex; one
                    # retry costs a fold and saves the whole arm.
                    try:
                        res = evaluators[t].evaluate([s])[0]
                    except Exception as first:
                        print("af3_retry target=%s sha=%s after %s"
                              % (t, rec["sequence_sha256"][:12], type(first).__name__),
                              flush=True)
                        res = evaluators[t].evaluate([s])[0]
                    rec["metrics"][t] = {
                        "i_ptm": float(res["i_ptm"]),
                        "ptm_binder": float(res.get("ptm_binder", "nan")),
                        "min_pae_interaction": float(res["min_pae_interaction"]),
                        "composite_score": float(res.get("composite_score", "nan"))}
                    rec["files"][t] = {"structure_file": str(res["structure_file"]),
                                       "json_file": str(res["json_file"])}
                rec["status"] = "complete"
                rec["score"] = score_of(rec, a.arm, a.penalty)
                rec["score_joint"] = score_of(rec, "joint", a.penalty)
            except Exception as exc:
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
                print("eval_failed sha=%s %s" % (rec["sequence_sha256"][:12],
                                                 rec["error"][:200]), flush=True)
            records.append(rec)
            made.append(rec)
            atomic_json(ckpt_path, {"last_completed_round": last_round,
                                    "records": records})
        return made


    def fit() -> None:
        done = [r for r in records if r["status"] == "complete"]
        if not done:
            return
        seqs = [r["sequence"] for r in done]
        # MultiHeadSurrogate.train takes Dict[metric -> list of values], one list
        # per metric, NOT a list of per-sequence dicts.
        def labels(target):
            return {"i_ptm": [float(r["metrics"][target]["i_ptm"]) for r in done]}
        surrogates[POSITIVE].train(seqs, labels(POSITIVE))
        if "i_ptm" not in surrogates[POSITIVE].metrics:
            raise RuntimeError("positive surrogate did not train an i_ptm head")
        emb = surrogates[POSITIVE].last_train_embeddings
        for t in NEGATIVES:
            sur = surrogates[t]
            if emb is not None and surrogates[POSITIVE].last_train_sequences == seqs:
                sur.last_train_sequences = list(seqs)
                sur.last_train_embeddings = emb
                sur._train_random_forest(emb, labels(t))
            else:
                sur.train(seqs, labels(t))
            if "i_ptm" not in sur.metrics:
                raise RuntimeError("%s surrogate did not train an i_ptm head" % t)


    def propose(iteration: int, sampler) -> list[str]:
        done = sorted((r for r in records if r["status"] == "complete"),
                      key=lambda r: -score_of(r, a.arm, a.penalty))[:a.top_n]
        parents, metrics = [], []
        for r in done:
            try:
                calc = CorrectedLossCalculator(r["files"][POSITIVE]["structure_file"],
                                               r["files"][POSITIVE]["json_file"])
                m = calc.get_residue_level_metrics()
                if m:
                    parents.append(r["sequence"])
                    metrics.append(m)
            except Exception as exc:
                print("residue_metric_unavailable sha=%s %s"
                      % (r["sequence_sha256"][:12], exc), flush=True)
        if not parents:
            parents = [r["sequence"] for r in done]
            print("sampler_mode=random_fallback", flush=True)
        else:
            print("sampler_mode=archived_guided parents=%d" % len(parents), flush=True)
        import numpy as np
        random.seed(a.seed + 1000003 * iteration)
        np.random.seed((a.seed + 1000003 * iteration) % (2 ** 32))
        gen = sampler.generate_candidates(parents, metrics,
                                          current_iteration=iteration,
                                          max_iterations=a.rounds)
        seen = {r["sequence"] for r in records}
        return sorted(set(gen) - seen)


    def select(cands: list[str], iteration: int) -> list[str]:
        # Go through MultiHeadSurrogate.predict rather than indexing .heads
        # directly: the head keys follow the trained metric list, so a literal
        # "i_ptm" lookup is not guaranteed. The embedding extractor is cached, so
        # calling predict once per target does not re-embed the candidates.
        pred = {}
        for t in (POSITIVE,) + NEGATIVES:
            got = surrogates[t].predict(cands)   # not `out`: that is the run directory
            if "i_ptm" not in got or len(got["i_ptm"]) != len(cands):
                raise RuntimeError("%s surrogate did not return one i_ptm per candidate"
                                   % t)
            pred[t] = got["i_ptm"]
        rows = []
        for i, s in enumerate(cands):
            pos = float(pred[POSITIVE][i])
            worst = max(float(pred[t][i]) for t in NEGATIVES)
            sc = pos if a.arm == "TrkA-only" else pos - a.penalty * worst
            rows.append({"sequence": s, "sequence_sha256": sha_sequence(s),
                         "predicted_positive": pos, "predicted_worst_negative": worst,
                         **{"predicted_%s" % t: float(pred[t][i])
                            for t in (POSITIVE,) + NEGATIVES},
                         "predicted_score": sc})
        rows.sort(key=lambda r: -r["predicted_score"])
        for rank, r in enumerate(rows):
            r["selection_rank"] = rank
            r["selected"] = rank < a.per_round
        path = out / ("selection_iter_%03d.csv" % iteration)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return [r["sequence"] for r in rows[:a.per_round]]


    t0 = time.time()
    if last_round < 0:
        print("bootstrap: folding %d seeds against %d targets"
              % (len(seeds), 1 + len(NEGATIVES)), flush=True)
        evaluate(seeds, 0)
        ok = [r for r in records if r["status"] == "complete"]
        if len(ok) < max(2, a.top_n // 3):
            raise RuntimeError("bootstrap produced only %d complete seeds" % len(ok))
        fit()
        last_round = 0
        atomic_json(ckpt_path, {"last_completed_round": last_round, "records": records})
        atomic_json(out / "checkpoint_iter_000.json",
                    {"last_completed_round": 0, "records": records})
        print("bootstrap complete: %d/%d seeds, %.0fs" % (len(ok), len(seeds),
                                                          time.time() - t0), flush=True)

    sampler = MultiMetricGuidedSampler(residue_sampling=DEFAULT_RESIDUE_SAMPLING,
                                       top_n_for_sampling=a.top_n)
    for it in range(last_round + 1, a.rounds + 1):
        cands = propose(it, sampler)
        if not cands:
            raise RuntimeError("round %d produced no new candidates" % it)
        picked = select(cands, it)
        print("arm=%s iteration=%d candidates=%d selected=%d"
              % (a.arm, it, len(cands), len(picked)), flush=True)
        new = evaluate(picked, it)
        got = [r for r in records if r["iteration"] == it and r["status"] == "complete"]
        if len(got) != a.per_round:
            raise RuntimeError("round %d completed %d of %d" % (it, len(got), a.per_round))
        fit()
        last_round = it
        atomic_json(ckpt_path, {"last_completed_round": it, "records": records})
        atomic_json(out / ("checkpoint_iter_%03d.json" % it),
                    {"last_completed_round": it, "records": records})
        best = max(got, key=lambda r: score_of(r, a.arm, a.penalty))
        print("arm=%s iteration=%d best_positive=%.3f best_worst_negative=%.3f "
              "best_score=%.3f elapsed=%.0fs"
              % (a.arm, it, best["metrics"][POSITIVE]["i_ptm"],
                 max(best["metrics"][n]["i_ptm"] for n in NEGATIVES),
                 score_of(best, a.arm, a.penalty), time.time() - t0), flush=True)
    print(json.dumps({"arm": a.arm, "completed_rounds": last_round,
                      "records": len(records),
                      "complete": sum(1 for r in records if r["status"] == "complete")}))


if __name__ == "__main__":
    main()
