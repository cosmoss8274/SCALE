"""Baseline paralog cross-reactivity: 30 verified TrkA binders folded against
TrkA, TrkB and TrkC with AlphaFold 3.

No optimisation and no surrogate: this measures where the existing TrkA-only
binders already stand on the two paralogues before any selectivity objective is
applied. All three receptors use MSAs and templates generated in one AlphaFold
Server submission (folds_2026_10_26_14_37.zip) so the three targets differ only in
sequence, and all three go through StrictAF3Interface with target_name not 'EGFR',
so the binder chain carries an empty MSA against every one of them - symmetric,
unlike the HER2/EGFR pair.
"""
import json, os, shutil, sys, time
from pathlib import Path

ROOT = Path("/data/projects/11004455/e0983548/scale_1019_2026")
RUN = ROOT / "runs/trkabc_baseline"
sys.path.insert(0, str(ROOT / "repo"))
from experiments.new_selectivity.strict_af3 import StrictAF3Interface

NAMES = {"TrkA": "TRKA", "TrkB": "TRKB", "TrkC": "TRKC"}
binders = json.loads(Path("binders.json").read_text())
targets = json.loads(Path("targets.json").read_text())
assert len(binders) == 30, len(binders)

msa_root = RUN / "msa"
for nm, tag in NAMES.items():
    d = msa_root / tag
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy("%s_only_data.json" % tag, d / ("%s_only_data.json" % tag))

rows, t0 = [], time.time()
for nm, tag in NAMES.items():
    seq = targets[nm]["sequence"]
    iface = StrictAF3Interface(
        target_name=tag, target_sequence=seq,
        msa_json=msa_root / tag / ("%s_only_data.json" % tag),
        expected_msa_sha256=None, project_root=ROOT,
        shared_cache=ROOT / "runs/trkabc_af3_cache", baseline={},
        af3_results_dir=RUN / "raw_af3" / tag, msa_cache_dir=RUN / "msa_cache" / tag)
    for b in binders:
        sid = "%s_b%02d" % (tag, b["rank"])
        rec = {"target": nm, "target_tag": tag, "rank": b["rank"],
               "source_round": b["round"], "sequence": b["sequence"],
               "i_ptm_TrkA_oldMSA": b["i_ptm_TrkA_oldMSA"], "sequence_id": sid}
        try:
            score, m = iface.predict(b["sequence"], True,
                                    str(RUN / "structures" / tag), sid)
            pae = m.get("chain_pair_pae_min") or []
            off = [pae[i][j] for i in range(len(pae)) for j in range(len(pae))
                   if i != j] if pae else []
            rec.update({"ranking_score": score, "i_ptm": m["iptm"], "ptm": m["ptm"],
                        "chain_ptm": m.get("chain_ptm"),
                        "chain_pair_pae_min": pae,
                        "min_pae_interaction": (min(off) if off else None),
                        "status": "ok"})
        except Exception as exc:
            rec.update({"status": "failed", "error": "%s: %s" % (type(exc).__name__, exc)})
            print("FAIL", sid, rec["error"][:200], flush=True)
        rows.append(rec)
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / "baseline_partial.json").write_text(json.dumps(rows, indent=1))
        print("%s %-12s %4.0fs  i_ptm=%s" % (sid, rec["status"], time.time() - t0,
                                            rec.get("i_ptm")), flush=True)
(RUN / "trkabc_baseline.json").write_text(json.dumps(rows, indent=1))
Path("out").mkdir(exist_ok=True)
shutil.copy(RUN / "trkabc_baseline.json", "out/trkabc_baseline.json")
ok = [r for r in rows if r["status"] == "ok"]
print("done %d/%d ok in %.0fs" % (len(ok), len(rows), time.time() - t0))
for nm in NAMES:
    v = [r["i_ptm"] for r in ok if r["target"] == nm]
    if v:
        v = sorted(v)
        print("%-5s n=%2d  median i-pTM %.3f  range %.2f-%.2f"
              % (nm, len(v), v[len(v)//2], v[0], v[-1]))
