"""Paralogue cross-reactivity of TrkA binders optimised against TrkA alone.

30 designs that meet the TrkA hit criterion (interface PAE <= 1.5 A and interface
pTM >= 0.80 in the original run) folded with AlphaFold 3 against the ligand-binding
domain of each of the three neurotrophin receptors: TrkA (P04629 282-382, the
on-target), TrkB (Q16620 283-385) and TrkC (Q16288 300-404). All three receptor
MSAs and templates come from one AlphaFold Server submission, so the targets differ
only in sequence; the binder chain carries an empty MSA against all three.

a  interface pTM per target, one line per design
b  interface PAE per target, same pairing
c  paired drop from the on-target value

These are predicted quantities. AlphaFold 3 interface pTM is not a measurement of
binding, and its ability to discriminate between homologous receptors is itself
unvalidated; the figure shows where the designs stand under the same predictor the
loop optimised against, nothing more.
"""
import json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE, TICK = 8.0, 6.6
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": BASE, "axes.labelsize": BASE, "axes.titlesize": BASE,
    "legend.fontsize": TICK, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.6, "ytick.major.size": 1.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
T = ["TrkA", "TrkB", "TrkC"]
rows = json.load(open("trkabc_baseline.json"))
assert len(rows) == 90 and all(r["status"] == "ok" for r in rows)
by = {t: {r["rank"]: r for r in rows if r["target"] == t} for t in T}
ranks = sorted(by["TrkA"])
assert all(len(by[t]) == 30 for t in T)
ramp = plt.get_cmap("Blues")
COL = {"TrkA": ramp(0.85), "TrkB": ramp(0.45), "TrkC": ramp(0.62)}

MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 62 * MM))
AX = [fig.add_axes([0.070 + i * 0.315, 0.175, 0.225, 0.715]) for i in range(3)]
stats = {"_definitions": {
    "designs": "30 TrkA hits (interface PAE <= 1.5 A and i-pTM >= 0.80 in the source run)",
    "targets": {"TrkA": "P04629 282-382", "TrkB": "Q16620 283-385",
                "TrkC": "Q16288 300-404"},
    "msa": "one AlphaFold Server submission for all three receptors; binder chain MSA empty",
    "caveat": "predicted interface pTM, not measured binding"}, "per_target": {},
    "paired": {}}

for ax, (col, lab) in zip(AX[:2], [("i_ptm", "Interface pTM"),
                                   ("min_pae_interaction", "Interface PAE (\u00c5)")]):
    for k in ranks:
        ax.plot(range(3), [by[t][k][col] for t in T], color="0.70", lw=0.35,
                zorder=1, alpha=0.8)
    for i, t in enumerate(T):
        v = [by[t][k][col] for k in ranks]
        ax.scatter([i] * len(v), v, s=7, color=COL[t], edgecolor="0.25",
                   linewidths=0.3, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color="0.12", lw=1.1, zorder=4)
        stats["per_target"].setdefault(t, {})[col] = {
            "median": round(float(np.median(v)), 4),
            "min": round(float(min(v)), 4), "max": round(float(max(v)), 4),
            "iqr": [round(float(np.percentile(v, 25)), 4),
                    round(float(np.percentile(v, 75)), 4)], "n": len(v)}
    ax.set_xticks(range(3)); ax.set_xticklabels(T)
    ax.set_xlim(-0.45, 2.45); ax.set_ylabel(lab, labelpad=2)
    ax.tick_params(pad=1.5)

ax = AX[2]
for i, t in enumerate(["TrkB", "TrkC"]):
    d = [by["TrkA"][k]["i_ptm"] - by[t][k]["i_ptm"] for k in ranks]
    ax.scatter([i] * len(d), d, s=7, color=COL[t], edgecolor="0.25",
               linewidths=0.3, zorder=3)
    ax.plot([i - 0.22, i + 0.22], [np.median(d)] * 2, color="0.12", lw=1.1, zorder=4)
    w, p = wilcoxon(d, alternative="greater")
    stats["paired"][t] = {"drop_median": round(float(np.median(d)), 4),
                          "drop_min": round(float(min(d)), 4),
                          "drop_max": round(float(max(d)), 4),
                          "wilcoxon_p": float(p), "n": len(d),
                          "off_target_ge_0.50": int(sum(1 for k in ranks
                                                        if by[t][k]["i_ptm"] >= 0.50)),
                          "off_target_ge_0.70": int(sum(1 for k in ranks
                                                        if by[t][k]["i_ptm"] >= 0.70))}
ax.axhline(0, color="0.5", lw=0.5, zorder=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["TrkB", "TrkC"])
ax.set_xlim(-0.45, 1.45)
ax.set_ylabel("Interface pTM drop\nfrom TrkA", labelpad=2)
ax.tick_params(pad=1.5)
for i, lb in enumerate("abc"):
    fig.text(0.070 + i * 0.315 - 0.055, 0.985, lb, fontsize=9.5, fontweight="bold",
             va="top", ha="left")
fig.savefig("fig_trkabc_baseline.png", dpi=600)
json.dump(stats, open("fig_trkabc_baseline_stats.json", "w"), indent=1)
for t in T:
    s = stats["per_target"][t]
    print("%-5s i-pTM %.3f [%.2f-%.2f]   PAE %.2f [%.2f-%.2f]"
          % (t, s["i_ptm"]["median"], s["i_ptm"]["min"], s["i_ptm"]["max"],
             s["min_pae_interaction"]["median"], s["min_pae_interaction"]["min"],
             s["min_pae_interaction"]["max"]))
for t, s in stats["paired"].items():
    print("%-5s drop %.3f [%.2f-%.2f]  p=%.2g  >=0.50: %d/30  >=0.70: %d/30"
          % (t, s["drop_median"], s["drop_min"], s["drop_max"], s["wilcoxon_p"],
             s["off_target_ge_0.50"], s["off_target_ge_0.70"]))
