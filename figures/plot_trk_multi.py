"""Trk multi-objective selectivity, in the layout of the HER2/EGFR selectivity figure.

Joint arm objective: i-pTM(TrkA) - 2.5 x max(i-pTM(TrkB), i-pTM(TrkC)). The
TrkA-only arm selects on i-pTM(TrkA) alone but folds and records all three
receptors every round, so its off-target curves are measured, not assumed.

a  per-round median interface pTM; colour = arm, line style = receptor
b  per-round median minimum interface PAE, same encoding
c  every optimised design: on-target TrkA interface pTM against the WORSE of the
   two paralogues, with the fraction of each arm that stays at or above the
   starting pool's own level

Threshold in c is 0.40, the round-0 seed pool's median worse-paralogue interface
pTM (0.440 over 100 seeds). A design above it is no more selective than the pool
the loop started from. 73.0% of the seed pool sits above that line.

100 starting binders, each meeting the TrkA hit criterion (interface PAE <= 1.5 A
and interface pTM >= 0.80) in the source run; 15 rounds x 5 new designs, every
design folded against all three receptors; round 0 is the shared pool.

THE CONTROL ARM IS TRUNCATED: AlphaFold 3 died with SIGSEGV on one TrkB complex in
its round 12, so it carries rounds 0-12 against the joint arm's 0-15. It is being
resumed; its curves stop where its data stops and have not converged.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
ARMC = {"joint": "#2b6cb0", "TrkA-only": "#c0392b"}
RECLS = {"TRKA": "-", "TRKC": "--", "TRKB": ":"}
TLAB = {"TRKA": "TrkA", "TRKB": "TrkB", "TRKC": "TrkC"}
THRESH = 0.40

data = {}
for arm, path in (("joint", "trk_multi_joint.json"), ("TrkA-only", "trk_multi_ctrl.json")):
    data[arm] = [r for r in json.load(open(path))["records"] if r["status"] == "complete"]


def worst(r):
    return max(r["metrics"]["TRKB"]["i_ptm"], r["metrics"]["TRKC"]["i_ptm"])


MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 62 * MM))
AX = [fig.add_axes([0.062 + i * 0.238, 0.185, 0.172, 0.700]) for i in range(3)]
stats = {"_definitions": {
    "objective_joint": "i_ptm(TRKA) - 2.5 * max(i_ptm(TRKB), i_ptm(TRKC))",
    "objective_control": "i_ptm(TRKA); paralogues folded and recorded, not optimised",
    "panel_c_threshold": THRESH,
    "panel_c_threshold_rationale": "round-0 seed pool median worse-paralogue i-pTM = 0.440",
    "seeds": 100, "per_round": 5,
    "control_truncated_at_round": 12,
    "control_truncation_cause": "AF3 SIGSEGV on one TrkB complex in round 12"},
    "arms": {}}

for ax, (col, lab) in zip(AX[:2], [("i_ptm", "Interface i-pTM"),
                                   ("min_pae_interaction", "Min interface PAE (\u00c5)")]):
    for arm in ("joint", "TrkA-only"):
        rounds = sorted({r["iteration"] for r in data[arm]})
        for t in ("TRKA", "TRKC", "TRKB"):
            med = [float(np.median([r["metrics"][t][col] for r in data[arm]
                                    if r["iteration"] == k])) for k in rounds]
            ax.plot(rounds, med, RECLS[t], color=ARMC[arm], lw=1.0, zorder=3)
            stats["arms"].setdefault(arm, {}).setdefault(col, {})[t] = {
                "rounds": rounds, "median": [round(v, 4) for v in med]}
    ax.set_xlabel("Iteration", labelpad=2)
    ax.set_ylabel(lab, labelpad=2)
    ax.set_xlim(-0.6, 15.6); ax.set_xticks([0, 5, 10, 15]); ax.tick_params(pad=1.5)
    if col == "i_ptm":
        ax.set_ylim(0, 1.0)

ax = AX[2]
ax.add_patch(Rectangle((0, THRESH), 1.0, 1.0 - THRESH, facecolor="#f6d4cf",
                       alpha=0.55, edgecolor="none", zorder=0))
ax.axhline(THRESH, color="#c0392b", lw=0.6, ls=(0, (3, 2)), zorder=1)
for arm in ("joint", "TrkA-only"):
    opt = [r for r in data[arm] if r["iteration"] >= 1]
    x = [r["metrics"]["TRKA"]["i_ptm"] for r in opt]
    y = [worst(r) for r in opt]
    ax.scatter(x, y, s=5.5, color=ARMC[arm], linewidths=0.2, edgecolor="white", zorder=3)
    frac = 100.0 * float(np.mean(np.array(y) >= THRESH))
    stats["arms"][arm]["optimised_designs"] = len(opt)
    stats["arms"][arm]["pct_worse_paralogue_ge_threshold"] = round(frac, 1)
    stats["arms"][arm]["median_worse_paralogue"] = round(float(np.median(y)), 4)
    stats["arms"][arm]["median_on_target"] = round(float(np.median(x)), 4)
ax.text(0.03, 0.965, "worse paralogue i-pTM $\\geq$ %.2f" % THRESH,
        transform=ax.transAxes, fontsize=TICK - 0.4, color="0.35", va="top")
ax.text(0.03, 0.885, "TrkA-only  %.1f%%" % stats["arms"]["TrkA-only"]["pct_worse_paralogue_ge_threshold"],
        transform=ax.transAxes, fontsize=TICK - 0.4, color=ARMC["TrkA-only"], va="top")
ax.text(0.03, 0.805, "joint  %.1f%%" % stats["arms"]["joint"]["pct_worse_paralogue_ge_threshold"],
        transform=ax.transAxes, fontsize=TICK - 0.4, color=ARMC["joint"], va="top")
ax.set_xlabel("On-target TrkA i-pTM", labelpad=2)
ax.set_ylabel("Worse paralogue i-pTM", labelpad=2)
ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.72); ax.tick_params(pad=1.5)

h = [Line2D([], [], color=ARMC["joint"], lw=1.0, label="joint (+ penalty)"),
     Line2D([], [], color=ARMC["TrkA-only"], lw=1.0, label="TrkA-only"),
     Line2D([], [], color="0.25", lw=1.0, ls="-", label="TrkA complex"),
     Line2D([], [], color="0.25", lw=1.0, ls="--", label="TrkC complex"),
     Line2D([], [], color="0.25", lw=1.0, ls=":", label="TrkB complex")]
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.785, 0.90), ncol=1,
           frameon=False, fontsize=TICK, handlelength=1.6, labelspacing=0.55)
for i, lb in enumerate("abc"):
    fig.text(0.062 + i * 0.238 - 0.046, 0.985, lb, fontsize=9.5, fontweight="bold",
             va="top", ha="left")
fig.savefig("fig_trk_multi.png", dpi=600)
json.dump(stats, open("fig_trk_multi_stats.json", "w"), indent=1)
for arm in ("joint", "TrkA-only"):
    s = stats["arms"][arm]
    print("%-10s n=%d  on-target median %.3f  worse-paralogue median %.3f  >=%.2f: %.1f%%"
          % (arm, s["optimised_designs"], s["median_on_target"],
             s["median_worse_paralogue"], THRESH, s["pct_worse_paralogue_ge_threshold"]))
