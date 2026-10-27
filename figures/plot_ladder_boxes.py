# SI figure: pool-size ladder as per-round distributions.
#
# 4 metrics (rows) x 5 targets (columns). In each panel, boxes for optimisation
# rounds 1, 5, 10 and 15 at three starting-pool sizes (100, 200, 500 candidates);
# n = 30 evaluated candidates per round per tier. Boxes are matplotlib defaults:
# median line, quartile box, whiskers to 1.5 IQR, outliers not drawn.
#
# Rounds 1/5/10/15 rather than all eight odd rounds: five target columns across
# 183 mm leaves about 30 mm per panel, and eight rounds x three tiers would put 24
# boxes in that space, roughly 1.2 mm each, which does not survive print.
#
# Metrics are those used in the main pool-size figure (interface pTM, binder pTM)
# and its line-plot companion (the composite objective the loop maximises, and the
# minimum interface PAE), in that order.
#
# MYBL2's interface pTM in this table is the BINDER-SIDE value,
# min(chain_pair_iptm[2][0], chain_pair_iptm[2][1]), the same quantity Figures 2 and
# 3 use; it spans 0.38-0.89 rather than the compressed 0.81-0.93 of the global value.
# Its composite still embeds the global i-pTM, so for that target the composite
# column is not a function of the interface pTM column.
import csv
import collections
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE, TICK = 8.0, 6.2
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": BASE, "axes.labelsize": BASE, "axes.titlesize": BASE,
    "legend.fontsize": TICK, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.6, "ytick.major.size": 1.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
TARGETS = ["BHRF1", "SC2RBD", "TRKA", "MYBL2", "HER2"]
TLAB = {"SC2RBD": "SARS-CoV-2 RBD", "TRKA": "TrkA"}
METRICS = [("i_ptm", "Interface pTM", "higher = better"),
           ("ptm_binder", "Binder pTM", "higher = better"),
           ("composite", "Composite objective", "higher = better"),
           ("min_pae_interaction", "Interface PAE (\u00c5)", "lower = better")]
TIERS = [100, 200, 500]
ROUNDS = [1, 5, 10, 15]
ramp = plt.get_cmap("Blues")
TCOL = {t: ramp(0.26 + 0.56 * i / (len(TIERS) - 1)) for i, t in enumerate(TIERS)}

rows = list(csv.DictReader(open("hpc/scp-46579a3478b0b9e5/ladder_percandidate.csv")))
assert len(rows) == 23625, len(rows)
assert len({(r["target"], int(r["tier"])) for r in rows}) == 35
vals = collections.defaultdict(list)
for r in rows:
    for col, _, _ in METRICS:
        v = r.get(col)
        if v not in ("", "nan", None):
            vals[(r["target"], int(r["tier"]), int(r["round"]), col)].append(float(v))

MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 142 * MM))
CX = [0.072 + i * 0.167 for i in range(5)]
CW = 0.123
RY = [0.766, 0.543, 0.320, 0.097]
RH = 0.182
W = 0.24

rep = {"_definitions": {
    "boxes": "matplotlib default: median, quartile box, whiskers to 1.5 IQR",
    "n_per_box": "30 evaluated candidates per round per tier",
    "rounds": ROUNDS, "tiers": TIERS,
    "mybl2_i_ptm": "binder-side min(chain_pair_iptm[2][0], chain_pair_iptm[2][1])",
    "arms_in_table": 35}, "targets": {}}

for mi, (col, lab, _sense) in enumerate(METRICS):
    for ti, t in enumerate(TARGETS):
        ax = fig.add_axes([CX[ti], RY[mi], CW, RH])
        pos, data, cols = [], [], []
        for k, rd in enumerate(ROUNDS):
            for j, tier in enumerate(TIERS):
                v = vals[(t, tier, rd, col)]
                if not v:
                    continue
                pos.append(k + (j - 1) * W)
                data.append(v); cols.append(TCOL[tier])
        bp = ax.boxplot(data, positions=pos, widths=W * 0.86, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor("0.25"); patch.set_linewidth(0.45)
        for key in ("whiskers", "caps", "medians"):
            for ln in bp[key]:
                ln.set_color("0.25"); ln.set_linewidth(0.45)
        ax.set_xticks(range(len(ROUNDS)))
        ax.set_xlim(-0.5, len(ROUNDS) - 0.5)
        ax.tick_params(pad=1.2)
        if mi == len(METRICS) - 1:
            ax.set_xticklabels([str(r) for r in ROUNDS])
            ax.set_xlabel("Optimisation round", labelpad=2, fontsize=TICK + 0.8)
        else:
            ax.set_xticklabels([])
        if ti == 0:
            ax.set_ylabel(lab, labelpad=2, fontsize=TICK + 0.4)
        if mi == 0:
            ax.set_title(TLAB.get(t, t), pad=3, fontsize=TICK + 1.2)
        rep["targets"].setdefault(t, {})[col] = {
            str(rd): {str(tier): ([round(float(np.median(vals[(t, tier, rd, col)])), 4),
                                   len(vals[(t, tier, rd, col)])]
                                  if vals[(t, tier, rd, col)] else None)
                      for tier in TIERS} for rd in ROUNDS}
    fig.text(0.006, RY[mi] + RH + 0.010, "abcde"[mi], fontsize=9.5, fontweight="bold",
             va="top", ha="left")

h = [Patch(facecolor=TCOL[k], edgecolor="0.25", lw=0.45, label="%d" % k) for k in TIERS]
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.908, 0.945), ncol=1,
           frameon=False, fontsize=TICK + 0.4, handlelength=1.1, handletextpad=0.45,
           labelspacing=0.55, title="Starting pool", title_fontsize=TICK + 0.4,
           alignment="left")

fig.savefig("fig_ladder_boxes.png", dpi=600)
json.dump(rep, open("fig_ladder_boxes_stats.json", "w"), indent=1)
print("rows=metrics, cols=targets; panels:", len(METRICS) * len(TARGETS))
