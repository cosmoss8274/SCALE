"""Trk multi-objective selectivity: curves plus one lineage against all three receptors.

a-c as in the companion figure. d-f: the same binder lineage - round 0, round 4,
round 15 of the joint arm, 74/80 and 75/80 identical step to step, 72/80 end to end -
predicted against TrkA (d), TrkB (e) and TrkC (f). Within each row the three
complexes are superposed on the receptor and rendered from one camera, cropped to a
common box, so the only thing that moves is the binder. Receptor surface grey with
the six N- and six C-terminal residues hidden (AF3 places those flexible ends
differently between complexes of the same receptor); binder cartoon coloured by
pLDDT. In the round-15 TrkC panel a disordered receptor loop is predicted swung
clear of the body - a feature of that single prediction, not a different construct.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from PIL import Image

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
THRESH = 0.40
PICK = json.load(open("trkstr/pick.json"))
data = {}
for arm, path in (("joint", "trk_multi_joint.json"), ("TrkA-only", "trk_multi_ctrl.json")):
    data[arm] = [r for r in json.load(open(path))["records"] if r["status"] == "complete"]
worst = lambda r: max(r["metrics"]["TRKB"]["i_ptm"], r["metrics"]["TRKC"]["i_ptm"])

MM = 1 / 25.4
fig = plt.figure(figsize=(183.0 * MM, 152.0 * MM))
TOPY, TOPH = 0.818, 0.150
AX = [fig.add_axes([0.062 + i * 0.238, TOPY, 0.172, TOPH]) for i in range(3)]
stats = {"panel_c_threshold": THRESH, "lineage": PICK, "arms": {}}

for ax, (col, lab) in zip(AX[:2], [("i_ptm", "Interface i-pTM"),
                                   ("min_pae_interaction", "Min interface PAE (\u00c5)")]):
    for arm in ("joint", "TrkA-only"):
        rounds = sorted({r["iteration"] for r in data[arm]})
        for t in ("TRKA", "TRKC", "TRKB"):
            med = [float(np.median([r["metrics"][t][col] for r in data[arm]
                                    if r["iteration"] == k])) for k in rounds]
            ax.plot(rounds, med, RECLS[t], color=ARMC[arm], lw=1.0, zorder=3)
            stats["arms"].setdefault(arm, {}).setdefault(col, {})[t] = \
                {"rounds": rounds, "median": [round(v, 4) for v in med]}
    ax.set_xlabel("Iteration", labelpad=2); ax.set_ylabel(lab, labelpad=2)
    ax.set_xlim(-0.6, 15.6); ax.set_xticks([0, 5, 10, 15]); ax.tick_params(pad=1.5)
    if col == "i_ptm":
        ax.set_ylim(0, 1.0)

ax = AX[2]
ax.add_patch(Rectangle((0, THRESH), 1.0, 1.0 - THRESH, facecolor="#f6d4cf",
                       alpha=0.55, edgecolor="none", zorder=0))
ax.axhline(THRESH, color="#c0392b", lw=0.6, ls=(0, (3, 2)), zorder=1)
for arm in ("joint", "TrkA-only"):
    opt = [r for r in data[arm] if r["iteration"] >= 1]
    y = [worst(r) for r in opt]
    ax.scatter([r["metrics"]["TRKA"]["i_ptm"] for r in opt], y, s=5.5,
               color=ARMC[arm], linewidths=0.2, edgecolor="white", zorder=3)
    stats["arms"][arm]["pct_ge_threshold"] = round(100.0 * float(np.mean(np.array(y) >= THRESH)), 1)
    stats["arms"][arm]["n_optimised"] = len(opt)
ax.text(0.03, 0.965, "worse paralogue i-pTM $\\geq$ %.2f" % THRESH, transform=ax.transAxes,
        fontsize=TICK - 0.4, color="0.35", va="top")
ax.text(0.03, 0.885, "TrkA-only  %.1f%%" % stats["arms"]["TrkA-only"]["pct_ge_threshold"],
        transform=ax.transAxes, fontsize=TICK - 0.4, color=ARMC["TrkA-only"], va="top")
ax.text(0.03, 0.805, "joint  %.1f%%" % stats["arms"]["joint"]["pct_ge_threshold"],
        transform=ax.transAxes, fontsize=TICK - 0.4, color=ARMC["joint"], va="top")
ax.set_xlabel("On-target TrkA i-pTM", labelpad=2)
ax.set_ylabel("Worse paralogue\ni-pTM", labelpad=2)
ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.72); ax.tick_params(pad=1.5)

h = [Line2D([], [], color=ARMC["joint"], lw=1.0, label="joint (+ penalty)"),
     Line2D([], [], color=ARMC["TrkA-only"], lw=1.0, label="TrkA-only"),
     Line2D([], [], color="0.25", lw=1.0, ls="-", label="TrkA complex"),
     Line2D([], [], color="0.25", lw=1.0, ls="--", label="TrkC complex"),
     Line2D([], [], color="0.25", lw=1.0, ls=":", label="TrkB complex")]
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.726, TOPY + TOPH + 0.010),
           ncol=1, frameon=False, fontsize=TICK, handlelength=1.6, labelspacing=0.55)

ROWS = [("TRKA", "On-target TrkA", PICK["trka"], "d"),
        ("TRKB", "Off-target TrkB", PICK["trkb"], "e"),
        ("TRKC", "Off-target TrkC", PICK["trkc"], "f")]
CW, CH = 0.1720, 0.2241
CXs = [0.062, 0.3, 0.538]
RYs = [0.5302, 0.2901, 0.05]
for (tag, rlab, vals, letter), ry in zip(ROWS, RYs):
    for j, (cx, v) in enumerate(zip(CXs, vals)):
        a = fig.add_axes([cx, ry, CW, CH]); a.axis("off")
        im = np.asarray(Image.open("trk9c/%s_%d.png" % (tag, j)).convert("RGBA"))
        a.imshow(im)
        # Put the label over the molecule itself, not over the panel box: the nine
        # panels share one crop box, so the amount of empty space above each model
        # differs and a label pinned to the box would float away from some of them.
        nz = im[:, :, 3] > 8
        rr = np.where(nz.any(axis=1))[0]; cc = np.where(nz.any(axis=0))[0]
        a.text((cc.min() + cc.max()) / 2.0, rr.min() - 0.018 * im.shape[0],
               "i-pTM %.2f" % v, fontsize=TICK, va="bottom", ha="center")
        if j == 0:
            a.text(-0.045, 0.5, rlab, transform=a.transAxes, rotation=90,
                   va="center", ha="center", fontsize=TICK + 0.4)
    fig.text(0.012, ry + CH - 0.006, letter, fontsize=9.5, fontweight="bold",
             va="top", ha="left")
for cx, lab in zip(CXs, ["round-0 seed", "iteration 4", "iteration 15"]):
    fig.text(cx + CW / 2, 0.038, lab, ha="center", va="top", fontsize=TICK + 0.6)

PL = [("< 50", "#F4761F"), ("50\u201370", "#F2C230"), ("70\u201390", "#7FC4E8"),
      ("> 90", "#1F5FBF")]
fig.text(0.726, 0.744, "Binder pLDDT", fontsize=TICK + 0.8, va="top", ha="left")
fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in PL],
           loc="upper left", bbox_to_anchor=(0.726, 0.728), frameon=False,
           fontsize=TICK, handlelength=1.1, handleheight=1.0, labelspacing=0.45)
for i, lb in enumerate("abc"):
    fig.text(0.062 + i * 0.238 - 0.048, 0.992, lb, fontsize=9.5, fontweight="bold",
             va="top", ha="left")
fig.savefig("fig_trk_full.png", dpi=600)
json.dump(stats, open("fig_trk_full_stats.json", "w"), indent=1)
print("joint %.1f%%  TrkA-only %.1f%%  lineage rounds %s  TrkA %s  TrkC %s"
      % (stats["arms"]["joint"]["pct_ge_threshold"],
         stats["arms"]["TrkA-only"]["pct_ge_threshold"],
         PICK["rounds"], PICK["trka"], PICK["trkc"]))
