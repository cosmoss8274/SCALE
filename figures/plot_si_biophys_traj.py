"""SI figure: interface biophysics against optimisation round.

Does the physical quality of the interface improve along the trajectory, or only
the AlphaFold 3 confidence the loop maximises? Every quantity here is computed from
the predicted coordinates alone; none is an input to the objective.

5 targets (columns) x 6 quantities (rows), all 16 rounds, 12 evaluated designs per
round per target - 960 complexes. Twelve rather than all 30 per round because
InterfaceAnalyzer with shape complementarity costs about 4 s per complex here;
12 x 16 = 192 points per target still support the per-round median and the trend.
The subset is the first 12 designs of each target-round group in the order the
records were written, not a score ranking. Points are individual designs coloured
by round, the dark line is the per-round median, and rho is the Spearman
correlation of the per-design value against round number (grey when p >= 0.05).

Rows a-d: Rosetta InterfaceAnalyzer (PyRosetta 4; interface A_B, AB_C for the
three-chain MYBL2 system; hydrogens built by Rosetta; NO relaxation of the AF3
coordinates). Rows e-f: binder net charge from the sequence ((Lys+Arg)-(Asp+Glu),
His not counted) and spatial aggregation propensity (Chennamsetty et al. 2009) on
the unbound binder, maximum over atoms, with Gly-X-Gly side-chain references built
in PyMOL and measured with FreeSASA under the settings used throughout.

No Rosetta energy is reported: on unrelaxed predicted coordinates dG_separated is
dominated by clashes that a separated-state repack relieves.
"""
import collections
import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

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
ROWS = [("dsasa", "Buried area, dSASA (\u00c5$^2$)"),
        ("dhsasa_frac", "Hydrophobic fraction\nof buried area"),
        ("hbonds", "Interface H-bonds"),
        ("sc", "Shape complementarity"),
        ("net_charge", "Binder net charge"),
        ("sap_max", "SAP (max)")]
CMAP = plt.get_cmap("viridis")

# The stratified subset: first 12 designs of each target-round group in manifest
# order. The Rosetta shard also holds extra BHRF1 complexes from an earlier
# full-set attempt; they are dropped so every panel has the same n.
man = json.load(open("bio_traj/manifest.json"))
seen = collections.Counter()
subset = []
for m in man:
    k = (m["target"], m["round"])
    if seen[k] < 12:
        seen[k] += 1
        subset.append(m["file"])
subset = set(subset)
assert len(subset) == 960, len(subset)

ros = {r["file"]: r for r in json.load(open("biophys_traj_rosetta.json"))
       if r["status"] == "ok"}
fre = {r["file"]: r for r in json.load(open("biophys_traj_designs.json"))}
rows = []
for f in subset:
    assert f in ros and f in fre, f
    r = dict(ros[f])
    r["net_charge"] = fre[f]["net_charge"]
    r["sap_max"] = fre[f]["sap_max"]
    rows.append(r)
assert len(rows) == 960
RD = sorted({r["round"] for r in rows})
assert RD == list(range(16))

MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 196 * MM))
CX = [0.082 + i * 0.156 for i in range(5)]
CW = 0.114
RY = [0.845, 0.695, 0.545, 0.395, 0.245, 0.095]
RH = 0.121
rng = np.random.default_rng(0)
stats = {"_definitions": {
    "n": "960 complexes: 5 targets x 16 rounds x 12 designs",
    "subset": "first 12 designs of each target-round group in manifest order",
    "rows_a_d": "Rosetta InterfaceAnalyzer on unrelaxed AF3 coordinates, sc on",
    "rows_e_f": "binder net charge from sequence; SAP on the unbound binder",
    "no_energy": "dG_separated not reported on unrelaxed coordinates",
    "rho": "Spearman of the per-design value against round"}, "targets": {}}

for mi, (col, lab) in enumerate(ROWS):
    for ti, t in enumerate(TARGETS):
        ax = fig.add_axes([CX[ti], RY[mi], CW, RH])
        sub = [r for r in rows if r["target"] == t]
        assert len(sub) == 192, (t, len(sub))
        x = np.array([r["round"] for r in sub], float)
        y = np.array([r[col] for r in sub], float)
        ax.scatter(x + rng.uniform(-0.28, 0.28, x.size), y, s=2.2, c=x, cmap=CMAP,
                   vmin=0, vmax=15, linewidths=0, alpha=0.7, rasterized=True, zorder=2)
        med = [float(np.median(y[x == r])) for r in RD]
        ax.plot(RD, med, color="0.12", lw=0.8, zorder=4)
        rho, p = spearmanr(x, y)
        # Headroom so the rho label never sits on the top tick label.
        lo, hi = float(np.min(y)), float(np.max(y))
        span = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.05 * span, hi + 0.20 * span)
        ax.text(0.03, 0.98, r"$\rho$ = %+.2f" % rho, transform=ax.transAxes,
                ha="left", va="top", fontsize=TICK - 0.4,
                color=("0.10" if p < 0.05 else "0.58"))
        ax.set_xlim(-0.8, 15.8); ax.set_xticks([0, 5, 10, 15]); ax.tick_params(pad=1.2)
        ax.set_xticklabels(["0", "5", "10", "15"] if mi == len(ROWS) - 1 else [])
        if mi == len(ROWS) - 1:
            ax.set_xlabel("Optimisation round", labelpad=2, fontsize=TICK + 0.8)
        if ti == 0:
            ax.set_ylabel(lab, labelpad=2, fontsize=TICK + 0.2)
        if mi == 0:
            ax.set_title(TLAB.get(t, t), pad=3, fontsize=TICK + 1.2)
        stats["targets"].setdefault(t, {})[col] = {
            "spearman_rho": round(float(rho), 4), "p": float(p), "n": int(x.size),
            "median_by_round": [round(v, 4) for v in med]}
    fig.text(0.008, RY[mi] + RH + 0.010, "abcdef"[mi], fontsize=9.5,
             fontweight="bold", va="top", ha="left")

cax = fig.add_axes([0.882, RY[-1], 0.015, RY[0] + RH - RY[-1]])
cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=mpl.colors.Normalize(0, 15))
cb.set_label("Optimisation round", fontsize=TICK + 0.4, labelpad=3)
cb.set_ticks([0, 5, 10, 15]); cb.ax.tick_params(labelsize=TICK, width=0.5, length=1.6, pad=1.5)
cb.outline.set_linewidth(0.5)
fig.legend(handles=[Line2D([], [], color="0.12", lw=0.8, label="per-round median")],
           loc="lower right", bbox_to_anchor=(0.998, 0.006), frameon=False, fontsize=TICK)
fig.savefig("si_biophys_traj.png", dpi=600)
json.dump(stats, open("si_biophys_traj_stats.json", "w"), indent=1)
for t in TARGETS:
    s = stats["targets"][t]
    print("%-7s sc %+.2f (p=%.1g)  dSASA %+.2f  HB %+.2f  apolar %+.2f  charge %+.2f  SAP %+.2f"
          % (t, s["sc"]["spearman_rho"], s["sc"]["p"], s["dsasa"]["spearman_rho"],
             s["hbonds"]["spearman_rho"], s["dhsasa_frac"]["spearman_rho"],
             s["net_charge"]["spearman_rho"], s["sap_max"]["spearman_rho"]))
