"""SI figure: interface biophysics of the predicted complexes, round 0 vs round 15.

Does raising interface pTM produce interfaces with the physical signature of real
ones, or only predictor confidence? Every quantity here is computed from the AF3
coordinates alone, with no reference to any AF3 confidence score.

Rows a-d come from Rosetta's InterfaceAnalyzer (PyRosetta 4, interface A_B, AB_C for
the three-chain MYBL2 system, hydrogens added on pose construction, NO relaxation of
the AF3 coordinates). Rows e-f are sequence- and geometry-derived quantities that
InterfaceAnalyzer does not provide: net charge from the binder sequence, and spatial
aggregation propensity (Chennamsetty et al. PNAS 2009) computed on the unbound
binder with Gly-X-Gly side-chain references built in PyMOL and measured with the
same FreeSASA settings.

The dashed line on rows a-d is the crystallographic reference interface of the same
target, run through the identical mover: 2WH6 A/B (BHRF1 / Bim BH3), 6M0J E/A (RBD /
ACE2), 1WWW X/VW (TrkA d5 / NGF dimer), 6C48 AB/C (LIN9+LIN52 / B-Myb peptide, the
same construct topology as the designs) and 3N85 A/LH (HER2 / Fab37). 3N85's partner
is an antibody fragment, not a physiological ligand; it is a real crystallographic
interface used as a scale bar, nothing more. No reference line is drawn on rows e-f:
net charge and SAP scale with chain length and identity, and the natural partners
range from a 26-residue peptide to a 603-residue receptor, so a line there would
compare quantities that are not comparable.

NO ROSETTA ENERGY IS REPORTED. On unrelaxed predicted coordinates dG_separated is
set by clashes that a separated-state repack relieves - on one round-15 BHRF1
complex it reads -57.1 REU without that repack and +12.8 REU with it.
"""
import json
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
ROWS = [("dsasa", "Buried area, dSASA (\u00c5$^2$)", True),
        ("dhsasa_frac", "Hydrophobic fraction\nof buried area", True),
        ("hbonds", "Interface H-bonds", True),
        ("sc", "Shape complementarity", True),
        ("net_charge", "Binder net charge", False),
        ("sap_max", "SAP (max)", False)]
ramp = plt.get_cmap("Blues")
C0, C15 = ramp(0.30), ramp(0.80)

R = json.load(open("biophys_rosetta.json"))
F = json.load(open("biophys_designs.json"))
N = json.load(open("biophys_native.json"))
by = {}
for r in R:
    by.setdefault((r["target"], r["group"]), {}).update(
        {k: by.get((r["target"], r["group"]), {}).get(k, []) for k in ()})
D = {}
for r in R:
    D.setdefault((r["target"], r["group"]), []).append(r)
E = {}
for r in F:
    E.setdefault((r["target"], r["group"]), []).append(r)
for k in D:
    assert len(D[k]) == len(E[k]) == 30, (k, len(D[k]), len(E[k]))


def vals(t, g, col):
    if col in ("net_charge", "sap_max"):
        return [x[col] for x in E[(t, g)]]
    return [x[col] for x in D[(t, g)]]


MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 190 * MM))
CX = [0.078 + i * 0.158 for i in range(5)]
CW = 0.116
RY = [0.842, 0.692, 0.542, 0.392, 0.242, 0.092]
RH = 0.121
stats = {"_definitions": {
    "dsasa": "InterfaceAnalyzer dSASA[1]+dSASA[2]: total area buried on both sides, A^2",
    "dhsasa_frac": "hydrophobic share of that buried area (dhSASA/dSASA)",
    "hbonds": "InterfaceAnalyzer interface_hbonds, energy-based, hydrogens built by Rosetta",
    "sc": "Lawrence-Colman shape complementarity from InterfaceAnalyzer",
    "net_charge": "(Lys+Arg)-(Asp+Glu) of the binder sequence, His not counted",
    "sap_max": "max per-atom spatial aggregation propensity on the unbound binder",
    "groups": "r0 = 30 designs sampled from the round-0 library; r15 = all 30 round-15 designs",
    "relaxation": "none; raw AF3 coordinates",
    "no_energy": "dG_separated not reported: unrelaxed coordinates make it clash-dominated",
    "native_rows": "reference line drawn on rows a-d only"},
    "native": N, "targets": {}}

for mi, (col, lab, has_ref) in enumerate(ROWS):
    for ti, t in enumerate(TARGETS):
        ax = fig.add_axes([CX[ti], RY[mi], CW, RH])
        v0, v15 = vals(t, "r0", col), vals(t, "r15", col)
        bp = ax.boxplot([v0, v15], positions=[0, 1], widths=0.52, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for patch, c in zip(bp["boxes"], [C0, C15]):
            patch.set_facecolor(c); patch.set_edgecolor("0.25"); patch.set_linewidth(0.45)
        for key in ("whiskers", "caps", "medians"):
            for ln in bp[key]:
                ln.set_color("0.25"); ln.set_linewidth(0.45)
        lo = min(min(v0), min(v15)); hi = max(max(v0), max(v15))
        if has_ref:
            nv = N[t][col]
            ax.axhline(nv, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=1)
            lo, hi = min(lo, nv), max(hi, nv)
        pad = (hi - lo) * 0.12 or 0.5
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1]); ax.tick_params(pad=1.2)
        ax.set_xticklabels(["0", "15"] if mi == len(ROWS) - 1 else [])
        if mi == len(ROWS) - 1:
            ax.set_xlabel("Round", labelpad=2, fontsize=TICK + 0.8)
        if ti == 0:
            ax.set_ylabel(lab, labelpad=2, fontsize=TICK + 0.2)
        if mi == 0:
            ax.set_title(TLAB.get(t, t), pad=3, fontsize=TICK + 1.2)
        u, p = mannwhitneyu(v0, v15, alternative="two-sided")
        stats["targets"].setdefault(t, {})[col] = {
            "r0_median": round(float(np.median(v0)), 4),
            "r15_median": round(float(np.median(v15)), 4),
            "r0_iqr": [round(float(np.percentile(v0, 25)), 4),
                       round(float(np.percentile(v0, 75)), 4)],
            "r15_iqr": [round(float(np.percentile(v15, 25)), 4),
                        round(float(np.percentile(v15, 75)), 4)],
            "n": [len(v0), len(v15)], "mannwhitney_p": float(p),
            "native": N[t][col] if has_ref else None}
    fig.text(0.008, RY[mi] + RH + 0.010, "abcdef"[mi], fontsize=9.5,
             fontweight="bold", va="top", ha="left")

h = [Patch(facecolor=C0, edgecolor="0.25", lw=0.45, label="round 0"),
     Patch(facecolor=C15, edgecolor="0.25", lw=0.45, label="round 15"),
     Line2D([], [], color="0.35", lw=0.7, ls=(0, (3, 2)), label="crystal structure")]
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.852, 0.945), ncol=1,
           frameon=False, fontsize=TICK + 0.4, handlelength=1.3, handletextpad=0.45,
           labelspacing=0.6, alignment="left")

fig.savefig("si_biophys.png", dpi=600)
json.dump(stats, open("si_biophys_stats.json", "w"), indent=1)
for t in TARGETS:
    s = stats["targets"][t]["sc"]
    print("%-7s sc %.3f -> %.3f (crystal %.3f)  p=%.2g" %
          (t, s["r0_median"], s["r15_median"], s["native"], s["mannwhitney_p"]))
