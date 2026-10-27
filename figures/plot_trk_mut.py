"""Sequence divergence along the Trk multi-objective run.

a Reconstructed lineage of the rendered design: each row is one round, each column
  one of the 80 binder positions; a filled cell is a substitution relative to the
  round-0 ancestor. The track above marks binder positions that contact TrkA in the
  round-15 complex and TrkC in the round-0 complex (any heavy atom within 4.5 A).
b Substitution frequency per position over every optimised design of the joint arm,
  each referred to its nearest round-0 seed.
c Divergence from the nearest seed for every design, both arms, against round.
d The substitutions separating the round-15 design from its round-0 ancestor.

The optimiser records no parent pointer; the lineage in a is a nearest-neighbour
reconstruction anchored at the three designs rendered in the companion figure.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
CA, CC = "#1F5FBF", "#E08A1E"
d = json.load(open("trk_lineage_mut.json"))
L = d["binder_length"]
chain = d["chain"]
anc = chain[0]["sequence"]
ct_a = np.array(d["contact_trka_r15"], bool)
ct_c = np.array(d["contact_trkc_r0"], bool)
MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 118 * MM))

axT = fig.add_axes([0.070, 0.905, 0.905, 0.045])
axA = fig.add_axes([0.070, 0.585, 0.905, 0.310])
for x in range(L):
    if ct_a[x]:
        axT.add_patch(plt.Rectangle((x, 0.52), 1, 0.44, color=CA, lw=0))
    if ct_c[x]:
        axT.add_patch(plt.Rectangle((x, 0.04), 1, 0.44, color=CC, lw=0))
axT.set_xlim(0, L); axT.set_ylim(0, 1); axT.axis("off")
axT.text(-0.008, 0.74, "TrkA", transform=axT.transAxes, ha="right", va="center",
         fontsize=TICK - 0.4, color=CA)
axT.text(-0.008, 0.26, "TrkC", transform=axT.transAxes, ha="right", va="center",
         fontsize=TICK - 0.4, color=CC)
for row, c in enumerate(chain):
    for x, (p, q) in enumerate(zip(anc, c["sequence"])):
        if p != q:
            col = CA if (ct_a[x] and not ct_c[x]) else CC if (ct_c[x] and not ct_a[x]) \
                else "#6B4FA8" if (ct_a[x] and ct_c[x]) else "0.55"
            axA.add_patch(plt.Rectangle((x, row + 0.12), 1, 0.76, color=col, lw=0))
axA.set_xlim(0, L); axA.set_ylim(16, 0)
axA.set_yticks(np.arange(16) + 0.5); axA.set_yticklabels(range(16))
axA.set_xticks(np.arange(0, L + 1, 10)); axA.set_xticklabels([])
axA.set_ylabel("Round", labelpad=2); axA.tick_params(pad=1.5)
axA.grid(axis="x", color="0.9", lw=0.4, zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(True); axA.spines[s].set_color("0.8")

axB = fig.add_axes([0.070, 0.420, 0.905, 0.140])
f = np.array(d["mut_freq_joint"])
cols = [CA if (ct_a[i] and not ct_c[i]) else CC if (ct_c[i] and not ct_a[i])
        else "#6B4FA8" if (ct_a[i] and ct_c[i]) else "0.65" for i in range(L)]
axB.bar(np.arange(L) + 0.5, f, width=0.82, color=cols, linewidth=0)
axB.set_xlim(0, L); axB.set_xticks(np.arange(0, L + 1, 10))
axB.set_xlabel("Binder position", labelpad=2)
axB.set_ylabel("Substituted\nfraction", labelpad=2); axB.tick_params(pad=1.5)

axC = fig.add_axes([0.070, 0.085, 0.330, 0.235])
for arm in ("TrkA-only", "joint"):
    v = d["divergence"][arm]
    ks = np.array([x["iteration"] for x in v], float)
    ds = np.array([x["d"] for x in v], float)
    axC.scatter(ks + np.random.RandomState(0).uniform(-0.17, 0.17, len(ks)), ds,
                s=4.5, color=ARMC[arm], linewidths=0.2, edgecolor="white", zorder=3)
    rr = sorted(set(ks))
    axC.plot(rr, [np.median(ds[ks == k]) for k in rr], color=ARMC[arm], lw=1.1, zorder=4)
axC.set_xlabel("Round", labelpad=2)
axC.set_ylabel("Substitutions from\nnearest seed", labelpad=2)
axC.set_xlim(0.4, 15.6); axC.set_xticks([1, 5, 10, 15]); axC.tick_params(pad=1.5)
axC.legend(handles=[Line2D([], [], color=ARMC[a], lw=1.1, label=a)
                    for a in ("joint", "TrkA-only")],
           loc="upper left", fontsize=TICK, handlelength=1.5, labelspacing=0.4)

axD = fig.add_axes([0.470, 0.085, 0.505, 0.235]); axD.axis("off")
ms = d["lineage_mutations"]
lab = {(1, 1): "TrkA + TrkC", (1, 0): "TrkA only", (0, 1): "TrkC only", (0, 0): "\u2013"}
clr = {(1, 1): "#6B4FA8", (1, 0): CA, (0, 1): CC, (0, 0): "0.45"}
axD.text(0.00, 1.00, "Round 0 to round 15", fontsize=TICK + 0.4, va="top", color="0.3")
axD.text(0.42, 1.00, "contacts", fontsize=TICK + 0.4, va="top", color="0.3")
for i, m in enumerate(ms):
    y = 0.86 - i * 0.115
    k = (m["contacts_trka"], m["contacts_trkc"])
    axD.text(0.00, y, "%s%d%s" % (m["from"], m["pos1"], m["to"]), fontsize=TICK + 0.6,
             va="top", family="monospace")
    axD.text(0.42, y, lab[k], fontsize=TICK + 0.4, va="top", color=clr[k])
for xy, lb in [((0.030, 0.985), "a"), ((0.030, 0.575), "b"),
               ((0.030, 0.340), "c"), ((0.432, 0.340), "d")]:
    fig.text(xy[0], xy[1], lb, fontsize=9.5, fontweight="bold", va="top", ha="left")
fig.savefig("fig_trk_mut.png", dpi=600)
print("end-to-end subs %d | contact-position subs %d | d-range joint %d-%d"
      % (len(ms), sum(1 for m in ms if m["contacts_trka"] or m["contacts_trkc"]),
         min(x["d"] for x in d["divergence"]["joint"]),
         max(x["d"] for x in d["divergence"]["joint"])))
