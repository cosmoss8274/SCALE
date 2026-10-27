"""Pilot: is an interface energy usable on these AF3 complexes after a separated repack?

The published panels report no energy term because on unrelaxed predicted coordinates
dG_separated is dominated by clashes that the separated-state repack relieves. This run
repeats InterfaceAnalyzer on a subset with set_pack_separated(True), which repacks the
unbound state before scoring, and asks two things: are the values physically sane
(negative, of a magnitude comparable to the crystal references), and do they separate
round 0 from round 15 the way the shape-complementarity term does.

Subset: the first 6 round-0 and 6 round-15 complexes per target from the same 300-structure
pack used for the published panels, so the comparison is on identical inputs.
"""
import json, os, sys, time
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false")
man = json.load(open("bio_pack/manifest.json"))
N = 6
sel, seen = [], {}
for m in man:
    k = (m["target"], m["group"])
    if seen.get(k, 0) < N:
        seen[k] = seen.get(k, 0) + 1
        sel.append(m)
print("selected %d of %d" % (len(sel), len(man)), flush=True)

out, t0 = [], time.time()
for i, m in enumerate(sel):
    pose = pose_from_pdb(os.path.join("bio_pack", m["file"]))
    ia = InterfaceAnalyzerMover()
    ia.set_interface("AB_C" if m["target"] == "MYBL2" else "A_B")
    ia.set_pack_separated(True)      # repack the unbound state before scoring
    ia.set_pack_input(False)         # bound state scored as predicted
    ia.set_compute_packstat(False)
    ia.apply(pose)
    d = ia.get_all_data()
    out.append({"target": m["target"], "group": m["group"], "file": m["file"],
                "i_ptm": m["i_ptm"],
                "dG_separated": float(ia.get_separated_interface_energy()),
                "dG_per_1000dSASA": float(ia.get_interface_dG() / max(d.dSASA[1], 1.0) * 1000.0),
                "dSASA": float(d.dSASA[1]),
                "delta_unsat": float(d.delta_unsat_hbonds)})
    if (i + 1) % 10 == 0:
        print("%d/%d  %.1f s/structure" % (i + 1, len(sel), (time.time() - t0) / (i + 1)), flush=True)
json.dump(out, open("biophys_ddg_pilot.json", "w"), indent=1)
print("done %d in %.0f s" % (len(out), time.time() - t0), flush=True)
