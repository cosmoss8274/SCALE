"""Rosetta InterfaceAnalyzer on the 300 AF3 complexes (30 round-0 + 30 round-15
designs for each of the five targets).

PyRosetta 4, InterfaceAnalyzerMover, interface A_B (AB_C for the three-chain MYBL2
system). Hydrogens are added by Rosetta on pose construction; coordinates are the
raw AF3 output with NO relaxation, which is deliberate - the question is whether the
structure AF3 predicted has the physical hallmarks of an interface, not whether it
can be minimised into one.

Reported, all from InterfaceAnalyzer:
  dSASA            total buried area, both sides (d.dSASA[1] + d.dSASA[2])
  dhSASA           its hydrophobic share
  interface_hbonds Rosetta hydrogen bonds across the interface (energy-based)
  delta_unsat_hbonds  buried polar atoms left without a partner
  sc_value         Lawrence-Colman shape complementarity
  n_interface_res  interface residue count

NO ENERGY TERM IS REPORTED. On unrelaxed predicted coordinates dG_separated is
dominated by clashes the separated-state repack relieves - on a round-15 BHRF1
complex it reads -57.1 REU without a separated repack and +12.8 REU with one, which
says nothing about binding. Reporting it would require a constrained FastRelax of
every complex first.
"""
import glob, json, os, time
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
               "-detect_disulf false", silent=True)
IFACE = {"MYBL2": "AB_C"}
man = json.load(open("bio_pack/manifest.json"))
rows, t0 = [], time.time()
for k, m in enumerate(man):
    p = os.path.join("bio_pack", m["file"])
    pose = pose_from_pdb(p)
    ia = InterfaceAnalyzerMover()
    ia.set_interface(IFACE.get(m["target"], "A_B"))
    ia.set_pack_separated(False)
    ia.set_compute_packstat(False)
    ia.set_compute_interface_sc(True)
    ia.set_calc_dSASA(True)
    ia.apply(pose)
    d = ia.get_all_data()
    rows.append(dict(target=m["target"], group=m["group"], file=m["file"],
                     i_ptm=m["i_ptm"], min_pae_interaction=m["min_pae_interaction"],
                     dsasa=round(d.dSASA[1] + d.dSASA[2], 1),
                     dhsasa=round(d.dhSASA[1] + d.dhSASA[2], 1),
                     dhsasa_frac=round((d.dhSASA[1] + d.dhSASA[2])
                                       / (d.dSASA[1] + d.dSASA[2]), 4)
                     if (d.dSASA[1] + d.dSASA[2]) > 0 else None,
                     hbonds=int(d.interface_hbonds),
                     unsat=int(d.delta_unsat_hbonds),
                     sc=round(float(d.sc_value), 4),
                     n_interface_res=int(ia.get_num_interface_residues())))
    if (k + 1) % 25 == 0:
        print("  %d/%d  %.0fs" % (k + 1, len(man), time.time() - t0), flush=True)
json.dump(rows, open("biophys_rosetta.json", "w"), indent=1)
print("done", len(rows), "in %.0fs" % (time.time() - t0))
