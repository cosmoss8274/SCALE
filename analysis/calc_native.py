"""Same InterfaceAnalyzer settings applied to the crystallographic reference
interface of each target. Chain groups were read off the RCSB polymer-entity
records, not assumed."""
import json
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
               "-detect_disulf false", silent=True)
SPEC = {"BHRF1": ("2WH6", "A_B", "Bim BH3 peptide"),
        "SC2RBD": ("6M0J", "E_A", "ACE2"),
        "TRKA": ("1WWW", "X_VW", "NGF dimer"),
        "MYBL2": ("6C48", "AB_C", "B-Myb peptide"),
        "HER2": ("3N85", "A_LH", "Fab37 Fab (not a physiological partner)")}
out = {}
for t, (pdb, iface, partner) in SPEC.items():
    pose = pose_from_pdb("native/%s_iface.pdb" % t)
    ia = InterfaceAnalyzerMover()
    ia.set_interface(iface)
    ia.set_pack_separated(False); ia.set_compute_packstat(False)
    ia.set_compute_interface_sc(True); ia.set_calc_dSASA(True)
    ia.apply(pose)
    d = ia.get_all_data()
    ds = d.dSASA[1] + d.dSASA[2]; dh = d.dhSASA[1] + d.dhSASA[2]
    out[t] = dict(pdb=pdb, interface=iface, partner=partner,
                  dsasa=round(ds, 1), dhsasa=round(dh, 1),
                  dhsasa_frac=round(dh / ds, 4),
                  hbonds=int(d.interface_hbonds), unsat=int(d.delta_unsat_hbonds),
                  sc=round(float(d.sc_value), 4),
                  n_interface_res=int(ia.get_num_interface_residues()))
    print("%-7s %s %-6s dSASA %6.0f  apolar %.2f  HB %2d  sc %.3f  res %3d  | %s"
          % (t, pdb, iface, ds, dh / ds, d.interface_hbonds, d.sc_value,
             ia.get_num_interface_residues(), partner))
json.dump(out, open("biophys_native.json", "w"), indent=1)
