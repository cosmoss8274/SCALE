from pymol import cmd
import json
cmd.load("native/1WWW.pdb", "a_src"); cmd.create("TrkA", "a_src and chain X and polymer")
cmd.load("native/1WWB.cif", "b_src"); cmd.create("TrkB", "b_src and chain X and polymer")
cmd.load("native/1WWC.cif", "c_src"); cmd.create("TrkC", "c_src and chain A and polymer")
cmd.delete("a_src"); cmd.delete("b_src"); cmd.delete("c_src")
rep = {}
for o in ("TrkA", "TrkB", "TrkC"):
    ri = sorted({int(a.resi) for a in cmd.get_model("%s and name CA" % o).atom})
    rep[o] = {"modelled_first": ri[0], "modelled_last": ri[-1], "n_CA": len(ri),
              "gaps": [ (ri[i], ri[i+1]) for i in range(len(ri)-1) if ri[i+1]-ri[i] > 1 ]}
    print("%s modelled %d-%d, %d CA, gaps %s" % (o, ri[0], ri[-1], len(ri), rep[o]["gaps"]))
for o in ("TrkB", "TrkC"):
    r = cmd.align("%s////CA" % o, "TrkA////CA", cycles=5)
    rep["align_%s_to_TrkA" % o] = {"rmsd": round(r[0], 3), "n_atoms": r[1],
                                   "raw_rmsd": round(r[3], 3), "raw_atoms": r[4]}
    print("%s -> TrkA: RMSD %.3f A over %d CA (initial %.3f over %d)"
          % (o, r[0], r[1], r[3], r[4]))
json.dump(rep, open("ortho/trk_superposition.json", "w"), indent=1)
