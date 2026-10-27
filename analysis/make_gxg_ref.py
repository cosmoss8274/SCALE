# Side-chain reference areas for SAP: SASA of residue X in an extended Gly-X-Gly
# tripeptide, built with PyMOL's fab and measured with FreeSASA under the same
# parameters used for the designs. This is the reference state in the original SAP
# definition (Chennamsetty et al., PNAS 2009) rather than a tabulated substitute.
from pymol import cmd
import freesasa, json, os
AA3 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
one = dict(zip(AA3, "ARNDCQEGHILKMFPSTWYV"))
BB = {"N", "CA", "C", "O", "OXT"}
os.makedirs("gxg", exist_ok=True)
ref = {}
for aa in AA3:
    cmd.reinitialize()
    cmd.fab("G" + one[aa] + "G", "pep", ss=2)          # ss=2 = extended backbone
    cmd.remove("hydro")
    p = "gxg/%s.pdb" % aa
    cmd.save(p, "pep")
    s = freesasa.Structure(p)
    r = freesasa.calc(s)
    tot = 0.0
    for i in range(s.nAtoms()):
        if int(s.residueNumber(i)) == 2 and s.atomName(i).strip() not in BB:
            tot += r.atomArea(i)
    ref[aa] = round(tot, 2)
    assert aa == "GLY" or tot > 0, aa
json.dump(ref, open("gxg_sidechain_ref.json", "w"), indent=1)
print(json.dumps(ref))
