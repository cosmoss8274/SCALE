"""Interface biophysics of AF3-predicted designs, round 0 against round 15.

Five quantities per complex, all from the AF3 coordinates:

dSASA            SASA(target alone) + SASA(binder alone) - SASA(complex), the total
                 area buried on BOTH sides of the interface. This is the definition
                 Rosetta's InterfaceAnalyzer reports; it is computed here with
                 FreeSASA (Mitternacht 2016), Lee-Richards, probe 1.4 A, 100 slices,
                 default ProtOr radii and the default polar/apolar classifier.
dSASA apolar     the apolar-atom share of the same difference.
Interface H-bonds cross-chain donor-acceptor heavy-atom pairs within 3.5 A. AF3
                 writes no hydrogens, so this is a heavy-atom distance criterion, not
                 an energy: donors = all N, plus Ser OG / Thr OG1 / Tyr OH;
                 acceptors = all O, plus His ND1/NE2. Each atom pair counted once.
Binder net charge (Lys+Arg) - (Asp+Glu) from the binder sequence; His not counted.
SAP              spatial aggregation propensity, Chennamsetty et al. PNAS 2009:
                 for each binder atom, the sum over side-chain atoms within 5 A of
                 (side-chain SASA / Gly-X-Gly side-chain SASA) x Black-Mould
                 hydrophobicity centred on glycine. Computed on the UNBOUND binder,
                 which is the state aggregation concerns; the per-structure value
                 reported is the maximum over atoms, the standard aggregation-hotspot
                 readout. Gly-X-Gly references were built with PyMOL and measured
                 with the same FreeSASA settings (gxg_sidechain_ref.json).
"""
import freesasa, json, os, tempfile, collections
import numpy as np

freesasa.setVerbosity(freesasa.silent)
PAR = freesasa.Parameters({'algorithm': freesasa.LeeRichards, 'probe-radius': 1.4,
                           'n-slices': 100})
GXG = json.load(open("gxg_sidechain_ref.json"))
# Black & Mould (1991) normalised hydrophobicity, centred on glycine (0.501).
BM = {"ALA":0.616,"ARG":0.000,"ASN":0.236,"ASP":0.028,"CYS":0.680,"GLN":0.251,
      "GLU":0.043,"GLY":0.501,"HIS":0.165,"ILE":0.943,"LEU":0.943,"LYS":0.283,
      "MET":0.738,"PHE":1.000,"PRO":0.711,"SER":0.359,"THR":0.450,"TRP":0.878,
      "TYR":0.880,"VAL":0.825}
BM = {k: v - BM["GLY"] for k, v in BM.items()}
BB = {"N", "CA", "C", "O", "OXT"}
CLS = freesasa.Classifier()
BINDER = {"MYBL2": "C"}


def read_pdb(path):
    at = []
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        at.append(dict(name=ln[12:16].strip(), resn=ln[17:20].strip(),
                       ch=ln[21], resi=int(ln[22:26]),
                       xyz=(float(ln[30:38]), float(ln[38:46]), float(ln[46:54])),
                       b=float(ln[60:66]), el=ln[76:78].strip(), line=ln.rstrip("\n")))
    return at


def sasa(lines):
    fd, p = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    open(p, "w").write("\n".join(lines) + "\nEND\n")
    s = freesasa.Structure(p)
    r = freesasa.calc(s, PAR)
    out = []
    for i in range(s.nAtoms()):
        try:
            cl = CLS.classify(s.residueName(i).strip(), s.atomName(i))
        except Exception:
            cl = "Polar"
        out.append((r.atomArea(i), cl, s.residueName(i).strip(),
                    s.atomName(i).strip(), int(s.residueNumber(i))))
    os.unlink(p)
    assert len(out) == len(lines), (len(out), len(lines))
    return out


def hbonds(at, bch):
    don, acc = [], []
    for i, a in enumerate(at):
        n, rn = a["name"], a["resn"]
        if a["el"] == "N" or (rn, n) in (("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")):
            don.append(i)
        if a["el"] == "O" or (rn == "HIS" and n in ("ND1", "NE2")):
            acc.append(i)
    D, A = set(don), set(acc)
    xyz = np.array([a["xyz"] for a in at])
    isb = np.array([a["ch"] == bch for a in at])
    cand = sorted(D | A)
    cx = xyz[cand]; cb = isb[cand]
    d = np.linalg.norm(cx[:, None, :] - cx[None, :, :], axis=2)
    n = 0
    ii, jj = np.where((d <= 3.5) & (cb[:, None] != cb[None, :]))
    for a_, b_ in zip(ii, jj):
        if a_ >= b_:
            continue
        i, j = cand[a_], cand[b_]
        if (i in D and j in A) or (j in D and i in A):
            n += 1
    return n


def sap(at, bch):
    idx = [i for i, a in enumerate(at) if a["ch"] == bch]
    lines = [at[i]["line"] for i in idx]
    ar = sasa(lines)
    sc_sasa = collections.defaultdict(float)
    for (a, cl, rn, an, ri), i in zip(ar, idx):
        if an not in BB:
            sc_sasa[(at[i]["resi"], rn)] += a
    ratio = {}
    for (ri, rn), v in sc_sasa.items():
        ref = GXG.get(rn, 0.0)
        ratio[ri] = (v / ref) if ref > 0 else 0.0
    sc = [(i, at[i]) for i in idx if at[i]["name"] not in BB]
    if not sc:
        return 0.0
    xyz = np.array([a["xyz"] for _, a in sc])
    w = np.array([ratio.get(a["resi"], 0.0) * BM.get(a["resn"], 0.0) for _, a in sc])
    allx = np.array([at[i]["xyz"] for i in idx])
    d = np.linalg.norm(allx[:, None, :] - xyz[None, :, :], axis=2)
    return float(((d <= 5.0) * w[None, :]).sum(axis=1).max())


CHG = {"LYS": 1, "ARG": 1, "ASP": -1, "GLU": -1}
man = json.load(open("bio_pack/manifest.json"))
rows = []
for k, m in enumerate(man):
    path = os.path.join("bio_pack", m["file"])
    at = read_pdb(path)
    bch = BINDER.get(m["target"], "B")
    assert any(a["ch"] == bch for a in at), m["file"]
    cl_lines = [a["line"] for a in at]
    bl = [a["line"] for a in at if a["ch"] == bch]
    tl = [a["line"] for a in at if a["ch"] != bch]
    a_cx, a_b, a_t = sasa(cl_lines), sasa(bl), sasa(tl)
    tot = sum(x[0] for x in a_b) + sum(x[0] for x in a_t) - sum(x[0] for x in a_cx)
    apo = (sum(x[0] for x in a_b if x[1] == "Apolar")
           + sum(x[0] for x in a_t if x[1] == "Apolar")
           - sum(x[0] for x in a_cx if x[1] == "Apolar"))
    res = {r: n for r, n in collections.Counter(
        (a["resi"], a["resn"]) for a in at if a["ch"] == bch).items()}
    chg = sum(CHG.get(rn, 0) for (_ri, rn) in res)
    rows.append(dict(target=m["target"], group=m["group"], file=m["file"],
                     i_ptm=m["i_ptm"], min_pae_interaction=m["min_pae_interaction"],
                     dsasa=round(tot, 1), dsasa_apolar=round(apo, 1),
                     dsasa_apolar_frac=round(apo / tot, 4) if tot > 0 else None,
                     hbonds=hbonds(at, bch), net_charge=chg,
                     sap_max=round(sap(at, bch), 3),
                     binder_residues=len(res),
                     binder_plddt=round(float(np.mean([a["b"] for a in at
                                                       if a["ch"] == bch])), 2)))
    if (k + 1) % 50 == 0:
        print("  %d/%d" % (k + 1, len(man)), flush=True)
json.dump(rows, open("biophys_designs.json", "w"), indent=1)
print("structures:", len(rows))
for t in ("BHRF1", "SC2RBD", "TRKA", "MYBL2", "HER2"):
    for g in ("r0", "r15"):
        v = [r for r in rows if r["target"] == t and r["group"] == g]
        print("%-7s %-4s n=%2d dSASA %7.0f  apolar %.2f  HB %4.1f  q %+5.1f  SAP %5.2f"
              % (t, g, len(v), np.median([x["dsasa"] for x in v]),
                 np.median([x["dsasa_apolar_frac"] for x in v]),
                 np.median([x["hbonds"] for x in v]),
                 np.median([x["net_charge"] for x in v]),
                 np.median([x["sap_max"] for x in v])))
