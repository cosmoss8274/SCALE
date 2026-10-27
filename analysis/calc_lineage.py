"""Reconstruct a design lineage through the joint Trk run and locate its mutations.

The optimiser's output records carry no parent pointer, so the chain is rebuilt
greedily: starting from the round-0 seed of the chosen round-15 design, at each
round take the sequence closest in Hamming distance to the one carried forward.
Binder contact positions are read from the AF3 complexes of the chain's ends:
against TrkA from the round-15 complex, against TrkC from the round-0 complex
(the ancestor that still bound TrkC). A binder residue counts as contacting when
any of its heavy atoms falls within 4.5 A of any receptor heavy atom.
"""
import json
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

CUT = 4.5
recs = {}
for arm, path in (("joint", "trk_multi_joint.json"), ("TrkA-only", "trk_multi_ctrl.json")):
    recs[arm] = [r for r in json.load(open(path))["records"] if r["status"] == "complete"]
pick = json.load(open("trkstr/pick.json"))
seq0, seq4, seq15 = pick["seqs"]
ham = lambda a, b: sum(x != y for x, y in zip(a, b))

# The run records carry no parent pointer, so the lineage is a RECONSTRUCTION:
# a nearest-neighbour walk in Hamming distance, anchored at the three designs that
# were rendered (rounds 0, 4, 15) and walked backward within each segment. With only
# five designs per round and several mutually close, an unanchored walk does not
# recover these three, so the chain between the anchors is indicative, not recorded.
rounds = sorted({r["iteration"] for r in recs["joint"]})
def walk_back(end_seq, ks):
    out, cur = [], end_seq
    for k in ks:
        pool = [r for r in recs["joint"] if r["iteration"] == k]
        prv = min(pool, key=lambda r: ham(cur, r["sequence"]))
        out.append((k, prv["sequence"])); cur = prv["sequence"]
    return out
chain = ([(0, seq0)] + walk_back(seq4, [3, 2, 1])[::-1] + [(4, seq4)]
         + walk_back(seq15, list(range(14, 4, -1)))[::-1] + [(15, seq15)])
assert [k for k, _ in chain] == list(range(16)), [k for k, _ in chain]
chain_seqs = [q for _, q in chain]
assert chain_seqs[0] == seq0 and chain_seqs[4] == seq4 and chain_seqs[15] == seq15


# Divergence of every design from its nearest round-0 seed, per arm.
div = {}
for arm in recs:
    seeds = [r["sequence"] for r in recs[arm] if r["iteration"] == 0]
    div[arm] = [{"iteration": r["iteration"],
                 "d": min(ham(r["sequence"], s) for s in seeds)}
                for r in recs[arm] if r["iteration"] >= 1]

# Per-position mutation frequency across all optimised designs of the joint arm.
L = len(seq0)
seeds_j = [r["sequence"] for r in recs["joint"] if r["iteration"] == 0]
freq = np.zeros(L)
opt_j = [r["sequence"] for r in recs["joint"] if r["iteration"] >= 1]
for s in opt_j:
    ref = min(seeds_j, key=lambda q: ham(s, q))
    for i, (a, b) in enumerate(zip(ref, s)):
        if a != b:
            freq[i] += 1
freq = freq / len(opt_j)

par = MMCIFParser(QUIET=True)
def contacts(cif):
    st = par.get_structure("x", cif)[0]
    rec = [a for a in st["A"].get_atoms() if a.element != "H"]
    rc = np.array([a.coord for a in rec])
    out = []
    for res in st["B"]:
        if not is_aa(res, standard=True):
            continue
        bc = np.array([a.coord for a in res if a.element != "H"])
        d = np.linalg.norm(bc[:, None, :] - rc[None, :, :], axis=2)
        out.append(int(d.min() < CUT))
    return np.array(out)

ct_a = contacts("trkstr/TRKA_r15_2.cif")
ct_c = contacts("trkstr/TRKC_r00_0.cif")
assert len(ct_a) == L and len(ct_c) == L, (len(ct_a), len(ct_c), L)

mut = [i for i in range(L) if chain_seqs[0][i] != chain_seqs[-1][i]]
out = {
    "contact_cutoff_A": CUT, "binder_length": L,
    "chain": [{"round": k, "sequence": s, "d_from_ancestor": ham(seq0, s)} for k, s in chain],
    "divergence": div,
    "mut_freq_joint": [round(float(v), 4) for v in freq],
    "contact_trka_r15": ct_a.tolist(), "contact_trkc_r0": ct_c.tolist(),
    "lineage_mutations": [{"pos1": i + 1, "from": chain_seqs[0][i], "to": chain_seqs[-1][i],
                           "contacts_trka": int(ct_a[i]), "contacts_trkc": int(ct_c[i])}
                          for i in mut],
}
json.dump(out, open("trk_lineage_mut.json", "w"), indent=1)
print("chain d from ancestor:", [ham(seq0, s) for s in chain_seqs])
print("contacts TrkA r15 %d  TrkC r0 %d  both %d"
      % (ct_a.sum(), ct_c.sum(), int((ct_a & ct_c).sum())))
print("lineage end-to-end mutations:", [(m["pos1"], m["from"] + "->" + m["to"],
                                         "A" * m["contacts_trka"] + "C" * m["contacts_trkc"])
                                        for m in out["lineage_mutations"]])
