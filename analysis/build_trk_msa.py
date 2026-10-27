"""Convert the AlphaFold Server bundle folds_2026_10_26_14_37.zip into the
receptor-only AF3 documents the SCALE interface expects, one per Trk receptor.

Each document is {dialect, version, name, sequences:[{protein: {...}}], modelSeeds,
bondedAtomPairs, userCCD} with the receptor as chain A carrying unpairedMsa,
pairedMsa and the four template hits with their query/template index maps, exactly
the shape experiments/multi_target_injection/prepare_msa.py writes and
experiments/new_selectivity/baseline_catalog.target_cache validates.
"""
import glob, hashlib, json, os

SPEC = {"trka": "TrkA", "trkb": "TrkB", "trkc": "TrkC"}
want = json.load(open("ortho/trk_targets_final.json"))
os.makedirs("trkmsa/out", exist_ok=True)
rep = {}
for d, nm in SPEC.items():
    seq = want[nm]["sequence"]
    jr = json.load(open("trkmsa/%s/fold_%s_job_request.json" % (d, d)))
    jr0 = jr[0] if isinstance(jr, list) else jr
    sub = jr0["sequences"][0]["proteinChain"]["sequence"]
    assert sub == seq, (nm, "submitted sequence differs from the construct")
    up = open(glob.glob("trkmsa/%s/msas/*_unpaired_msa_chains_a.a3m" % d)[0]).read()
    pa = open(glob.glob("trkmsa/%s/msas/*_paired_msa_chains_a.a3m" % d)[0]).read()
    # the first MSA record must be the query itself
    assert up.split("\n")[1].replace("-", "").upper().startswith(seq[:20]), nm
    q2h = json.load(open(glob.glob("trkmsa/%s/templates/*query_to_hit.json" % d)[0]))
    tpl = []
    for h in q2h:
        p = os.path.join("trkmsa", d, "templates", h["name"])
        assert os.path.exists(p), p
        assert len(h["queryIndices"]) == len(h["templateIndices"]), h["name"]
        assert max(h["queryIndices"]) < len(seq), (nm, h["name"], "query index out of range")
        tpl.append({"mmcif": open(p).read(), "queryIndices": h["queryIndices"],
                    "templateIndices": h["templateIndices"]})
    protein = {"id": "A", "sequence": seq, "modifications": [],
               "unpairedMsa": up, "pairedMsa": pa, "templates": tpl}
    doc = {"dialect": "alphafold3", "version": 4, "name": "%s_only" % nm.upper(),
           "sequences": [{"protein": protein}], "modelSeeds": [42],
           "bondedAtomPairs": None, "userCCD": None}
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    out = "trkmsa/out/%s_only_data.json" % nm.upper()
    open(out, "w").write(text)
    rep[nm] = {"file": out, "mb": round(len(text)/1e6, 2),
               "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
               "doc_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "length": len(seq), "unpaired_seqs": up.count(">"),
               "paired_seqs": pa.count(">"), "templates": len(tpl),
               "uniprot": want[nm]["uniprot"], "range": want[nm]["range"],
               "server_bundle": "folds_2026_10_26_14_37.zip"}
    print("%-5s %-34s %6.2f MB  unpaired %5d  paired %5d  templates %d  seq_sha %s"
          % (nm, out, rep[nm]["mb"], rep[nm]["unpaired_seqs"], rep[nm]["paired_seqs"],
             len(tpl), rep[nm]["sequence_sha256"][:12]))
json.dump(rep, open("trkmsa/out/trk_msa_manifest.json", "w"), indent=1)
