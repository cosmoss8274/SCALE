"""Flag values in run records that are exactly equal to an imputed default.

`binderopt/optimizer.py` substitutes a fixed value when a metric is absent from an
evaluation result (`_METRIC_DEFAULTS`), and `losses.py` does the same for l_rmsd with a
different value. Once written to a record an imputed value is indistinguishable from a
measured one, so this script reports the rows where a metric is exactly equal to one of
those constants. Exact equality is suspicious, not proof: a real measurement can land on
7.0 A. Treat the output as a list to check, not a verdict.

    python analysis/flag_imputed.py data/*.json
"""
import json
import sys
from pathlib import Path

IMPUTED = {"l_rmsd": (10.0, 5.0), "min_pae_interaction": (7.0,)}


def scan(path: Path):
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return [("unreadable", str(exc))]
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    hits = []
    for i, rec in enumerate(records):
        metrics = rec.get("metrics") if isinstance(rec, dict) else None
        if not isinstance(metrics, dict):
            continue
        for target, values in metrics.items():
            if not isinstance(values, dict):
                continue
            for metric, defaults in IMPUTED.items():
                if metric not in values:
                    continue
                for default in defaults:
                    if abs(float(values[metric]) - default) < 1e-9:
                        hits.append((i, rec.get("iteration"), target, metric, default))
    return hits


def main(paths) -> int:
    total = 0
    for p in (Path(x) for x in paths):
        hits = scan(p)
        total += len(hits)
        print("%-34s %d suspect value(s)" % (p.name, len(hits)))
        for h in hits:
            print("    record %-4s round %-4s %-6s %s == %s (the imputed value)" % h)
    print("\n%d value(s) exactly equal to an imputed default across %d file(s)"
          % (total, len(list(paths))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["data/trk_multi_joint.json", "data/trk_multi_ctrl.json"]))
