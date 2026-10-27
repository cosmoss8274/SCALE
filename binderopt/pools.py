"""Reading a starting pool produced by any generator.

SCALE does not generate the first designs; it improves designs something else produced -
RFdiffusion + ProteinMPNN, BoltzGen, ODesign, a template redesign, or a hand-written list.
The only thing the loop needs from that step is a list of binder sequences, optionally with
metrics the generator already computed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA = set("ACDEFGHIKLMNPQRSTVWY")


def _validate(sequences: List[str], expect_length: Optional[int]) -> None:
    if not sequences:
        raise ValueError("pool is empty")
    bad = {c for s in sequences for c in s} - AA
    if bad:
        raise ValueError("non-standard residues in pool: %s" % ", ".join(sorted(bad)))
    lengths = {len(s) for s in sequences}
    if expect_length is not None and lengths != {expect_length}:
        raise ValueError("expected all binders to be %d aa, found %s"
                         % (expect_length, sorted(lengths)))


def read_fasta(path: str | Path, expect_length: Optional[int] = None) -> List[str]:
    """Sequences from a FASTA file, in file order, duplicates removed."""
    seqs, current = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if current:
                seqs.append("".join(current)); current = []
        elif line.strip():
            current.append(line.strip().upper())
    if current:
        seqs.append("".join(current))
    seqs = list(dict.fromkeys(seqs))
    _validate(seqs, expect_length)
    return seqs


def read_csv(path: str | Path, sequence_column: str = "sequence",
             metric_columns: Optional[List[str]] = None,
             expect_length: Optional[int] = None) -> Tuple[List[str], Dict[str, List[float]]]:
    """Sequences and any already-computed metrics from a CSV.

    Returns ``(sequences, metrics)`` where ``metrics`` maps column name to a list aligned
    with ``sequences``. Rows with a blank sequence are skipped; duplicates keep the first.
    """
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        raise ValueError("%s has no rows" % path)
    if sequence_column not in rows[0]:
        raise KeyError("column %r not in %s; columns are %s"
                       % (sequence_column, path, ", ".join(rows[0])))
    seen, sequences, keep = set(), [], []
    for row in rows:
        seq = (row.get(sequence_column) or "").strip().upper()
        if not seq or seq in seen:
            continue
        seen.add(seq); sequences.append(seq); keep.append(row)
    _validate(sequences, expect_length)

    metrics: Dict[str, List[float]] = {}
    for col in metric_columns or []:
        if col not in rows[0]:
            raise KeyError("metric column %r not in %s" % (col, path))
        metrics[col] = [float(r[col]) if (r.get(col) or "").strip() else float("nan")
                        for r in keep]
    return sequences, metrics
