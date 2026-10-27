"""The generator interface, and the one implementation that needs no GPU."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from ..pools import read_csv, read_fasta


class Generator(ABC):
    """Produces candidate binder sequences for the loop to start from.

    One method. A generator may ignore ``n`` if it has a fixed supply -- reading a file
    of 100 designs and being asked for 500 yields 100, not an error -- so callers treat
    ``n`` as an upper bound and check what they got.
    """

    #: Short name for provenance, stamped onto the pool's record.
    name: str = "generator"

    @abstractmethod
    def generate(self, n: int) -> List[str]:
        """Up to ``n`` binder sequences, duplicates removed."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class FileGenerator(Generator):
    """A pool someone else already produced: a CSV or FASTA on disk.

    This is the honest default. SCALE does not claim to generate binders, and a file is
    how every real campaign starts -- the output of RFdiffusion + ProteinMPNN, BoltzGen,
    ODesign, a template redesign, or a hand-written list.

    Args:
        path: ``.csv`` (needs ``sequence_column``) or ``.fasta``/``.fa``/``.faa``.
        sequence_column: column holding the sequence, for CSV.
        expect_length: require every binder to be this long; ``None`` allows any.
        metric_columns: CSV columns to carry alongside, e.g. a generator's own score.
    """

    name = "file"

    def __init__(self, path: str | Path, sequence_column: str = "sequence",
                 expect_length: Optional[int] = None,
                 metric_columns: Optional[List[str]] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError("no pool at %s" % self.path)
        self.sequence_column = sequence_column
        self.expect_length = expect_length
        self.metric_columns = metric_columns
        self.metrics: Dict[str, List[float]] = {}
        self._sequences: Optional[List[str]] = None

    def _load(self) -> List[str]:
        if self._sequences is None:
            if self.path.suffix.lower() in (".fasta", ".fa", ".faa"):
                self._sequences = read_fasta(self.path, expect_length=self.expect_length)
            else:
                self._sequences, self.metrics = read_csv(
                    self.path, sequence_column=self.sequence_column,
                    metric_columns=self.metric_columns,
                    expect_length=self.expect_length)
        return self._sequences

    def generate(self, n: int) -> List[str]:
        return self._load()[:n]

    def __len__(self) -> int:
        return len(self._load())

    def describe(self) -> str:
        return "%s (%s)" % (self.name, self.path.name)
