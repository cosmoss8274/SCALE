"""RFdiffusion + ProteinMPNN as a generator.

A thin adapter over :mod:`binderopt.seeding`, which already drives both programs. It is
here so that this pipeline is selected the same way as any other design source rather
than being the one the loop knows about by name.

Needs both programs installed and their environments configured -- see
``binderopt/external.py`` for the variables. Importing this module never requires them;
only constructing the generator does.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..seeding import SeedDataGenerator, SeedingConfig
from .base import Generator
from .registry import register_generator


class RFdiffusionMPNNGenerator(Generator):
    """Backbones from RFdiffusion, sequences from ProteinMPNN.

    Args:
        config: a :class:`~binderopt.seeding.SeedingConfig` naming the target PDB,
            the contig and the hotspot residues.
        work_dir: where backbones and sequence files are written.
    """

    name = "rfdiffusion_mpnn"

    def __init__(self, config: SeedingConfig, work_dir: str | Path = "seed_data"):
        self.config = config
        self.work_dir = Path(work_dir)
        self._generator = SeedDataGenerator(config, str(self.work_dir))

    def generate(self, n: int) -> List[str]:
        """Run the pipeline and return up to ``n`` distinct binder sequences.

        The number produced is governed by the config's ``n_backbones`` and
        ``n_sequences_per_backbone``; ``n`` only truncates the result, so set those to
        control the GPU cost rather than relying on ``n``.
        """
        rows = self._generator.run()
        seen, out = set(), []
        for row in rows:
            seq = getattr(row, "sequence", None)
            if seq and seq not in seen:
                seen.add(seq)
                out.append(seq)
            if len(out) >= n:
                break
        return out

    def describe(self) -> str:
        return "%s (target=%s)" % (self.name, self.config.target_name)


@register_generator("rfdiffusion_mpnn")
def _build(**kwargs) -> RFdiffusionMPNNGenerator:
    return RFdiffusionMPNNGenerator(**kwargs)
