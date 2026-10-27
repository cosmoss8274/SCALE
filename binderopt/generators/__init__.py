"""Where the first designs come from.

SCALE is an optimiser placed *after* a generator: something else proposes the starting
pool, and SCALE decides which of them to measure and what to try next. That boundary is
deliberately thin -- the loop needs a list of binder sequences and nothing else -- so any
generator can supply it, including one that does not exist yet.

A generator is any object with ``generate(n) -> list[str]``. They are selected by name,
the same way oracles are::

    from binderopt.generators import get_generator

    gen = get_generator("file", path="data/example_pool_trka.csv")
    gen = get_generator("rfdiffusion_mpnn", config=seeding_config)
    pool = gen.generate(100)

Registering a new one is a decorator, in your own code -- no edit to this package::

    from binderopt.generators import register_generator, Generator

    @register_generator("boltzgen")
    def _build(**kwargs):
        return MyBoltzGenAdapter(**kwargs)

The cheapest useful generator is ``file``: a CSV or FASTA someone else produced. That is
the one the notebook uses, because reading a pool from disk keeps the demonstration about
the optimisation loop rather than about installing a design pipeline.
"""
from .base import Generator, FileGenerator
from .registry import GENERATORS, get_generator, register_generator

__all__ = ["Generator", "FileGenerator", "get_generator", "register_generator", "GENERATORS"]
