# Predicted structures

Nine AlphaFold 3 predictions of TrkA/TrkB/TrkC complexes, three rounds each. They are here
so that the notebook's structure viewer and the figure scripts have something to show before
you have run an oracle yourself. Nothing in the loop depends on them.

## Terms

These files are **AlphaFold 3 Output**. They are not covered by the repository's MIT licence.
They are provided under, and subject to, the AlphaFold 3 Output Terms of Use, a complete copy
of which is in [`TERMS_OF_USE.md`](TERMS_OF_USE.md) next to this file. Each `.cif` also carries
the notice in its header; please keep it there if you redistribute them.

In short, and without replacing the terms themselves: **non-commercial use only**, and they may
not be used to train a competing structure predictor.

If you use these structures, or findings derived from them, cite:

> Abramson, J., Adler, J., Dunger, J. et al. Accurate structure prediction of biomolecular
> interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024).
> https://doi.org/10.1038/s41586-024-07487-4

## AlphaFold 3 is not required to run SCALE

SCALE's default structure oracle is [Boltz-2](https://github.com/jwohlwend/boltz), which is
open source and installs from PyPI. The AlphaFold 3 adapter in `binderopt/oracles/af3.py` is
optional and needs model parameters you must request from Google yourself; the notebook never
touches it. These nine files are historical output from the run behind the manuscript, not an
input the code reproduces.
