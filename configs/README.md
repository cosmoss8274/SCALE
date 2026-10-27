# Configuration

A run is configured by deep-merging `defaults.yaml` with `targets/<name>.yaml`; target
values win, and CLI flags win over both.

## Paths are placeholders, not absolute

The target configs refer to data by `${VAR}` placeholder rather than by absolute path, so
this repository carries no single site's directory layout. `binderopt.targets._load_yaml`
expands them at load time and raises, naming the variable, if one is unset — a config that
half-resolves fails later and far less clearly.

| Variable | What it points at |
|---|---|
| `SCALE_ASSETS` | run assets: benchmark CSVs, AF3 output trees, MSA caches, reference backbones |
| `SCALE_RESULTS` | completed run outputs |
| `SCALE_AF3_DB_DIR` | AlphaFold 3 sequence databases |
| `SCALE_AF3_MODEL_DIR` | AlphaFold 3 model parameters |

```bash
export SCALE_ASSETS=/path/to/your/assets
export SCALE_RESULTS=/path/to/your/results
```

Tool locations (AlphaFold 3, RFdiffusion, ProteinMPNN and their interpreters) are separate
and documented in `binderopt/external.py`.

## The notebook needs none of this

`notebooks/` runs the loop from a sequence and a CSV pool, with Boltz-2 as the oracle. The
configs here are for full campaigns driven by `design.py` on a cluster.
