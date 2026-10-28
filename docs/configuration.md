# Configuration

Cluster campaigns use `design.py` with target configurations in `binderopt/configs/targets/`.
Target values override `binderopt/configs/defaults.yaml`; command-line options override both.

```bash
python design.py --target_name trka --iters 5 --mutants_per_round 5
```

Set external-tool paths for the backends you use:

| Environment variable | Location |
|---|---|
| `SCALE_AF3_RUN` | AlphaFold 3 `run_alphafold.py` |
| `SCALE_AF3_DB_DIR` | AlphaFold 3 sequence databases |
| `SCALE_AF3_MODEL_DIR` | AlphaFold 3 model parameters |
| `SCALE_RFDIFFUSION_DIR` | RFdiffusion checkout |
| `SCALE_RFDIFFUSION_PYTHON` | RFdiffusion environment Python |
| `SCALE_PROTEINMPNN_DIR` | ProteinMPNN checkout |
| `SCALE_PROTEINMPNN_PYTHON` | ProteinMPNN environment Python |
| `SCALE_ASSETS` | Initial CSVs, predictions, MSAs and reference structures |
| `SCALE_RESULTS` | Run output directory |

Use `${VARIABLE}` placeholders for site-specific paths in target YAML files. The notebook
accepts sequences and pools directly and does not require these cluster configurations.
