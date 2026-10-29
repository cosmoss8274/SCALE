<p align="center"><img src="assets/scale-logo.svg" width="420" alt="SCALE" /></p>
<p align="center">Structure-conditioned active learning for protein binder design</p>
<p align="center"><a href="https://colab.research.google.com/github/cosmoss8274/SCALE/blob/main/notebooks/SCALE_demo_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" /></a></p>

SCALE refines de novo binder designs by reusing the structural feedback that generate-and-filter pipelines discard. Across five targets, computational hit rates rose from under 11% to 60–100%. No model fine-tuning, no gradients through the structure predictor, and any generator or objective can be plugged in.

## Quick start

Open the Colab notebook, select a T4 GPU and run the cells in order. The example uses
TrkA with Boltz-2. Replace the receptor sequence and candidate pool to use your own target.
The default budget is 10 initial evaluations followed by five rounds of five new designs.

For a local installation:

```bash
git clone https://github.com/cosmoss8274/SCALE.git
cd SCALE
python -m pip install -e ".[notebook]"
```

Run `notebooks/SCALE_demo.ipynb` in an environment with a CUDA-compatible predictor.
The Colab notebook installs Boltz-2 in a separate Python 3.12 environment.

## Inputs

- **Receptor:** one amino-acid sequence; optionally supply a receptor MSA in A3M format.
- **Candidates:** a CSV with a `sequence` column, or a FASTA file. Use standard amino-acid
  letters and equal-length sequences for substitution-based search.
- **Predictor:** Boltz-2, an installed AlphaFold 3 interface, or a registered custom oracle.
- **Objective:** a scalar metric or weighted combination to maximise.

Minimal candidate CSV:

```csv
sequence
ACDEFGHIKLMNPQRSTVWY
YWVTSRQPNMLKIHGFEDCA
```

An optional starting-batch CSV selects which pool entries are evaluated first.
Boltz-2 can obtain the receptor MSA from the public MMseqs2 service; this sends the
receptor sequence to that service.

## Search and outputs

The notebook exposes the prediction budget, mutation settings and minimum sequence
distances to the starting pool, within a batch and to earlier new designs. Outputs include
predicted complexes, per-design metrics, parent sequences, sequence distances, a compact
trajectory plot and a ZIP archive. Predictions are checkpointed individually for resumption.

The default objective is interface pTM. Confidence metrics, geometric descriptors and score
directions are defined in [Metrics](docs/metrics.md).

## Predictors and generators

```python
from binderopt.generators import get_generator
from binderopt.oracles import get_oracle
from binderopt.objectives import maximise, minimise, weighted

pool = get_generator("file", path="candidates.csv").generate(100)
oracle = get_oracle("boltz2", target_sequence=receptor_sequence)
objective = maximise("i_ptm")
```

The `rfdiffusion_mpnn` generator runs an installed RFdiffusion/ProteinMPNN pipeline.
The `af3` oracle wraps an installed AlphaFold 3 interface. The `replay` oracle reads
existing prediction records. See [Extensions](docs/extending.md) for custom implementations
and [Configuration](docs/configuration.md) for external-tool paths and cluster runs.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not slow"
python notebooks/build_notebook.py
```

## References and licence

SCALE code is MIT-licensed. Predictor weights and example AF3 structures retain their
respective licences; see [example structure terms](data/structures/TERMS_OF_USE.md).

- [ESM-2](https://doi.org/10.1126/science.ade2574)
- [Boltz-2](https://github.com/jwohlwend/boltz)
- [AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
- [ColabDesign](https://github.com/sokrypton/ColabDesign)
