# Extensions

## Structure predictor

A predictor implements `predict(binder_sequence, **kwargs) -> (confidence, metrics)`.
Register a factory to select it by name:

```python
from binderopt.oracles import register_oracle, get_oracle
from binderopt.oracles.metrics import normalise

class MyPredictor:
    def predict(self, binder_sequence, **kwargs):
        result = run_predictor(binder_sequence)
        metrics = normalise(
            {"i_ptm": result.iptm, "ptm_binder": result.binder_ptm,
             "min_pae_interaction": result.interface_pae},
            oracle="my_predictor", oracle_version="1.0",
            structure_file=result.structure_path,
        )
        return result.confidence, metrics

@register_oracle("my_predictor")
def build_predictor(**kwargs):
    return MyPredictor(**kwargs)

oracle = get_oracle("my_predictor")
```

Optional `residue_metrics` contain one value per binder residue: `plddt` on a 0–1 scale,
and `pae_inter`/`pae_intra` in Å. Omit unavailable residue arrays.

## Candidate generator

A generator implements `generate(n) -> list[str]` and optionally `describe()`:

```python
from binderopt.generators import register_generator, get_generator

@register_generator("my_generator")
def build_generator(**kwargs):
    return MyGenerator(**kwargs)

pool = get_generator("my_generator").generate(100)
```

Return up to `n` unique amino-acid sequences. CSV and FASTA pools use the built-in `file`
generator; the optimisation loop accepts candidates independently of their generating tool.
