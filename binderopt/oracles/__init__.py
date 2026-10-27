"""Structure-oracle backends for the SCALE loop.

The loop never calls a predictor directly: it calls an oracle, which is any object
satisfying `binderopt.af3_interface.StructureEvaluator` — `predict(binder_sequence,
**kwargs) -> (confidence, metrics)`. Swapping AlphaFold 3 for Boltz-2, Chai-1 or a
recorded table is therefore a one-line change in a config, not a change to the loop.

    from binderopt.oracles import get_oracle
    oracle = get_oracle("af3", target=target_cfg, ...)

Every oracle stamps `oracle` and `oracle_version` onto the metrics it returns. This is
not decoration: confidence metrics are NOT comparable across predictors. The hit
criteria used in the SCALE manuscript (for example TrkA: interface PAE <= 1.5 A and
interface pTM >= 0.80) are calibrated on AlphaFold 3 output, and a Boltz-2 interface
pTM of 0.80 does not denote the same thing. Results from different oracles must not be
pooled in one figure or filtered with one threshold, and the cache key includes the
oracle name so that a backend switch cannot silently hit a stale entry.
"""
from .registry import ORACLES, get_oracle, register_oracle
from .replay import ReplayOracle

__all__ = ["get_oracle", "register_oracle", "ORACLES", "ReplayOracle"]
