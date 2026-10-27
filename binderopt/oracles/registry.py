"""Name -> oracle factory, so the backend is a configuration value."""
from __future__ import annotations

from typing import Callable, Dict

ORACLES: Dict[str, Callable] = {}


def register_oracle(name: str) -> Callable:
    """Decorator registering a factory under `name`."""
    def deco(factory: Callable) -> Callable:
        key = name.lower()
        if key in ORACLES:
            raise ValueError("oracle %r already registered" % key)
        ORACLES[key] = factory
        return factory
    return deco


def get_oracle(name: str, **kwargs):
    """Build the oracle registered under `name`.

    Raises KeyError listing the available names rather than falling back to a default:
    silently substituting a different structure predictor would change what every
    downstream number means.
    """
    key = name.lower()
    if key not in ORACLES:
        raise KeyError("unknown oracle %r; available: %s" % (name, ", ".join(sorted(ORACLES))))
    return ORACLES[key](**kwargs)


def _register_builtins() -> None:
    """Import the backends so their decorators run. Optional dependencies are tolerated:
    a backend whose runtime is absent simply does not appear in ORACLES."""
    from . import replay  # noqa: F401  - pure python, always available
    try:
        from . import af3  # noqa: F401
    except Exception:
        pass
    try:
        from . import boltz  # noqa: F401
    except Exception:
        pass


_register_builtins()
