"""Name -> generator factory, so the design source is a configuration value."""

from __future__ import annotations

from typing import Callable, Dict

GENERATORS: Dict[str, Callable] = {}


def register_generator(name: str) -> Callable:
    """Decorator registering a factory under `name`."""

    def deco(factory: Callable) -> Callable:
        key = name.lower()
        if key in GENERATORS:
            raise ValueError("generator %r already registered" % key)
        GENERATORS[key] = factory
        return factory

    return deco


def get_generator(name: str, **kwargs):
    """Build the generator registered under `name`.

    Raises KeyError listing the available names rather than falling back to a default:
    the starting pool determines what the loop explores, so substituting a different
    source silently would change the result without changing the configuration.
    """
    key = name.lower()
    if key not in GENERATORS:
        raise KeyError(
            "unknown generator %r; available: %s"
            % (name, ", ".join(sorted(GENERATORS)))
        )
    return GENERATORS[key](**kwargs)


def _register_builtins() -> None:
    """Import the backends so their decorators run. A backend whose runtime is absent
    simply does not appear in GENERATORS, exactly as for oracles."""
    from .base import FileGenerator

    @register_generator("file")
    def _build_file(**kwargs) -> FileGenerator:
        return FileGenerator(**kwargs)

    try:
        from . import rfdiffusion  # noqa: F401
    except ImportError:
        pass


_register_builtins()
