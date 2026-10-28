"""Environment-based locations for external prediction and design tools."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_DESCRIPTIONS = {
    "SCALE_AF3_RUN": "AlphaFold 3's run_alphafold.py",
    "SCALE_AF3_DB_DIR": "the AlphaFold 3 sequence databases",
    "SCALE_AF3_MODEL_DIR": "the AlphaFold 3 model parameters",
    "SCALE_RFDIFFUSION_DIR": "the RFdiffusion checkout",
    "SCALE_RFDIFFUSION_PYTHON": "the Python interpreter of RFdiffusion's environment",
    "SCALE_PROTEINMPNN_DIR": "the ProteinMPNN checkout",
    "SCALE_PROTEINMPNN_PYTHON": "the Python interpreter of ProteinMPNN's environment",
}


class ExternalToolNotConfigured(RuntimeError):
    """An external tool was needed but its environment variable is unset or wrong."""


def tool_path(var: str, *, must_exist: bool = True) -> str:
    """Resolve one external tool's path from the environment.

    Raises :class:`ExternalToolNotConfigured` naming the variable, rather than returning a
    path that happens to be valid on one machine.
    """
    what = _DESCRIPTIONS.get(var, var)
    value = os.environ.get(var)
    if not value:
        raise ExternalToolNotConfigured(
            "%s is not set; point it at %s, e.g. export %s=/path/to/it"
            % (var, what, var)
        )
    if must_exist and not Path(value).exists():
        raise ExternalToolNotConfigured(
            "%s is set to %r but that path does not exist (should be %s)"
            % (var, value, what)
        )
    return value


def optional_tool_path(var: str) -> Optional[str]:
    """The configured path, or ``None`` when unset -- for callers that degrade gracefully."""
    value = os.environ.get(var)
    return value or None


def is_configured(var: str) -> bool:
    """Whether ``var`` is set and points at something that exists."""
    value = os.environ.get(var)
    return bool(value) and Path(value).exists()


# -- ${VAR} expansion in config files -------------------------------------------------

#: Roots the target configs are written against. A config says
#: ``${SCALE_ASSETS}/benchmark_TRKA_rfdiffusion.csv``; each site sets the variable once.
CONFIG_ROOTS = {
    "SCALE_ASSETS": "the run assets: benchmark CSVs, AF3 output trees, MSA caches, "
    "reference backbones",
    "SCALE_RESULTS": "the directory holding completed run outputs",
    "SCALE_AF3_DB_DIR": _DESCRIPTIONS["SCALE_AF3_DB_DIR"],
    "SCALE_AF3_MODEL_DIR": _DESCRIPTIONS["SCALE_AF3_MODEL_DIR"],
}

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class UnresolvedPlaceholder(RuntimeError):
    """A config referenced ``${VAR}`` and ``VAR`` is not set."""


def expand_placeholders(value, *, where: str = "config"):
    """Recursively expand ``${VAR}`` in strings inside dicts, lists and scalars."""
    if isinstance(value, dict):
        return {k: expand_placeholders(v, where=where) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_placeholders(v, where=where) for v in value]
    if not isinstance(value, str):
        return value

    missing = sorted(
        {
            m.group(1)
            for m in _PLACEHOLDER.finditer(value)
            if not os.environ.get(m.group(1))
        }
    )
    if missing:
        lines = [
            "%s needs %s, which %s not set:"
            % (
                where,
                ", ".join("${%s}" % v for v in missing),
                "is" if len(missing) == 1 else "are",
            )
        ]
        lines += [
            "    export %s=...   # %s"
            % (v, CONFIG_ROOTS.get(v, "see binderopt/external.py"))
            for v in missing
        ]
        raise UnresolvedPlaceholder("\n".join(lines))
    return _PLACEHOLDER.sub(lambda m: os.environ[m.group(1)], value)
