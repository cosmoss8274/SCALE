"""Target registry: discover and load configs/targets/*.yaml, merged with defaults.yaml."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from .config import RunConfig, deep_merge
from .external import expand_placeholders

logger = logging.getLogger(__name__)

_SCALE_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _SCALE_ROOT / "configs"
_DEFAULTS_PATH = _CONFIGS_DIR / "defaults.yaml"
_TARGETS_DIR = _CONFIGS_DIR / "targets"


def list_targets() -> List[str]:
    """Return sorted target registry keys discovered from configs/targets/*.yaml."""
    if not _TARGETS_DIR.exists():
        return []
    return sorted(p.stem for p in _TARGETS_DIR.glob("*.yaml"))


def _load_yaml(path: Path) -> dict:
    """Parse a config file and expand any ``${VAR}`` path placeholders.

    See :mod:`binderopt.external` for the variables; configs carry placeholders rather
    than absolute paths so the repository does not ship one machine's directory layout.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return expand_placeholders(raw, where=str(path))


def load_target(target_name: str) -> RunConfig:
    """Load a target config, deep-merged with defaults.yaml (target values win).

    Args:
        target_name: registry key, must match a file configs/targets/<target_name>.yaml.

    Returns:
        A fully-populated RunConfig (before any CLI overrides are applied).
    """
    target_path = _TARGETS_DIR / f"{target_name}.yaml"
    if not target_path.exists():
        available = ", ".join(list_targets()) or "(none found)"
        raise FileNotFoundError(
            f"No target config at {target_path}. Available targets: {available}"
        )

    defaults = _load_yaml(_DEFAULTS_PATH) if _DEFAULTS_PATH.exists() else {}
    target_overrides = _load_yaml(target_path)

    merged = deep_merge(defaults, target_overrides)
    merged.setdefault("target_name", target_name)

    config = RunConfig.from_dict(merged)

    if not config.output_dir:
        config.output_dir = str(_SCALE_ROOT / "results" / config.target_name)

    if config.extra.get("initial_csv_is_stale"):
        logger.warning(
            "=" * 70 + "\n"
            "Target '%s': initial_csv (%s) is marked STALE in its target yaml.\n"
            "Running optimization against it will seed from scientifically\n"
            "invalid data. See the 'seeding' / staleness comment in\n"
            "configs/targets/%s.yaml for why, and regenerate via\n"
            "prepare_seed_data.py before running design.py for real.\n" + "=" * 70,
            config.target_name, config.initial_csv, config.target_name,
        )

    return config


def ensure_benchmark_csv(
    rfdiffusion_csv_path: str,
    target_name: str,
    target_sequence: str,
    output_csv_path: str,
) -> str:
    """Convert a raw RFdiffusion/AF3 summary CSV into the benchmark CSV schema.

    The benchmark schema (columns: sequence, target_sequence, target, seed,
    design_json, cif_file, sample_dir, nll, design_dir, af3_output_dir, iptm,
    ptm, ranking_score, iteration) is what ``binderopt.optimizer`` expects to load.
    No-ops (returns ``output_csv_path`` unchanged) if that file already exists.

    Args:
        rfdiffusion_csv_path: raw CSV with columns binder_sequence, structure_dir, iptm, ptm, ranking_score.
        target_name: target registry name, stored in the 'target' column.
        target_sequence: target protein sequence, stored in the 'target_sequence' column.
        output_csv_path: where to write the converted benchmark-format CSV.

    Returns:
        The path to the benchmark-format CSV (newly written or pre-existing).
    """
    output_path = Path(output_csv_path)
    if output_path.exists():
        logger.info("Using existing benchmark CSV: %s", output_path)
        return str(output_path)

    logger.info("Converting RFdiffusion CSV %s -> %s", rfdiffusion_csv_path, output_path)
    with open(rfdiffusion_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = [
        "sequence", "target_sequence", "target", "seed", "design_json",
        "cif_file", "sample_dir", "nll", "design_dir", "af3_output_dir",
        "iptm", "ptm", "ranking_score", "iteration",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            structure_dir = row["structure_dir"]
            writer.writerow({
                "sequence": row["binder_sequence"],
                "target_sequence": target_sequence,
                "target": target_name,
                "seed": 0,
                "design_json": "",
                "cif_file": "",
                "sample_dir": structure_dir,
                "nll": "",
                "design_dir": "",
                "af3_output_dir": Path(structure_dir).name,
                "iptm": row.get("iptm", ""),
                "ptm": row.get("ptm", ""),
                "ranking_score": row.get("ranking_score", ""),
                "iteration": 0,
            })

    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return str(output_path)
