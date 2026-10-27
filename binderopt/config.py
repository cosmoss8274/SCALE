"""RunConfig: the single configuration object threaded through the pipeline.

A RunConfig is built by deep-merging ``configs/defaults.yaml`` with a target
yaml file (``configs/targets/<name>.yaml``), then applying CLI overrides.
Target values win over defaults; CLI flags win over both.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass
class RunConfig:
    """Resolved configuration for a single optimization run.

    Most fields mirror the keys found in ``configs/defaults.yaml`` and
    ``configs/targets/<name>.yaml``. Unrecognized/extra YAML keys are kept in
    ``extra`` so that target-specific blocks (e.g. ``selectivity``,
    ``residue_sampling``) survive without requiring a dataclass field for
    every possible key.
    """

    target_name: str
    target_sequence: str = ""

    # Optimization loop
    n_iterations: int = 50
    mutants_per_round: int = 30
    top_n_for_sampling: int = 30
    checkpoint_frequency: int = 1

    # Model / embeddings
    esm2_model: str = "esm2_t36_3B_UR50D"
    surrogate_model: str = "random_forest"
    batch_size_gpu: int = 8
    batch_size_cpu: int = 32
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"

    # Evaluation
    use_af3: bool = True
    use_protenix: bool = False
    calculate_rmsd: bool = True
    reference_structures_dir: Optional[str] = None
    # Empty means "read SCALE_AF3_DB_DIR / SCALE_AF3_MODEL_DIR at the point of use"; see
    # binderopt/external.py. Left empty rather than defaulted to a path, which would only
    # ever be right on one machine.
    af3_db_dir: str = ""
    af3_model_dir: str = ""
    af3_structure_dir: str = ""
    af3_target_msa_dir: str = ""

    # Weights / thresholds
    metric_weights: Dict[str, float] = field(default_factory=dict)
    af3_thresholds: Dict[str, float] = field(default_factory=dict)

    # Weight learning (optional, off by default)
    learn_weights: bool = False
    weight_retrain_start: int = 0
    weight_retrain_interval: int = 9999

    # Injection (optional, off by default)
    injection_enabled: bool = False
    injection_interval: int = 3
    n_seqs_to_inject: int = 50
    injection_sources: Dict[str, Any] = field(default_factory=dict)
    use_injected_as_parents: bool = True

    # Active per-candidate ProteinMPNN NLL scoring (on by default; degrades
    # gracefully to NaN if the ProteinMPNN repo/env isn't available -- see
    # binderopt/mpnn_scorer.py).
    active_nll_scoring: bool = True

    # Max number of initial sequences allowed to have NLL back-filled via
    # ProteinMPNN score_only at startup. Each call takes ~3-5s (subprocess
    # spawn), so the default of 100 keeps startup overhead under ~8 min.
    # Set to a large value (e.g. 9999) to score all initial sequences.
    initial_nll_score_limit: int = 100

    # Residue-level guided sampling
    residue_sampling: Dict[str, Any] = field(default_factory=dict)

    # Random forest hyperparameters
    rf_params: Dict[str, Any] = field(default_factory=dict)

    # Data
    initial_csv: str = ""

    # Selectivity (optional dual-target scoring; off unless target yaml sets it)
    selectivity: Optional[Dict[str, Any]] = None

    # I/O
    output_dir: str = ""
    device: str = "cpu"
    seed: int = 0
    resume_from: Optional[str] = None

    # Catch-all for anything else present in the merged yaml
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Build a RunConfig from a merged config dict, stashing unknown keys in `extra`."""
        known_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        kwargs["extra"] = extra
        return cls(**kwargs)


def build_arg_parser(target_choices: list) -> argparse.ArgumentParser:
    """Construct the CLI argument parser for design.py."""
    parser = argparse.ArgumentParser(
        prog="design.py",
        description="Run a protein-binder optimization design loop for a given target.",
    )
    parser.add_argument(
        "--target_name",
        required=True,
        choices=sorted(target_choices),
        help="Target registry key (configs/targets/<target_name>.yaml).",
    )
    parser.add_argument("--iters", type=int, default=None, help="Override n_iterations.")
    parser.add_argument(
        "--mutants_per_round", type=int, default=None, help="Override mutants_per_round."
    )
    parser.add_argument(
        "--top_n_for_sampling", type=int, default=None, help="Override top_n_for_sampling."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: scale/results/<target_name>_<timestamp>).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--learn_weights",
        action="store_true",
        help="Enable adaptive weight learning (off by default).",
    )
    parser.add_argument(
        "--no_injection",
        action="store_true",
        help="Force-disable sequence injection regardless of target config.",
    )
    parser.add_argument(
        "--no_active_nll_scoring",
        action="store_true",
        help="Force-disable active per-candidate ProteinMPNN NLL scoring (on by default).",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Path to a checkpoint to resume from."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device hint ('cuda' or 'cpu'); passed through to ESM2 extractor.",
    )
    return parser


def apply_cli_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Apply CLI argument overrides on top of a merged RunConfig (CLI wins)."""
    if args.iters is not None:
        config.n_iterations = args.iters
    if args.mutants_per_round is not None:
        config.mutants_per_round = args.mutants_per_round
    if args.top_n_for_sampling is not None:
        config.top_n_for_sampling = args.top_n_for_sampling
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    config.seed = args.seed
    if args.learn_weights:
        config.learn_weights = True
    if args.no_injection:
        config.injection_enabled = False
    if args.no_active_nll_scoring:
        config.active_nll_scoring = False
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.device is not None:
        config.device = args.device
    return config
