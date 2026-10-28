#!/usr/bin/env python3
"""CLI entry point for the binderopt protein-binder optimization pipeline.

Usage:
    python3 design.py --target_name myb --iters 10
    python3 design.py --target_name her2 --iters 30 --mutants_per_round 20 --no_injection
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from binderopt import targets as targets_module
from binderopt.af3_interface import AlphaFold3Interface
from binderopt.config import apply_cli_overrides, build_arg_parser
from binderopt.optimizer import ProteinOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("design")


def main() -> None:
    target_choices = targets_module.list_targets()
    parser = build_arg_parser(target_choices)
    args = parser.parse_args()

    config = targets_module.load_target(args.target_name)
    config = apply_cli_overrides(config, args)

    if args.output_dir is None:
        if config.resume_from:
            # Resume into the same directory the checkpoint came from
            config.output_dir = str(Path(config.resume_from).resolve().parent)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scale_root = Path(__file__).resolve().parent
            config.output_dir = str(
                scale_root / "results" / f"{config.target_name}_{timestamp}"
            )

    if config.device is None or config.device == "cpu":
        try:
            import torch

            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            config.device = "cpu"

    logger.info("Target: %s", config.target_name)
    logger.info("Output dir: %s", config.output_dir)
    logger.info(
        "Iterations: %d, mutants/round: %d, top_n_for_sampling: %d",
        config.n_iterations,
        config.mutants_per_round,
        config.top_n_for_sampling,
    )

    initial_csv = config.initial_csv
    rfdiffusion_csv_path = config.extra.get("rfdiffusion_csv_path")
    if rfdiffusion_csv_path:
        initial_csv = targets_module.ensure_benchmark_csv(
            rfdiffusion_csv_path=rfdiffusion_csv_path,
            target_name=config.target_name,
            target_sequence=config.target_sequence,
            output_csv_path=initial_csv,
        )

    interface = AlphaFold3Interface(
        db_dir=config.af3_db_dir,
        model_dir=config.af3_model_dir,
        target_names=[config.target_name.upper()],
        target_sequences=[config.target_sequence],
        af3_results_dir=str(Path(config.output_dir) / "af3_on_target"),
    )

    neg_interface = None
    selectivity = config.selectivity or config.extra.get("selectivity")
    if selectivity:
        logger.info("Selectivity mode: off-target = %s", selectivity.get("target_name"))
        neg_interface = AlphaFold3Interface(
            db_dir=config.af3_db_dir,
            model_dir=config.af3_model_dir,
            target_names=[selectivity["target_name"]],
            target_sequences=[selectivity["target_sequence"]],
            af3_results_dir=str(Path(config.output_dir) / "af3_off_target"),
        )

    optimizer = ProteinOptimizer(interface, config=config, neg_interface=neg_interface)

    try:
        optimizer.run(initial_csv=initial_csv, output_dir=config.output_dir)
    finally:
        interface.cleanup()
        if neg_interface is not None:
            neg_interface.cleanup()

    logger.info("Done. Results written to %s", config.output_dir)


if __name__ == "__main__":
    main()
