"""Cluster optimisation of protein binder sequences with optional off-target scoring."""

from __future__ import annotations

import csv
import json
import logging
import math
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .af3_interface import AlphaFold3Interface, prediction_files
from .config import RunConfig
from .injection import SequenceInjector
from .losses import ColabDesignLossCalculator, calculate_structure_losses_from_af3
from .models import MultiHeadSurrogate
from .mpnn_scorer import ProteinMPNNScorer
from .sampling import MultiMetricGuidedSampler
from .weight_learner import AdaptiveWeightLearner
from .oracles.af3 import translate_confidences

logger = logging.getLogger(__name__)

# Metrics extracted directly from AF3 evaluation results into all_metrics.
_CORE_METRICS = [
    "plddt",
    "pae_intra",
    "pae_inter",
    "con_intra",
    "con_inter",
    "rg",
    "helix",
    "termini",
    "l_rmsd",
    "min_pae_interaction",
    "ptm_binder",
    "ptm",
]


def _metric_value(metric, result, where=None):
    value = result.get(metric)
    return float("nan") if value is None else float(value)


class AF3Evaluator:
    """Evaluates candidate sequences with AF3 and computes ColabDesign losses on the result."""

    def __init__(
        self,
        af3_interface: AlphaFold3Interface,
        structure_dir: str,
        rmsd_calculator=None,
        current_iteration: int = 0,
    ):
        self.af3_interface = af3_interface
        self.structure_dir = Path(structure_dir)
        self.rmsd_calculator = rmsd_calculator
        self.current_iteration = current_iteration

    def evaluate(self, sequences: List[str]) -> List[Dict]:
        """Run AF3 + loss calculation for each sequence; failures are skipped, not fatal."""
        self.current_iteration += 1
        results = []
        failed = []

        for i, seq in enumerate(sequences):
            try:
                import hashlib

                seq_hash = hashlib.md5(seq.encode()).hexdigest()[:16]
                seq_id = (
                    f"mutation_iter{self.current_iteration:03d}_seq{i:03d}_{seq_hash}"
                )

                confidence, metrics = self.af3_interface.predict(
                    binder_sequence=seq,
                    save_structure=True,
                    structure_dir=str(self.structure_dir),
                    sequence_id=seq_id,
                )

                native = metrics
                structure = native.get("structure_file")
                confidence_json = native.get("json_file")
                if not structure or not confidence_json:
                    raise ValueError(
                        "AF3 did not return the saved model and confidence paths"
                    )
                result = dict(
                    native,
                    sequence=seq,
                    confidence=confidence,
                    binder_chain=getattr(self.af3_interface, "binder_chain", "B"),
                    **translate_confidences(
                        native, getattr(self.af3_interface, "binder_chain", "B")
                    ),
                    structure_file=structure,
                    json_file=confidence_json,
                    structure_dir=str(Path(structure).parent),
                )
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "AF3 failed for sequence %d/%d (%s...): %s",
                    i + 1,
                    len(sequences),
                    seq[:30],
                    exc,
                )
                failed.append(seq)

        if failed:
            logger.warning(
                "%d/%d sequences failed AF3 prediction", len(failed), len(sequences)
            )
        if not results:
            raise RuntimeError(f"All {len(sequences)} sequences failed AF3 prediction")

        for result in results:
            losses = calculate_structure_losses_from_af3(
                result["structure_file"],
                result["json_file"],
                rmsd_calculator=self.rmsd_calculator,
                binder_chain=result.get("binder_chain", "B"),
            )
            residue_level = losses.pop("residue_level", None)
            result.update(losses)
            if residue_level is not None:
                result["residue_metrics"] = residue_level

        return results


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value, default=np.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ProteinOptimizer:
    """Target-agnostic surrogate-guided protein binder optimization loop."""

    def __init__(
        self,
        interface: AlphaFold3Interface,
        config: RunConfig,
        neg_interface: Optional[AlphaFold3Interface] = None,
    ):
        """
        Args:
            interface: primary (on-target) AF3 interface.
            config: resolved RunConfig for this run.
            neg_interface: optional off-target AF3 interface; when provided,
                enables selectivity scoring (positive_score - penalty * negative_score).
        """
        self.interface = interface
        self.neg_interface = neg_interface
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)

        self.surrogate = MultiHeadSurrogate(
            esm2_model=config.esm2_model,
            rf_params=config.rf_params,
            metric_weights=config.metric_weights,
            af3_thresholds=config.af3_thresholds,
            mutants_per_round=config.mutants_per_round,
            calculate_rmsd=config.calculate_rmsd,
            reference_structures_dir=config.reference_structures_dir,
            device=config.device,
            batch_size_gpu=config.batch_size_gpu,
            batch_size_cpu=config.batch_size_cpu,
        )

        self.neg_surrogate = None
        if self.neg_interface is not None:
            self.neg_surrogate = MultiHeadSurrogate(
                esm2_model=config.esm2_model,
                rf_params=config.rf_params,
                metric_weights=config.metric_weights,
                af3_thresholds=config.af3_thresholds,
                mutants_per_round=config.mutants_per_round,
                calculate_rmsd=False,
                device=config.device,
                batch_size_gpu=config.batch_size_gpu,
                batch_size_cpu=config.batch_size_cpu,
            )
            logger.info(
                "Selectivity mode enabled: %s (positive) vs off-target (negative)",
                config.target_name,
            )

        self.sampler = MultiMetricGuidedSampler(
            amino_acids=config.amino_acids,
            residue_sampling=config.residue_sampling,
            top_n_for_sampling=config.top_n_for_sampling,
        )

        self.structure_dir = Path(config.af3_structure_dir)
        self.evaluator = AF3Evaluator(
            interface, config.af3_structure_dir, self.surrogate.rmsd_calculator
        )
        self.neg_evaluator = (
            AF3Evaluator(
                neg_interface, str(Path(config.af3_structure_dir) / "off_target"), None
            )
            if neg_interface is not None
            else None
        )

        self.injector = None
        if config.injection_enabled:
            self.injector = SequenceInjector(config.injection_sources)
            logger.info(
                "Injection enabled: every %d iterations, %d seqs/injection",
                config.injection_interval,
                config.n_seqs_to_inject,
            )

        self.weight_learner = None
        if config.learn_weights:
            self.weight_learner = AdaptiveWeightLearner(config.af3_thresholds)
            logger.info("Adaptive weight learning enabled")

        self.mpnn_scorer = None
        if getattr(config, "active_nll_scoring", True):
            self.mpnn_scorer = ProteinMPNNScorer(
                binder_chain=getattr(interface, "binder_chain", "B")
            )
            if not self.mpnn_scorer.enabled:
                logger.warning(
                    "active_nll_scoring requested but ProteinMPNNScorer could not be enabled "
                    "(missing repo/env paths); nll will fall back to NaN for freshly-evaluated sequences."
                )

        self.off_target_records = {}
        self.output_path: Optional[Path] = None
        self.optimization_log: List[Dict] = []
        self.sequence_to_af3_dir: Dict[str, str] = {}

    def _find_structure_for_sequence(self, sequence: str) -> Optional[str]:
        """Resolve the AF3 model.cif path for a previously-evaluated sequence."""
        af3_dir = self.sequence_to_af3_dir.get(sequence)
        if not af3_dir:
            return None

        structure_path = self.structure_dir / str(af3_dir).strip()
        if not structure_path.exists():
            return None

        try:
            return str(prediction_files(structure_path)[0])
        except FileNotFoundError:
            return None

    def _filter_sequences_with_structures(
        self, sequences: List[str], metrics_dict: Dict[str, List[float]]
    ) -> Tuple[List[str], Dict[str, List[float]]]:
        """Keep only sequences for which a structure file can be resolved."""
        filtered_sequences = []
        filtered_metrics = {metric: [] for metric in metrics_dict}
        for i, sequence in enumerate(sequences):
            if self._find_structure_for_sequence(sequence):
                filtered_sequences.append(sequence)
                for metric in metrics_dict:
                    filtered_metrics[metric].append(metrics_dict[metric][i])
        return filtered_sequences, filtered_metrics

    def _calculate_combined_scores(
        self,
        metrics_dict: Dict[str, List[float]],
        neg_metrics_dict: Optional[Dict[str, List[float]]] = None,
        negative_penalty_weight: float = 2.5,
    ) -> List[float]:
        """Weighted combined score; subtracts a penalized off-target score when neg_metrics_dict is given."""
        from .objectives import weighted_scores

        if not metrics_dict:
            return []
        scores = weighted_scores(metrics_dict, self.config.metric_weights)
        if neg_metrics_dict is not None:
            if "i_ptm" not in neg_metrics_dict:
                raise ValueError(
                    "Selectivity requires measured or predicted off-target i_ptm"
                )
            off = np.asarray(neg_metrics_dict["i_ptm"], dtype=float)
            if off.shape != scores.shape or not np.isfinite(off).all():
                raise ValueError(
                    "Off-target scores must align with the on-target batch"
                )
            scores -= negative_penalty_weight * off
        return scores.tolist()

    def _extract_metrics_from_results(
        self, eval_results: List[Dict]
    ) -> Dict[str, List[float]]:
        """Pull the core metric set (+ i_ptm, + nll passthrough) out of AF3Evaluator results."""
        metrics: Dict[str, List[float]] = {}
        if not eval_results:
            return metrics

        available = set().union(*(r.keys() for r in eval_results))
        for metric in _CORE_METRICS:
            if metric in available:
                metrics[metric] = [
                    _metric_value(metric, r, "_extract_metrics_from_results")
                    for r in eval_results
                ]

        if "i_ptm" in available:
            metrics["i_ptm"] = [r.get("i_ptm", np.nan) for r in eval_results]
        elif "iptm" in available:
            metrics["i_ptm"] = [r.get("iptm", np.nan) for r in eval_results]

        nll_values = []
        for r in eval_results:
            passthrough = r.get("nll", np.nan)
            if passthrough is not None and not (
                isinstance(passthrough, float) and np.isnan(passthrough)
            ):
                nll_values.append(passthrough)
                continue

            structure_file = r.get("structure_file")
            sequence = r.get("sequence")
            if (
                self.mpnn_scorer is not None
                and self.mpnn_scorer.enabled
                and structure_file
                and sequence
            ):
                nll_values.append(self.mpnn_scorer.score(structure_file, sequence))
            else:
                nll_values.append(np.nan)
        metrics["nll"] = nll_values
        return metrics

    def _merge_metrics(self, all_metrics, new_metrics):
        """Append aligned metric batches, retaining missing labels as NaN."""
        n_old = len(next(iter(all_metrics.values()))) if all_metrics else 0
        n_new = len(next(iter(new_metrics.values()))) if new_metrics else 0
        if any(len(v) != n_new for v in new_metrics.values()):
            raise ValueError("Metric columns have inconsistent lengths")
        for metric in set(all_metrics) | set(new_metrics):
            all_metrics.setdefault(metric, [np.nan] * n_old)
            all_metrics[metric].extend(new_metrics.get(metric, [np.nan] * n_new))

    def _load_initial_data(
        self, initial_csv: str
    ) -> Tuple[List[str], Dict[str, List[float]]]:
        """Load the initial CSV, resolve structure files, and compute initial metrics."""
        rows = _read_csv_rows(initial_csv)
        if rows and "error" in rows[0]:
            rows = [r for r in rows if not r.get("error")]

        seq_col = "sequence" if (rows and "sequence" in rows[0]) else "truncated_seq"
        if rows and seq_col not in rows[0]:
            raise ValueError(
                "Initial CSV must have a 'sequence' or 'truncated_seq' column"
            )

        valid_sequences, valid_af3_dirs, valid_nll = [], [], []
        for row in rows:
            seq = row.get(seq_col)
            af3_dir = row.get("af3_output_dir")
            if not seq or not af3_dir:
                continue
            relative = Path(str(af3_dir).strip())
            candidates = (
                self.structure_dir / relative,
                self.structure_dir.parent / relative,
            )
            structure_path = next((path for path in candidates if path.is_dir()), None)
            if structure_path is None:
                continue
            try:
                prediction_files(structure_path)
            except FileNotFoundError:
                continue
            cleaned = str(structure_path.resolve())
            valid_sequences.append(seq)
            valid_af3_dirs.append(cleaned)
            valid_nll.append(_safe_float(row.get("nll")))

        if not valid_sequences:
            raise ValueError("No sequences found with corresponding structures")

        self.sequence_to_af3_dir = dict(zip(valid_sequences, valid_af3_dirs))

        initial_eval_results = []
        kept_sequences, kept_nll = [], []
        for seq, af3_dir, nll in zip(valid_sequences, valid_af3_dirs, valid_nll):
            structure_path = self.structure_dir / af3_dir
            try:
                model_file, summary_json, json_path = prediction_files(structure_path)
            except FileNotFoundError:
                continue

            try:
                with open(summary_json) as f:
                    summary = json.load(f)
                initial_eval_results.append(
                    {
                        "sequence": seq,
                        "structure_file": str(model_file),
                        "json_file": str(json_path),
                        "binder_chain": getattr(self.interface, "binder_chain", "B"),
                        "structure_dir": str(structure_path),
                        **translate_confidences(
                            summary, getattr(self.interface, "binder_chain", "B")
                        ),
                    }
                )
                kept_sequences.append(seq)
                kept_nll.append(nll)
            except Exception as exc:
                logger.warning(
                    "Could not process initial structure for %s...: %s", seq[:20], exc
                )

        if not initial_eval_results:
            raise ValueError("No valid initial structures could be processed")

        initial_eval_results = self.surrogate.calculate_structure_metrics(
            initial_eval_results
        )

        _MAX_INITIAL_NLL = getattr(self.config, "initial_nll_score_limit", 100)
        n_missing_nll = sum(
            1
            for v in kept_nll
            if (v is None or (isinstance(v, float) and math.isnan(v)))
        )
        if (
            n_missing_nll > 0
            and n_missing_nll <= _MAX_INITIAL_NLL
            and self.mpnn_scorer is not None
            and self.mpnn_scorer.enabled
        ):
            nll_computed = 0
            nll_values_seen = []
            print(
                f"[NLL init] Scoring {n_missing_nll} initial sequences with ProteinMPNN (~3-5s each)..."
            )
            for i, (result, seq, nll) in enumerate(
                zip(initial_eval_results, kept_sequences, kept_nll)
            ):
                if math.isnan(nll) if isinstance(nll, float) else nll is None:
                    structure_file = result.get("structure_file")
                    if structure_file:
                        try:
                            scored = self.mpnn_scorer.score(structure_file, seq)
                            kept_nll[i] = scored
                            nll_computed += 1
                            nll_values_seen.append(scored)
                            mean_so_far = sum(nll_values_seen) / len(nll_values_seen)
                            print(
                                f"  [{nll_computed}/{n_missing_nll}] seq {i:4d}  "
                                f"nll={scored:.4f}  mean={mean_so_far:.4f}",
                                flush=True,
                            )
                        except Exception as e:
                            logger.debug(
                                "NLL score failed for initial seq %d: %s", i, e
                            )
            if nll_computed:
                final_mean = sum(nll_values_seen) / len(nll_values_seen)
                print(
                    f"[NLL init] Done: {nll_computed}/{n_missing_nll} scored  "
                    f"mean={final_mean:.4f}  min={min(nll_values_seen):.4f}  max={max(nll_values_seen):.4f}"
                )
                logger.info(
                    "Computed NLL for %d/%d initial sequences via ProteinMPNN score_only",
                    nll_computed,
                    len(kept_nll),
                )
        elif n_missing_nll > _MAX_INITIAL_NLL:
            logger.info(
                "Skipping initial NLL computation for %d sequences (> %d threshold -- too slow). "
                "NLL will be computed for newly generated candidates during optimization instead.",
                n_missing_nll,
                _MAX_INITIAL_NLL,
            )

        all_metrics: Dict[str, List[float]] = {}
        final_sequences = []
        available_metrics = set().union(*(r.keys() for r in initial_eval_results))
        for result, seq, nll in zip(initial_eval_results, kept_sequences, kept_nll):
            if (
                "plddt" not in result
            ):  # plddt always computed; l_rmsd only when calculate_rmsd=True
                continue
            final_sequences.append(seq)
            for metric in _CORE_METRICS:
                if metric in available_metrics:
                    all_metrics.setdefault(metric, []).append(
                        _metric_value(metric, result)
                    )
            all_metrics.setdefault("i_ptm", []).append(result.get("i_ptm", np.nan))
            all_metrics.setdefault("nll", []).append(nll)

        if not all_metrics:
            raise ValueError("No valid metrics found in initial data")

        logger.info(
            "Loaded %d initial sequences with %d metrics",
            len(final_sequences),
            len(all_metrics),
        )
        return final_sequences, all_metrics

    def run(self, initial_csv: str, output_dir: Optional[str] = None) -> Dict:
        """Run the full optimization loop. Returns a dict of final sequences/metrics/scores."""
        output_dir = output_dir or self.config.output_dir
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.config.resume_from:
            all_sequences, all_metrics, all_iterations, start_iteration = (
                self._load_checkpoint(self.config.resume_from)
            )
        else:
            sequences, all_metrics = self._load_initial_data(initial_csv)
            logger.info("Training initial surrogate model...")
            self.surrogate.train(sequences, all_metrics)

            all_sequences = list(sequences)
            all_iterations = [0] * len(sequences)
            start_iteration = 1
            self._save_checkpoint(all_sequences, all_metrics, all_iterations, 0)

        self._update_off_target(all_sequences)
        self.evaluator.current_iteration = start_iteration - 1

        for iteration in range(start_iteration, self.config.n_iterations + 1):
            logger.info("=== Iteration %d/%d ===", iteration, self.config.n_iterations)
            self._run_iteration(iteration, all_sequences, all_metrics, all_iterations)
            self._save_checkpoint(all_sequences, all_metrics, all_iterations, iteration)
            # Write running partial results after every iteration -- sorted by combined score.
            # Overwrites each time; safe to read at any point without waiting for the full run.
            partial_combined = self._measured_scores(all_sequences, all_metrics)
            partial_results = {
                "sequence": all_sequences,
                "combined_score": partial_combined,
                "iteration": all_iterations,
                **all_metrics,
            }
            self._write_csv(self.output_path / "partial_results.csv", partial_results)
            logger.info(
                "partial_results.csv updated (%d sequences, iter %d/%d)",
                len(all_sequences),
                iteration,
                self.config.n_iterations,
            )

        final_combined = self._measured_scores(all_sequences, all_metrics)
        final_results = {
            "sequence": all_sequences,
            "combined_score": final_combined,
            "iteration": all_iterations,
            **all_metrics,
        }
        self._write_csv(self.output_path / "final_results.csv", final_results)
        with open(self.output_path / "optimization_log.json", "w") as f:
            json.dump(self.optimization_log, f, indent=2)

        if self.weight_learner is not None:
            self.weight_learner.save_history(
                str(self.output_path / "weight_learning_history.json")
            )

        return final_results

    def _update_off_target(self, sequences):
        """Measure missing off-target labels and train their separate surrogate."""
        if self.neg_evaluator is None:
            return
        todo = [seq for seq in sequences if seq not in self.off_target_records]
        if todo:
            for result in self.neg_evaluator.evaluate(todo):
                self.off_target_records[result["sequence"]] = result
        missing = [seq for seq in sequences if seq not in self.off_target_records]
        if missing:
            raise ValueError(
                "Off-target evaluation did not complete for the sequence batch"
            )
        seqs = list(self.off_target_records)
        self.neg_surrogate.train(
            seqs, {"i_ptm": [self.off_target_records[seq]["i_ptm"] for seq in seqs]}
        )

    def _measured_scores(self, sequences, metrics):
        off = (
            None
            if self.neg_evaluator is None
            else {"i_ptm": [self.off_target_records[seq]["i_ptm"] for seq in sequences]}
        )
        return self._calculate_combined_scores(metrics, off)

    def _run_iteration(
        self,
        iteration: int,
        all_sequences: List[str],
        all_metrics: Dict[str, List[float]],
        all_iterations: List[int],
    ) -> None:
        injected_sequences: List[str] = []

        if (
            self.injector is not None
            and self.config.injection_enabled
            and iteration % self.config.injection_interval == 0
        ):
            injected_sequences = self._handle_injection(
                iteration, all_sequences, all_metrics, all_iterations
            )

        current_combined = self._measured_scores(all_sequences, all_metrics)

        eligible = np.flatnonzero(np.isfinite(current_combined))
        top_n = min(self.config.top_n_for_sampling, len(eligible))
        if top_n == 0:
            raise ValueError("No fully scored parent sequences")
        top_indices = eligible[
            np.argsort(np.asarray(current_combined)[eligible])[-top_n:]
        ]
        top_sequences = [all_sequences[i] for i in top_indices]

        top_sequences, top_residue_metrics = self._extract_residue_metrics(
            top_sequences
        )
        if not top_sequences:
            logger.warning(
                "No parent sequences with resolvable structures; stopping early"
            )
            return

        candidate_mutations = self.sampler.generate_candidates(
            top_sequences,
            top_residue_metrics,
            current_iteration=iteration,
            max_iterations=self.config.n_iterations,
        )
        candidate_mutations = sorted(set(candidate_mutations) - set(all_sequences))
        if not candidate_mutations:
            logger.info("No new mutation candidates, stopping")
            return

        n_select = min(self.config.mutants_per_round, len(candidate_mutations))
        if self.neg_surrogate is None:
            selected_sequences, _ = self.surrogate.top_n_selection(
                candidate_mutations, n_select
            )
        else:
            scores = self._calculate_combined_scores(
                self.surrogate.predict(candidate_mutations),
                self.neg_surrogate.predict(candidate_mutations),
            )
            eligible = np.flatnonzero(np.isfinite(scores))
            if len(eligible) < n_select:
                raise ValueError("Not enough finite selectivity scores")
            selected = eligible[
                np.argsort(np.asarray(scores)[eligible])[-n_select:][::-1]
            ]
            selected_sequences = [candidate_mutations[i] for i in selected]

        eval_results = self.evaluator.evaluate(selected_sequences)
        for result in eval_results:
            self.sequence_to_af3_dir[result["sequence"]] = Path(
                result["structure_dir"]
            ).name

        new_metrics = self._extract_metrics_from_results(eval_results)
        if not new_metrics:
            raise ValueError("No metrics extracted from evaluation results")

        new_sequences = [
            r.get("sequence", selected_sequences[i]) for i, r in enumerate(eval_results)
        ]
        filtered_new_sequences, filtered_new_metrics = (
            self._filter_sequences_with_structures(new_sequences, new_metrics)
        )

        if not filtered_new_sequences:
            logger.info(
                "No new mutation sequences had structures, skipping training update"
            )
            self.optimization_log.append(
                {
                    "iteration": iteration,
                    "sequences_evaluated": len(selected_sequences),
                    "sequences_with_structures": 0,
                    "total_sequences": len(all_sequences),
                    "best_new_score": 0,
                    "overall_best_score": max(
                        self._calculate_combined_scores(all_metrics)
                    ),
                    "improvement_from_initial": 0,
                }
            )
            return

        self._update_off_target(filtered_new_sequences)
        eval_combined = self._measured_scores(
            filtered_new_sequences, filtered_new_metrics
        )

        all_sequences.extend(filtered_new_sequences)
        self._merge_metrics(all_metrics, filtered_new_metrics)
        all_iterations.extend([iteration] * len(filtered_new_sequences))

        self.surrogate.train(all_sequences, all_metrics)

        # Print cross-val scores every iteration so training quality is immediately visible.
        self.surrogate.cross_val_scores(all_metrics)

        self._maybe_relearn_weights(iteration, all_sequences, all_metrics)

        best_new = max(eval_combined) if eval_combined else 0
        overall_best = max(self._measured_scores(all_sequences, all_metrics))
        initial_size = (
            len(all_sequences) - len(filtered_new_sequences) - len(injected_sequences)
        )
        initial_combined = self._measured_scores(
            all_sequences[:initial_size],
            {m: vals[:initial_size] for m, vals in all_metrics.items()},
        )
        initial_best = max(initial_combined) if initial_combined else 0

        self.optimization_log.append(
            {
                "iteration": iteration,
                "sequences_evaluated": len(selected_sequences),
                "sequences_with_structures": len(filtered_new_sequences),
                "total_sequences": len(all_sequences),
                "best_new_score": best_new,
                "overall_best_score": overall_best,
                "improvement_from_initial": overall_best - initial_best,
                **(
                    {"injected_sequences": len(injected_sequences)}
                    if injected_sequences
                    else {}
                ),
            }
        )
        logger.info(
            "Best new: %.3f, overall: %.3f, improvement: %+.3f",
            best_new,
            overall_best,
            overall_best - initial_best,
        )

    def _handle_injection(
        self,
        iteration: int,
        all_sequences: List[str],
        all_metrics: Dict[str, List[float]],
        all_iterations: List[int],
    ) -> List[str]:
        """Evaluate injected sequences with AF3 and fold them into the running dataset."""
        already_evaluated = set(all_sequences)
        candidates = self.injector.get_sequences_to_inject(
            iteration=iteration,
            n_seqs=self.config.n_seqs_to_inject,
            already_evaluated=already_evaluated,
        )
        if not candidates:
            return []

        candidate_seqs = [seq for seq, _ in candidates]
        nll_by_seq = dict(candidates)

        logger.info(
            "[Injection] Evaluating %d new sequences with AF3...", len(candidate_seqs)
        )
        eval_results = self.evaluator.evaluate(candidate_seqs)
        for result in eval_results:
            result["nll"] = nll_by_seq.get(result["sequence"], np.nan)
            self.sequence_to_af3_dir[result["sequence"]] = Path(
                result["structure_dir"]
            ).name

        temp_metrics = self._extract_metrics_from_results(eval_results)
        new_sequences = [r.get("sequence") for r in eval_results]
        injected_sequences, injected_metrics = self._filter_sequences_with_structures(
            new_sequences, temp_metrics
        )

        if not injected_sequences:
            logger.info("[Injection] No injected sequences had valid structures")
            return []

        self._update_off_target(injected_sequences)
        all_sequences.extend(injected_sequences)
        self._merge_metrics(all_metrics, injected_metrics)
        all_iterations.extend([iteration] * len(injected_sequences))

        logger.info(
            "[Injection] Retraining surrogate with %d injected sequences",
            len(injected_sequences),
        )
        self.surrogate.train(all_sequences, all_metrics)

        return injected_sequences

    def _extract_residue_metrics(
        self, sequences: List[str]
    ) -> Tuple[List[str], List[Dict]]:
        """Compute per-residue metrics for sequences with resolvable structures (used for mutation guidance)."""
        kept_sequences, residue_metrics = [], []
        for seq in sequences:
            structure_file_path = self._find_structure_for_sequence(seq)
            if not structure_file_path:
                continue
            json_path = Path(structure_file_path).with_name(
                Path(structure_file_path).name.replace(
                    "_model.cif", "_confidences.json"
                )
            )
            if not json_path.exists():
                continue
            calculator = ColabDesignLossCalculator(
                str(structure_file_path),
                str(json_path),
                binder_chain=getattr(self.interface, "binder_chain", "B"),
            )
            metrics = calculator.get_residue_level_metrics()
            if metrics is not None:
                residue_metrics.append(metrics)
                kept_sequences.append(seq)
        return kept_sequences, residue_metrics

    def _maybe_relearn_weights(
        self,
        iteration: int,
        all_sequences: List[str],
        all_metrics: Dict[str, List[float]],
    ) -> None:
        """Optionally retrain metric_weights via AdaptiveWeightLearner (disabled unless config.learn_weights)."""
        if self.weight_learner is None:
            return
        if (
            iteration < self.config.weight_retrain_start
            or iteration % self.config.weight_retrain_interval != 0
        ):
            return

        learned_weights = self.weight_learner.learn_weights(
            all_metrics, list(self.config.metric_weights.keys()), iteration=iteration
        )
        if not learned_weights:
            return

        for metric, weight in learned_weights.items():
            if metric in self.config.metric_weights:
                self.config.metric_weights[metric] = weight
        self.surrogate.metric_weights = self.config.metric_weights

        current_combined = self._calculate_combined_scores(all_metrics)
        hit_rate_stats = self.weight_learner.calculate_hit_rate(
            all_metrics, np.array(current_combined), top_n=min(100, len(all_sequences))
        )
        logger.info(
            "Hit rate after reweighting: %.1f%% (enrichment %.2fx)",
            hit_rate_stats["hit_rate"] * 100,
            hit_rate_stats["enrichment"],
        )
        self.surrogate.train(all_sequences, all_metrics)

    @staticmethod
    def _write_csv(path: Path, data: Dict[str, List]) -> None:
        keys = list(data.keys())
        n_rows = len(data[keys[0]]) if keys else 0
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(n_rows):
                writer.writerow([data[k][i] for k in keys])

    def _save_checkpoint(
        self,
        sequences: List[str],
        metrics: Dict[str, List[float]],
        iterations: List[int],
        iteration: int,
    ) -> None:
        checkpoint = {
            "scoring_version": 2,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "off_target_records": self.off_target_records,
            "iteration": iteration,
            "all_sequences": sequences,
            "all_metrics": metrics,
            "all_iterations": iterations,
            "optimization_log": self.optimization_log,
            "sequence_to_af3_dir": self.sequence_to_af3_dir,
            "timestamp": time.time(),
        }
        with open(self.output_path / f"checkpoint_iter_{iteration}.pkl", "wb") as f:
            pickle.dump(checkpoint, f)
        self.surrogate.save_model(
            str(self.output_path / f"surrogate_iter_{iteration}.pkl")
        )
        with open(self.output_path / "latest_checkpoint.txt", "w") as f:
            f.write(str(iteration))

        if iteration % max(self.config.checkpoint_frequency, 1) == 0:
            logger.info(
                "Checkpoint saved at iteration %d (%d sequences)",
                iteration,
                len(sequences),
            )

    def _load_checkpoint(
        self, checkpoint_path: str
    ) -> Tuple[List[str], Dict[str, List[float]], List[int], int]:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        if checkpoint.get("scoring_version") != 2:
            raise ValueError("Checkpoint uses a different structural scoring version")
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
        self.off_target_records = checkpoint.get("off_target_records", {})
        self.optimization_log = checkpoint.get("optimization_log", [])
        self.sequence_to_af3_dir = checkpoint.get("sequence_to_af3_dir", {})

        surrogate_file = (
            Path(checkpoint_path).parent
            / f"surrogate_iter_{checkpoint['iteration']}.pkl"
        )
        if surrogate_file.exists():
            self.surrogate.load_model(str(surrogate_file))

        return (
            checkpoint["all_sequences"],
            checkpoint["all_metrics"],
            checkpoint["all_iterations"],
            checkpoint["iteration"] + 1,
        )
