"""Extension registration, chain identity and independent off-target scoring."""

from types import SimpleNamespace
import numpy as np
import pytest

from binderopt.generators import GENERATORS, get_generator, register_generator
from binderopt.oracles import ORACLES, get_oracle, register_oracle
from binderopt.oracles.af3 import translate_confidences
from binderopt.optimizer import ProteinOptimizer, AF3Evaluator
from binderopt.targets import list_targets, load_target
from binderopt.mpnn_scorer import ProteinMPNNScorer


def test_custom_predictor_and_generator_registration(monkeypatch):
    monkeypatch.setitem(ORACLES, "test_adapter", None)
    monkeypatch.delitem(ORACLES, "test_adapter")
    monkeypatch.setitem(GENERATORS, "test_source", None)
    monkeypatch.delitem(GENERATORS, "test_source")
    register_oracle("test_adapter")(lambda **kwargs: kwargs)
    register_generator("test_source")(lambda **kwargs: kwargs)
    assert get_oracle("test_adapter", version=3) == {"version": 3}
    assert get_generator("test_source", count=4) == {"count": 4}
    with pytest.raises(KeyError):
        get_oracle("missing_adapter")
    with pytest.raises(KeyError):
        get_generator("missing_source")


def test_packaged_targets_resolve(monkeypatch, tmp_path):
    for key in [
        "SCALE_ASSETS",
        "SCALE_RESULTS",
        "SCALE_AF3_DB_DIR",
        "SCALE_AF3_MODEL_DIR",
    ]:
        monkeypatch.setenv(key, str(tmp_path / key))
    assert "trka" in list_targets()
    for name in list_targets():
        cfg = load_target(name)
        assert cfg.target_name == name and cfg.target_sequence


def test_chain_specific_af3_metrics_use_the_named_binder():
    metrics = translate_confidences(
        dict(
            chain_ids=["X", "Y", "Z"],
            chain_ptm=[0.1, 0.2, 0.9],
            chain_pair_pae_min=[[0, 0.01, 4], [0.01, 0, 6], [3, 5, 0]],
            chain_pair_iptm=[[0.1, 0.99, 0.6], [0.99, 0.2, 0.8], [0.6, 0.8, 0.9]],
        ),
        binder_chain="Z",
    )
    assert metrics["min_pae_interaction"] == 3
    assert metrics["ptm_binder"] == 0.9
    assert metrics["i_ptm"] == pytest.approx(0.7)


def test_mpnn_scores_the_explicit_binder_even_when_longer(prediction, tmp_path):
    scorer = ProteinMPNNScorer(enabled=False, binder_chain="B")
    assert (
        scorer._convert_cif_to_pdb(
            prediction[0], tmp_path / "model.pdb", expected_length=4
        )
        == "B"
    )
    assert (
        scorer._convert_cif_to_pdb(
            prediction[0], tmp_path / "invalid.pdb", expected_length=3
        )
        is None
    )


def test_production_af3_evaluator_uses_returned_model_and_measured_confidence(
    prediction, tmp_path
):
    model, confidence, _ = prediction
    native = dict(
        iptm=0.7,
        ranking_score=0.8,
        chain_ids=["A", "B"],
        chain_ptm=[0.9, 0.85],
        chain_pair_pae_min=[[0, 2], [3, 0]],
        chain_pair_iptm=[[0.9, 0.8], [0.8, 0.85]],
        structure_file=str(model),
        json_file=str(confidence),
    )
    interface = SimpleNamespace(
        binder_chain="B", predict=lambda **kwargs: (0.8, native)
    )
    result = AF3Evaluator(interface, str(tmp_path)).evaluate(["AAAA"])[0]
    assert result["i_ptm"] == pytest.approx(0.8)
    assert result["min_pae_interaction"] == 2
    np.testing.assert_allclose(result["residue_metrics"]["plddt"], [0.6, 0.7, 0.8, 0.9])


def test_off_target_surrogate_uses_separate_labels_and_reuses_measurements():
    calls = []
    fits = []

    def evaluate(sequences):
        calls.extend(sequences)
        return [dict(sequence=s, i_ptm={"A": 0.1, "B": 0.8}[s]) for s in sequences]

    model = ProteinOptimizer.__new__(ProteinOptimizer)
    model.neg_evaluator = SimpleNamespace(evaluate=evaluate)
    model.neg_surrogate = SimpleNamespace(
        train=lambda seq, metrics: fits.append((seq, metrics))
    )
    model.off_target_records = {}
    model.config = SimpleNamespace(metric_weights={"i_ptm": 1.0})
    model._update_off_target(["A", "B"])
    model._update_off_target(["A", "B"])
    assert calls == ["A", "B"]
    assert fits[-1][1]["i_ptm"] == [0.1, 0.8]
    assert model._measured_scores(["A", "B"], {"i_ptm": [0.9, 0.9]}) == pytest.approx(
        [0.65, -1.1]
    )


def test_metric_append_keeps_missing_labels_missing():
    model = ProteinOptimizer.__new__(ProteinOptimizer)
    old = {"i_ptm": [0.8], "rmsd": [1.0]}
    model._merge_metrics(old, {"i_ptm": [0.9], "nll": [2.0]})
    assert np.isnan(old["rmsd"][1]) and np.isnan(old["nll"][0])
    assert old["i_ptm"] == [0.8, 0.9]


def test_entirely_unmeasured_optional_score_is_inactive():
    from binderopt.objectives import weighted_scores

    scores = weighted_scores(
        {"i_ptm": [0.7, 0.8], "nll": [np.nan, np.nan]},
        {"i_ptm": 1.0, "nll": -1.0},
    )
    np.testing.assert_allclose(scores, [0.7, 0.8])


def test_af3_structure_and_confidence_come_from_same_ranked_model(tmp_path):
    import json
    from binderopt.af3_interface import prediction_files

    root = tmp_path / "prediction"
    sample = root / "seed-42_sample-0"
    sample.mkdir(parents=True)
    for folder, name in [(root, "ranked"), (sample, "sample")]:
        (folder / f"{name}_model.cif").write_text("data_test\n")
        for suffix in ["summary_confidences", "confidences"]:
            (folder / f"{name}_{suffix}.json").write_text(json.dumps({"name": name}))
    model, summary, confidence = prediction_files(root)
    assert model == root / "ranked_model.cif"
    assert json.loads(summary.read_text())["name"] == "ranked"
    assert json.loads(confidence.read_text())["name"] == "ranked"
    summary.unlink()
    with pytest.raises(FileNotFoundError):
        prediction_files(root)


def test_adaptive_weights_fit_finite_labels_in_raw_metric_units():
    from binderopt.weight_learner import AdaptiveWeightLearner

    labels = {"i_ptm": [0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9, np.nan]}
    learner = AdaptiveWeightLearner({"i_ptm": 0.5})
    weights = learner.learn_weights(labels, ["i_ptm"])
    assert weights == pytest.approx({"i_ptm": 1.0})
    assert learner.weight_history[-1]["n_samples"] == 8
    assert learner.weight_history[-1]["cv_auc"] == pytest.approx(1.0)


def test_replay_rejects_missing_objective_instead_of_zero():
    from binderopt.oracles.replay import ReplayOracle

    oracle = ReplayOracle([{"sequence": "AAAA", "metrics": {"ptm_binder": 0.8}}])
    with pytest.raises(ValueError, match="unavailable"):
        oracle.predict("AAAA")


def test_checkpoint_restores_random_states_and_off_target_cache(tmp_path):
    import pickle
    import random

    optimizer = ProteinOptimizer.__new__(ProteinOptimizer)
    optimizer.output_path = tmp_path
    optimizer.config = SimpleNamespace(checkpoint_frequency=1)
    optimizer.off_target_records = {"AAAA": {"i_ptm": 0.2}}
    optimizer.optimization_log = []
    optimizer.sequence_to_af3_dir = {"AAAA": "prediction"}
    optimizer.surrogate = SimpleNamespace(save_model=lambda path: None)
    random.seed(19)
    np.random.seed(19)
    optimizer._save_checkpoint(["AAAA"], {"i_ptm": [0.8]}, [0], 0)
    expected = (random.random(), np.random.random())
    random.seed(91)
    np.random.seed(91)
    seq, metrics, iterations, next_iteration = optimizer._load_checkpoint(
        str(tmp_path / "checkpoint_iter_0.pkl")
    )
    assert (random.random(), np.random.random()) == expected
    assert seq == ["AAAA"] and metrics == {"i_ptm": [0.8]}
    assert iterations == [0] and next_iteration == 1
    assert optimizer.off_target_records["AAAA"]["i_ptm"] == 0.2
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(pickle.dumps({"iteration": 0}))
    with pytest.raises(ValueError, match="scoring version"):
        optimizer._load_checkpoint(str(legacy))
