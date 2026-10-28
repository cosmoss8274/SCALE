"""AF3 confidence mapping, structural geometry and objective semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from binderopt.losses import ColabDesignLossCalculator
from binderopt.models import RMSDCalculator, MultiHeadSurrogate
from binderopt.objectives import (
    maximise,
    minimise,
    selectivity,
    weighted,
    weighted_scores,
)
from binderopt.oracles.metrics import (
    MetricContractError,
    validate,
    validate_residue_metrics,
)


def write_model(path, translation=None, binder_shift=0):
    target = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], dtype=float)
    binder = np.array([[0, 0, 5], [3, 0, 5], [0, 3, 5], [2, 2, 6]], dtype=float)
    binder[:, 2] += binder_shift
    coords = list(target) + list(binder)
    counts = [4, 5, 6, 4, 8, 3, 5]
    plddts = [95, 95, 95, 60, 70, 80, 90]
    atom_chains, values, lines = [], [], []
    for i, (xyz, n, confidence) in enumerate(zip(coords, counts, plddts)):
        chain, residue = ("A", i + 1) if i < 3 else ("B", i - 2)
        for j in range(n):
            point = xyz + np.array([0, 0, j * 0.01])
            if translation is not None:
                point += translation
            name = "CA" if j == 0 else f"C{j}"
            lines.append(
                f"ATOM {len(lines)+1} C {name} . ALA {chain} 1 {residue} ? "
                f"{point[0]} {point[1]} {point[2]} 1.0 90 {residue} {chain} {name} 1"
            )
            atom_chains.append(chain)
            values.append(confidence)
    headers = [
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "auth_seq_id",
        "auth_asym_id",
        "auth_atom_id",
        "pdbx_PDB_model_num",
    ]
    path.write_text(
        "data_test\nloop_\n"
        + "\n".join("_atom_site." + x for x in headers)
        + "\n"
        + "\n".join(lines)
        + "\n#\n"
    )
    return atom_chains, values


@pytest.fixture
def prediction(tmp_path):
    model = tmp_path / "case_model.cif"
    chains, values = write_model(model)
    # Confidence token order intentionally differs from CIF chain order.
    pae = np.full((7, 7), 4.0)
    pae[:4, 4:] = 2
    pae[4:, :4] = 6
    np.fill_diagonal(pae, 0)
    contacts = np.full((7, 7), 0.8)
    contacts[:4, 4:] = [0.2, 0.5, 0.9]
    full = dict(
        token_chain_ids=["B"] * 4 + ["A"] * 3,
        atom_chain_ids=chains,
        atom_plddts=values,
        pae=pae.tolist(),
        contact_probs=contacts.tolist(),
    )
    confidence = tmp_path / "case_confidences.json"
    confidence.write_text(json.dumps(full))
    (tmp_path / "case_summary_confidences.json").write_text(
        json.dumps(
            dict(
                chain_ids=["A", "B"], iptm=0.7, chain_pair_iptm=[[0.9, 0.8], [0.8, 0.9]]
            )
        )
    )
    return model, confidence, full


def test_atom_plddt_is_averaged_by_residue_and_normalised(prediction):
    model, confidence, _ = prediction
    calc = ColabDesignLossCalculator(model, confidence)
    assert calc.binder_chain == "B" and calc.binder_len == 4
    assert calc.calculate_plddt_loss() == pytest.approx(0.25)
    np.testing.assert_allclose(
        calc.get_residue_level_metrics()["plddt"], [0.6, 0.7, 0.8, 0.9]
    )


def test_pae_uses_token_mapping_and_both_directions(prediction):
    calc = ColabDesignLossCalculator(*prediction[:2])
    intra, inter = calc.calculate_pae_losses()
    assert intra == pytest.approx(4 / 31)
    assert inter == pytest.approx(4 / 31)
    np.testing.assert_allclose(calc.get_residue_level_metrics()["pae_inter"], [2] * 4)
    assert calc.get_iptm() == pytest.approx(0.8)


def test_contact_loss_uses_probability_and_residue_reduction(prediction):
    calc = ColabDesignLossCalculator(*prediction[:2])
    intra, inter = calc.calculate_contact_losses()
    assert intra is None  # No residue pairs are more than five positions apart.
    assert inter == pytest.approx(-np.log(0.9))
    assert calc.calculate_contact_losses(k_inter=2)[1] == pytest.approx(
        -np.log([0.5, 0.9]).mean()
    )
    with pytest.raises(ValueError):
        calc.calculate_contact_losses(cutoff=10)


def test_missing_contacts_and_failed_rmsd_are_not_imputed(prediction):
    model, confidence, full = prediction
    full.pop("contact_probs")
    confidence.write_text(json.dumps(full))
    calc = ColabDesignLossCalculator(model, confidence)
    assert calc.calculate_contact_losses() == (None, None)
    assert "con_inter" not in calc.calculate_all_losses()
    with pytest.raises(ValueError, match="unavailable"):
        calc.calculate_rmsd_loss(
            SimpleNamespace(calculate_minimal_rmsd=lambda *a, **k: np.nan)
        )


def test_invalid_atom_or_token_mapping_is_rejected(prediction):
    model, confidence, full = prediction
    full["atom_plddts"] = full["atom_plddts"][:-1]
    confidence.write_text(json.dumps(full))
    with pytest.raises(ValueError, match="atom_plddts"):
        ColabDesignLossCalculator(model, confidence)


def test_binder_rmsd_keeps_target_alignment(tmp_path):
    references = tmp_path / "references"
    references.mkdir()
    write_model(references / "ref.cif")
    prediction = tmp_path / "prediction.cif"
    write_model(prediction, translation=np.array([10.0, -3.0, 2.0]), binder_shift=5)
    calculator = RMSDCalculator(str(references))
    value = calculator.calculate_minimal_rmsd(
        prediction, residue_indices=[3, 4, 5, 6], alignment_indices=[0, 1, 2]
    )
    assert value == pytest.approx(5.0, abs=1e-5)


def test_objective_signs_and_off_target_choice():
    assert maximise("x")({"x": 2}) == 2
    assert minimise("x")({"x": 2}) == -2
    assert weighted({"confidence": 2, "pae": -0.5})(
        {"confidence": 0.8, "pae": 2}
    ) == pytest.approx(0.6)
    objective = selectivity("on", ["off1", "off2"], penalty=2)
    assert objective(
        {"on": {"i_ptm": 0.9}, "off1": {"i_ptm": 0.1}, "off2": {"i_ptm": 0.7}}
    ) == pytest.approx(-0.5)
    energy = selectivity(
        "on", ["off"], penalty=1, metric="energy", higher_is_better=False
    )
    assert energy({"on": {"energy": -10}, "off": {"energy": -3}}) == 7
    with pytest.raises(ValueError):
        maximise("x")({"x": np.nan})


def test_weighted_batch_does_not_reward_missing_losses():
    scores = weighted_scores(
        {"i_ptm": [0.8, 0.9], "pae": [2.0, np.nan]}, {"i_ptm": 1, "pae": -1}
    )
    assert scores[0] == pytest.approx(-0.6) and np.isneginf(scores[1])


def test_metric_ranges_and_scales():
    with pytest.raises(MetricContractError):
        validate(
            dict(
                i_ptm=1.1,
                ptm_binder=0.8,
                min_pae_interaction=2,
                oracle="x",
                oracle_version="1",
            )
        )
    with pytest.raises(MetricContractError):
        validate_residue_metrics({"plddt": [-0.1]})


def test_surrogate_can_add_a_previously_unlabelled_head():
    model = MultiHeadSurrogate.__new__(MultiHeadSurrogate)
    model.metrics = []
    model.heads = {}
    model.metric_weights = {"i_ptm": 1, "nll": -1}
    model.rf_params = {"n_estimators": 3, "random_state": 0}
    model.training_history = []
    embeddings = np.arange(12).reshape(4, 3)
    model._train_random_forest(
        embeddings, {"i_ptm": [0.2, 0.4, np.nan, 0.8], "nll": [np.nan] * 4}
    )
    assert "nll" not in model.metrics
    model._train_random_forest(
        embeddings, {"i_ptm": [0.2, 0.4, 0.6, 0.8], "nll": [1.0, np.nan, 2.0, 3.0]}
    )
    assert "nll" in model.metrics
