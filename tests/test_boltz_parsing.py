"""Validate the Boltz-2 adapter's output parsing without running a prediction."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderopt.oracles.boltz import Boltz2Oracle

N_RECEPTOR, N_BINDER = 12, 8
NAME = "case"


def build_tree(root: Path, write_pae: bool) -> Path:
    pred = root / "predictions" / NAME
    pred.mkdir(parents=True, exist_ok=True)
    (pred / ("confidence_%s_model_0.json" % NAME)).write_text(
        json.dumps(
            {
                "confidence_score": 0.81,
                "ptm": 0.77,
                "iptm": 0.64,
                "complex_plddt": 0.88,
                "complex_iplddt": 0.85,
                "chains_ptm": {"0": 0.90, "1": 0.55},
                "pair_chains_iptm": {
                    "0": {"0": 0.90, "1": 0.71},
                    "1": {"0": 0.71, "1": 0.55},
                },
            }
        )
    )
    (pred / ("%s_model_0.cif" % NAME)).write_text("data_case\n")
    if write_pae:
        pae = np.full((N_RECEPTOR + N_BINDER,) * 2, 20.0)
        pae[3, N_RECEPTOR + 2] = 1.37  # the one tight cross-chain pair
        pae[N_RECEPTOR + 2, 3] = 4.20
        np.savez(pred / ("pae_%s_model_0.npz" % NAME), pae=pae)
    return root


def make_oracle(tmp):
    oracle = Boltz2Oracle.__new__(Boltz2Oracle)  # bypass the PATH check for boltz
    oracle.target_sequence = "A" * N_RECEPTOR
    return oracle


def main() -> int:
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        root = build_tree(Path(tmp) / "with_pae", write_pae=True)
        m = make_oracle(tmp)._collect(root, NAME)

        checks = [
            ("i_ptm from pair_chains_iptm[0][1]", m["i_ptm"], 0.71),
            ("ptm_binder from chains_ptm['1']", m["ptm_binder"], 0.55),
            ("min interface PAE from the npz", m["min_pae_interaction"], 1.37),
            ("complex_iptm passthrough", m["complex_iptm"], 0.64),
            ("complex_plddt passthrough", m["complex_plddt"], 0.88),
        ]
        for label, got, want in checks:
            ok = abs(got - want) < 1e-9
            print(
                "%-38s %-8s got %.4f want %.4f"
                % (label, "ok" if ok else "FAIL", got, want)
            )
            if not ok:
                failures.append(label)
        if m["oracle"] != "boltz2":
            failures.append("provenance stamp")
        print(
            "%-38s %-8s %s"
            % (
                "provenance stamp",
                "ok" if m["oracle"] == "boltz2" else "FAIL",
                m["oracle"],
            )
        )

        # Without --write_pae the metric must be NaN, never a substituted value.
        root2 = build_tree(Path(tmp) / "no_pae", write_pae=False)
        m2 = make_oracle(tmp)._collect(root2, NAME)
        ok = np.isnan(m2["min_pae_interaction"])
        print(
            "%-38s %-8s %s"
            % (
                "no pae npz -> NaN, not a stand-in",
                "ok" if ok else "FAIL",
                m2["min_pae_interaction"],
            )
        )
        if not ok:
            failures.append("NaN on missing PAE")

    print(
        "\n%s"
        % ("ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures))
    )
    return 1 if failures else 0


def test_boltz_parsing():
    """pytest entry point. ``main`` returns non-zero on failure and prints the detail."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
