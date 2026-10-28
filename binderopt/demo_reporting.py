"""Measured score, novelty and batch diversity for the interactive demo."""

import numpy as np

from .diversity import mismatches, nearest, select_diverse


def annotate_distances(history, pool):
    seeds = {h["sequence_id"]: h["sequence"] for h in history if h["round"] == 0}
    for h in history:
        distance = nearest(h["sequence"], pool)
        h["mutations_from_pool"] = distance
        h["identity_to_pool"] = 1 - distance / len(h["sequence"])
        founder = h.get("founder_id", h["sequence_id"])
        if founder in seeds:
            h["mutations_from_founder"] = mismatches(h["sequence"], seeds[founder])


def diverse_portfolio(history, policy, n=5):
    """Choose measured candidates under the same batch diversity constraints.

    Final ranking must not undo the diversity enforced during acquisition.
    If greedy selection across rounds blocks a full portfolio, an already
    feasible measured batch remains available, without lowering any threshold.
    """
    candidates = [
        h
        for h in history
        if h["round"] > 0
        and h["mutations_from_pool"] >= policy.counts(len(h["sequence"]))[0]
    ]
    if len(candidates) < n:
        raise ValueError("Not enough measured novel designs for a portfolio")
    try:
        chosen = select_diverse(candidates, [h["i_ptm"] for h in candidates], n, policy)
    except RuntimeError:
        groups = [
            [h for h in candidates if h["round"] == r]
            for r in sorted({h["round"] for h in candidates})
        ]
        groups = sorted(
            (g for g in groups if len(g) >= n),
            key=lambda g: -float(np.mean([h["i_ptm"] for h in g])),
        )
        for group in groups:
            try:
                chosen = select_diverse(group, [h["i_ptm"] for h in group], n, policy)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("No full measured portfolio satisfies the constraints")
    originals = {h["sequence_id"]: h for h in candidates}
    return sorted(
        (originals[h["sequence_id"]] for h in chosen), key=lambda h: -h["i_ptm"]
    )


def plot_trajectory(history, objective, policy, output_path):
    """Compact score and metric trajectories from every measured design.

    ``policy`` remains in the signature for existing notebook callers. Sequence
    distances and portfolio selection are reported separately from this figure.
    """
    import matplotlib.pyplot as plt
    from .plot_style import apply_figure_style

    apply_figure_style(sizes=(9, 7.5, 8))
    rounds = sorted({h["round"] for h in history})
    groups = [[h for h in history if h["round"] == r] for r in rounds]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    blue, grey, amber = "#1F5FBF", "#9aa0a6", "#E08A1E"

    ax = axes[0]
    ax.scatter(
        [h["round"] for h in history],
        [objective(h) for h in history],
        s=22,
        color=grey,
        edgecolor="white",
        linewidths=0.3,
        zorder=2,
        label="Each design",
    )
    best = [max(objective(h) for h in history if h["round"] <= r) for r in rounds]
    ax.plot(
        rounds, best, "-o", color=blue, lw=1.3, ms=3.5, zorder=3, label="Best so far"
    )
    ax.set(xlabel="Round", ylabel="Objective", xticks=rounds)
    ax.legend(loc="lower right")

    ax = axes[1]
    for key, colour, label in (
        ("i_ptm", blue, "Interface pTM"),
        ("min_pae_interaction", amber, "Interface PAE (Å)"),
    ):
        values = [
            np.asarray([h.get(key, float("nan")) for h in g], dtype=float)
            for g in groups
        ]
        finite = [v[np.isfinite(v)] for v in values]
        if not any(v.size for v in finite):
            continue
        medians = [float(np.median(v)) if v.size else float("nan") for v in finite]
        target_ax = ax if key == "i_ptm" else ax.twinx()
        if key != "i_ptm":
            target_ax.spines["right"].set_visible(True)
        target_ax.plot(rounds, medians, "-o", color=colour, lw=1.2, ms=3.2)
        target_ax.set_ylabel(label, color=colour)
        target_ax.tick_params(axis="y", colors=colour)
    ax.set(xlabel="Round", xticks=rounds)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
