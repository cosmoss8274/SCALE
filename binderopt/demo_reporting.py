"""Measured score, novelty and batch diversity for the interactive demo."""
from collections import Counter
from itertools import combinations

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
    candidates = [h for h in history if h["round"] > 0 and
                  h["mutations_from_pool"] >= policy.counts(len(h["sequence"]))[0]]
    if len(candidates) < n:
        raise ValueError("Not enough measured novel designs for a portfolio")
    try:
        chosen = select_diverse(candidates, [h["i_ptm"] for h in candidates], n, policy)
    except RuntimeError:
        groups = [[h for h in candidates if h["round"] == r]
                  for r in sorted({h["round"] for h in candidates})]
        groups = sorted((g for g in groups if len(g) >= n),
                        key=lambda g: -float(np.mean([h["i_ptm"] for h in g])))
        for group in groups:
            try:
                chosen = select_diverse(group, [h["i_ptm"] for h in group], n, policy)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("No full measured portfolio satisfies the constraints")
    originals = {h["sequence_id"]: h for h in candidates}
    return sorted((originals[h["sequence_id"]] for h in chosen),
                  key=lambda h: -h["i_ptm"])


def plot_trajectory(history, objective, policy, output_path):
    import matplotlib.pyplot as plt
    from .plot_style import apply_figure_style
    apply_figure_style(sizes=(10, 9, 9))
    rounds = sorted({h["round"] for h in history})
    groups = [[h for h in history if h["round"] == r] for r in rounds]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    blue, grey, amber = "#1F5FBF", "#9aa0a6", "#D88718"

    ax = axes[0, 0]
    for r, group in zip(rounds, groups):
        ax.scatter([r] * len(group), [objective(h) for h in group], s=22,
                   color=grey, alpha=.8, label="Measured design" if r == 0 else None)
    ax.plot(rounds, [np.median([objective(h) for h in g]) for g in groups],
            "-o", color=blue, label="Round median")
    ax.plot(rounds, [max(objective(h) for h in history if h["round"] <= r) for r in rounds],
            "--", color=amber, label="Best so far")
    feasible_best = []
    for r in rounds:
        feasible = [h for h in history if 0 < h["round"] <= r and
                    h["mutations_from_pool"] >= policy.counts(len(h["sequence"]))[0]]
        feasible_best.append(max((objective(h) for h in feasible), default=float("nan")))
    ax.plot(rounds, feasible_best, "-D", ms=4, color="#168B83",
            label="Best novel design so far")
    ax.set(xlabel="Round", ylabel="Objective (higher is better)", xticks=rounds,
           title="a  Measured scores")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.24), ncol=2)

    ax = axes[0, 1]
    for r, group in zip(rounds, groups):
        if r:
            ax.scatter([h["mutations_from_pool"] for h in group],
                       [objective(h) for h in group], s=35, label=f"Round {r}")
    required = policy.counts(len(history[0]["sequence"]))[0]
    ax.axvline(required, color=grey, ls="--", lw=1)
    ax.set(xlabel="Mismatches to nearest starting-pool sequence",
           ylabel="Objective (higher is better)", title="b  Score and novelty")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.24), ncol=3)

    ax = axes[1, 0]
    mins, medians, new_rounds = [], [], []
    for r, group in zip(rounds, groups):
        if r and len(group) > 1:
            distances = [mismatches(a["sequence"], b["sequence"])
                         for a, b in combinations(group, 2)]
            new_rounds.append(r)
            mins.append(min(distances))
            medians.append(float(np.median(distances)))
    ax.plot(new_rounds, mins, "-o", color=blue, label="Minimum")
    ax.plot(new_rounds, medians, "-s", color=amber, label="Median")
    ax.axhline(policy.counts(len(history[0]["sequence"]))[1],
               color=grey, ls="--", label="Required minimum")
    ax.set(xlabel="Round", ylabel="Pairwise mismatches within batch", xticks=new_rounds,
           title="c  Batch sequence diversity")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.24), ncol=2)

    ax = axes[1, 1]
    founders = [h["sequence_id"] for h in history if h["round"] == 0]
    counts = [Counter(h["founder_id"] for h in group) for r, group in zip(rounds, groups) if r]
    new_rounds = [r for r in rounds if r]
    bottom = np.zeros(len(new_rounds))
    colours = plt.get_cmap("tab10")
    for i, founder in enumerate(founders):
        values = np.array([c[founder] for c in counts])
        if values.sum():
            ax.bar(new_rounds, values, bottom=bottom, color=colours(i % 10),
                   label=f"Seed {i + 1}", width=.65)
            bottom += values
    ax.set(xlabel="Round", ylabel="Designs per founding seed", xticks=new_rounds,
           title="d  Contributions from multiple starting sequences")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    for ax in axes.flat:
        ax.margins(.08)
    counts_text = ", ".join(f"r{r}: {len(g)}" for r, g in zip(rounds, groups))
    fig.suptitle(f"One structure prediction per design ({counts_text})", fontsize=10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
