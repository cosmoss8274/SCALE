"""Matplotlib styling for compact trajectory plots."""


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    """Configure fonts, axes and export resolution."""
    import matplotlib as mpl

    if frame not in {"open", "boxed", "none"}:
        raise ValueError("frame must be open, boxed or none")
    base, secondary, tick = sizes
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": base,
            "legend.fontsize": secondary,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "axes.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": frame == "boxed",
            "axes.spines.right": frame == "boxed",
            "axes.spines.left": frame != "none",
            "axes.spines.bottom": frame != "none",
            "axes.grid": bool(grid),
            "legend.frameon": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 1.2,
        }
    )
    if font:
        mpl.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
