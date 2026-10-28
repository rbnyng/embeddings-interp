"""Matplotlib style for the paper figures: Nature's figure defaults on top of a plain serif base style.

`js` provides use, subplots, clean, style_legend, panel_labels and PALETTE. Set PAPER_STYLE=classic to use only the
serif base.
"""
import os
import string

import matplotlib.pyplot as plt
import numpy as np


class _Base:
    PALETTE = ["#4c72b0", "#dd8452", "#2f4550", "#ccb974", "#8fa1b3",
               "#64b5cd", "#8172b3"]

    @staticmethod
    def use():
        plt.rcParams.update({"font.family": "serif", "figure.dpi": 150,
                             "axes.grid": True, "grid.alpha": 0.3,
                             "axes.grid.axis": "y", "legend.frameon": False})

    @classmethod
    def clean(cls, ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    @classmethod
    def subplots(cls, *a, apply_use=False, **k):
        if apply_use:
            cls.use()
        k.setdefault("dpi", 150)
        fig, axes = plt.subplots(*a, **k)
        for ax in np.atleast_1d(axes).ravel():
            cls.clean(ax)
        return fig, axes

    @staticmethod
    def style_legend(ax, title=None, **k):
        ax.legend(title=title, frameon=False, **k)

    @staticmethod
    def panel_labels(axes, labels=None, x=-0.06, y=1.06):
        axs = np.atleast_1d(axes).ravel()
        labels = labels or list(string.ascii_lowercase)
        for ax, lab in zip(axs, labels):
            ax.text(x, y, lab, transform=ax.transAxes,
                    fontsize=13, fontweight="bold")


js = _Base()

# ---- Nature figure guide: sans-serif text of 5 to 7 pt, bold 8 pt panel letters, widths of 89, 120-136 or 183 mm,
# height at most 170 mm, and lines at least 0.25 pt. The sizes below assume a figure drawn at its print width.
NATURE_RC = {
    "font.family": "sans-serif",
    # Arial if present, otherwise fonts with the same metrics (Liberation Sans for Arial, Nimbus Sans for Helvetica)
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 7, "figure.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.title_fontsize": 6,
    "axes.linewidth": 0.5, "lines.linewidth": 0.75, "lines.markersize": 3, "patch.linewidth": 0.5,
    "grid.linewidth": 0.4, "hatch.linewidth": 0.35,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0, "axes.titlepad": 3.0,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # editable text, no Type 3 fonts
    "savefig.dpi": 450, "figure.dpi": 150,
}


def _nature_use(base_use):
    def use(*a, **k):
        base_use(*a, **k)
        plt.rcParams.update(NATURE_RC)
    return use


if os.environ.get("PAPER_STYLE", "nature") != "classic":
    class _Nature:
        """js with use() switched to the Nature defaults. Everything else comes from the base style."""
        def __init__(self, base):
            self._base = base
            self.use = _nature_use(base.use)

        def __getattr__(self, name):
            return getattr(self._base, name)
    js = _Nature(js)

# Figures choose their own colors for things like burns and crops. PALETTE is only a generic default.
