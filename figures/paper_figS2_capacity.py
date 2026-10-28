"""Supplementary Figure S2: how many burn features are verified as dictionaries grow, with a fixed and an anchored bar.

a  verified burn features by dictionary expansion m / d (k = 8, 16, 32), with the fixed bar of 0.2 and the anchored bar
b  the fixed bar as a multiple of each dictionary's median contrast, for Tessera (d = 128) and AlphaEarth (d = 64)
c  how much the verified count grows from the smallest to the largest dictionary, for each k and for AlphaEarth

Writes figures/figS2_capacity.png and .pdf, and results/figS2_numbers.json.
"""
import json

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.config import FIG_DIR, RESULTS_DIR
from lib.numbers import FigureNumbers
from lib.style import js

INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
ANCH, FIX = "#c1440e", "#8a8f94"
FS_L, FS_LAB, FS_TICK, FS_ANN = 11.5, 9.0, 8.2, 8.4

D_TES, D_AEF = 128, 64
m8 = json.load(open(RESULTS_DIR / "m8192.json"))
ae = json.load(open(RESULTS_DIR / "alphaearth_capacity.json"))

MS_TES = [512, 1024, 2048, 4096, 8192]
MS_AEF = [256, 512, 1024, 2048, 4096]
xt = [m / D_TES for m in MS_TES]

js.use()
fig = plt.figure(figsize=(9.4, 3.6), dpi=200)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.92, 1.10], wspace=0.42)

def letter(ax, s, x0=-0.17, y0=1.04):
    ax.text(x0, y0, s, transform=ax.transAxes, fontsize=FS_L, fontweight="bold",
            color=INK, va="bottom")

def bare(ax, left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, colors=INK)

# ---- a: the two bars on the same dictionaries -------------------------------
axa = fig.add_subplot(gs[0]); letter(axa, "a")
MARK = {8: "o", 16: "s", 32: "^"}
for k in (8, 16, 32):
    anch = m8["summary"][f"k{k}|anchored"]["core"]
    fix = m8["summary"][f"k{k}|fixed"]["core"]
    axa.plot(xt, anch, "-", marker=MARK[k], ms=5, color=ANCH, lw=1.8,
             mec="white", mew=0.7, zorder=3)
    axa.plot(xt, fix, "--", marker=MARK[k], ms=4.5, color=FIX, lw=1.4,
             mec="white", mew=0.6, zorder=2)
    axa.text(xt[-1] * 1.09, anch[-1] + {8: 0, 16: -2.2, 32: 2.2}[k], f"k={k}",
             fontsize=FS_ANN, color=ANCH, va="center")
axa.set_xscale("log", base=2)
axa.set_xticks(xt); axa.set_xticklabels([f"{int(v)}×" for v in xt])
axa.set_xlim(xt[0] / 1.35, xt[-1] * 2.35)
axa.set_xlabel("dictionary expansion", fontsize=FS_LAB)
axa.set_ylabel("verified burn features", fontsize=FS_LAB)
axa.legend(handles=[Line2D([], [], color=ANCH, lw=1.8, label="anchored bar"),
                    Line2D([], [], color=FIX, lw=1.4, ls="--",
                           label="fixed absolute bar")],
           fontsize=FS_ANN, frameon=False, loc="upper left",
           handlelength=1.5, borderaxespad=0.1)
bare(axa)

# ---- b: fixed bar relative to median contrast -------------------------------
axb = fig.add_subplot(gs[1]); letter(axb, "b", x0=-0.24)
PUB_BAR = 0.2
cells = [c for c in m8["cells"] if c.get("ok")]
for k in (8, 16, 32):
    med = [float(np.median([c["median_abs_C1"] for c in cells
                            if c["m"] == m and c["k"] == k])) for m in MS_TES]
    axb.plot(xt, [PUB_BAR / v for v in med], "-", marker=MARK[k], ms=4.5,
             color=INK2, lw=1.5, mec="white", mew=0.6, alpha=0.85,
             label="Tessera" if k == 8 else None)
aem = [PUB_BAR / v for v in ae["med_contrast"]]
axb.plot([m / D_AEF for m in MS_AEF], aem, "-", marker="D", ms=4.5,
         color="#1f5f8b", lw=1.5, mec="white", mew=0.6, label="AlphaEarth")
axb.set_xscale("log", base=2); axb.set_yscale("log")
axb.set_xticks(xt); axb.set_xticklabels([f"{int(v)}×" for v in xt])
axb.set_xlabel("dictionary expansion", fontsize=FS_LAB)
axb.set_ylabel("fixed bar, in multiples of the\ndictionary's own median contrast",
               fontsize=FS_LAB, linespacing=1.4)
axb.legend(fontsize=FS_ANN, frameon=False, loc="upper left", handlelength=1.4,
           borderaxespad=0.1)
bare(axb)

# ---- c: growth from the smallest to the largest dictionary ------------------
axc = fig.add_subplot(gs[2]); letter(axc, "c", x0=-0.40)
SETTINGS = []
for k in (8, 16, 32):
    a_ = m8["summary"][f"k{k}|anchored"]["core"]
    f_ = m8["summary"][f"k{k}|fixed"]["core"]
    SETTINGS.append((f"burn  k={k}", f_[0], f_[-1], a_[0], a_[-1]))
SETTINGS.append(("AlphaEarth", ae["fixed_core"][0], ae["fixed_core"][-1],
                 ae["anchored_core"][0], ae["anchored_core"][-1]))

def ratio(a, b):
    """Last count divided by first count. A first count of zero is treated as 1, so the result is a lower bound."""
    return (b / a) if a > 0 else (b / 1.0)

yp = np.arange(len(SETTINGS))[::-1]
for yi, (nm, f0, f1, a0, a1) in zip(yp, SETTINGS):
    rf, ra = ratio(f0, f1), ratio(a0, a1)
    axc.plot([rf, ra], [yi, yi], "-", color=FAINT, lw=1.5, zorder=1)
    axc.plot([rf], [yi], "o", ms=6, color=FIX, mec="white", mew=0.8, zorder=3)
    axc.plot([ra], [yi], "o", ms=7, color=ANCH, mec="white", mew=0.8, zorder=4)
axc.axvline(1.0, color=INK, lw=1.0, ls=":", zorder=2)
axc.text(1.0, len(SETTINGS) - 0.32, "no growth", fontsize=FS_ANN, color=INK,
         ha="center", va="bottom")
axc.set_xscale("log")
axc.set_xticks([0.5, 1, 2, 5, 10, 20])
axc.set_xticklabels(["0.5×", "1×", "2×", "5×", "10×", "20×"])
axc.set_yticks(yp); axc.set_yticklabels([s[0] for s in SETTINGS])
axc.set_ylim(-0.6, len(SETTINGS) - 0.05)
axc.set_xlabel("growth in verified features,\nsmallest to largest capacity",
               fontsize=FS_LAB, linespacing=1.4)
bare(axc, left=False); axc.tick_params(left=False)

# ---- numbers the text quotes ------------------------------------------------
NUM = FigureNumbers("figS2")
for k in (8, 16, 32):
    anch = m8["summary"][f"k{k}|anchored"]["core"]
    fix = m8["summary"][f"k{k}|fixed"]["core"]
    NUM.put(f"capacity.burn_k{k}.anchored", anch, "verified core, 4x -> 64x expansion")
    NUM.put(f"capacity.burn_k{k}.fixed", fix)
    NUM.put(f"capacity.at_64x.k{k}", [fix[-1], anch[-1]],
            "[fixed bar, anchored bar] at 64x expansion")
NUM.put("capacity.alphaearth.anchored", ae["anchored_core"])
NUM.put("capacity.alphaearth.fixed", ae["fixed_core"])
NUM.put("capacity.settings_growing", sum(ratio(s[3], s[4]) > 1 for s in SETTINGS),
        f"of {len(SETTINGS)} measured")
# pooled over k, so these are not points on panel b
for key, m in (("capacity.bar_multiple_smallest", MS_TES[0]),
               ("capacity.bar_multiple_largest", MS_TES[-1])):
    NUM.put(key, round(PUB_BAR / float(np.median(
        [c["median_abs_C1"] for c in cells if c["m"] == m])), 1),
        "the fixed 0.2 bar, in multiples of the dictionary's own median contrast"
        if m == MS_TES[0] else
        "the same bar at 64x expansion")
NUM.save()

out = FIG_DIR / "figS2_capacity.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", out)

