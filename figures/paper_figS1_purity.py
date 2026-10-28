"""Supplementary Figure S1: the Figure 2 corn and soy features next to one fixed raw-dimension rule, tile by tile.

One point per Corn Belt tile: the F1 of the feature (active or not) for its crop (y), against the F1 of the best single
raw dimension, with its end and threshold fixed on the Figure 2 tile (x).

Writes figures/figS1_purity.png and .pdf, and results/figS1_numbers.json.
"""
import json

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.config import FIG_DIR, RESULTS_DIR
from lib.numbers import FigureNumbers
from lib.style import js

INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
COL = {"corn": "#d68910", "soy": "#2e7d32"}
FEAT = {"corn": 904, "soy": 114}

fx = json.load(open(RESULTS_DIR / "feature_purity_fixed.json"))
best = json.load(open(RESULTS_DIR / "feature_purity_best.json"))
pur = json.load(open(RESULTS_DIR / "feature_purity.json"))

js.use()
LO = 0.7            # bottom of the y axis. The x axis reaches down to the lowest raw F1.
fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=200)
ax.plot([LO, 1], [LO, 1], "-", color=FAINT, lw=1.0, zorder=1)
off = {}
xmin = 1.0
for c in ("corn", "soy"):
    r = np.array([row[c]["raw_f1"] for row in fx["rows"]])
    s_ = np.array([row[c]["sae_f1"] for row in fx["rows"]])
    ax.scatter(r, s_, s=18, color=COL[c], alpha=0.8, lw=0, zorder=3)
    off[c] = int((r < LO).sum())
    xmin = min(xmin, r.min())
    assert s_.min() >= LO, "a feature F1 falls below the y floor"
ax.set_xlim(np.floor(xmin * 10) / 10, 1); ax.set_ylim(LO, 1)
ax.set_xlabel("raw dimension, one threshold for all tiles (F1)", fontsize=8.5)
ax.set_ylabel("feature, active or inactive (F1)", fontsize=8.5)
ax.tick_params(labelsize=7.8)
ax.grid(False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[c], ms=5,
                          label=f"{c} (feature {FEAT[c]})") for c in ("corn", "soy")],
          fontsize=7.8, frameon=False, loc="lower left", handletextpad=0.3)

out = FIG_DIR / "figS1_purity.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.04, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("wrote", out)

# ---- numbers the SI quotes ------------------------------------------------------------
NUM = FigureNumbers("figS1")
rows = best["rows"]
for c in ("corn", "soy"):
    NUM.put(f"purity.{c}.same_feature_tiles", int(sum(r[c]["sae_feature"] == FEAT[c] for r in rows)),
            f"tiles on which feature {FEAT[c]} is the best single {c} feature")
    NUM.put(f"purity.{c}.n_tiles", len(rows))
    NUM.put(f"purity.{c}.sae_f1", fx["summary"][c]["sae_f1_median"], "median over tiles")
    NUM.put(f"purity.{c}.raw_tuned_f1", best["summary"][c]["raw_f1_median"],
            "best raw dimension, end and threshold tuned on each tile")
    NUM.put(f"purity.{c}.raw_fixed_f1", fx["summary"][c]["raw_f1_median"],
            "best raw dimension, one rule chosen on the Fig. 2 tile")
    NUM.put(f"purity.{c}.sae_better_tiles",
            int(round(fx["summary"][c]["sae_better_share"] * fx["summary"][c]["n_tiles"])))
    NUM.put(f"purity.{c}.raw_fixed_p10", fx["summary"][c]["raw_f1_p10"], "10th percentile over tiles")
    NUM.put(f"purity.{c}.sae_p10", fx["summary"][c]["sae_f1_p10"], "10th percentile over tiles")
for c in ("corn", "soy"):
    NUM.put(f"purity.{c}.raw_fixed_below", off[c], f"tiles where the fixed raw rule's F1 is below {LO}")
NUM.put("purity.axis_floor", LO)
NUM.put("purity.split_ends", pur["summary"]["raw_split_ends"],
        "share of raw dimensions whose two 2% tails have different dominant classes, crop atlas")
NUM.save()
