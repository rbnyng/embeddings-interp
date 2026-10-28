"""Figure 6: Tessera and AlphaEarth Foundations embeddings of the same events, side by side.

a  raw embeddings, with no dictionary: SE US wildfire signal over pre-event noise, and Iowa corn-soy separation
b  signal left two years after the peak in three fire settings, on the same events and masks
c  each model's family peak year minus the fire year, on the western US fires that both models screened

Writes figures/fig6_crossmodel.png and .pdf, and results/fig6_numbers.json.
"""
import numpy as np
import json

import matplotlib.pyplot as plt
from collections import Counter

from lib.config import FIG_DIR, RESULTS_DIR
from lib.numbers import FigureNumbers
from lib.style import js

INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
TES, AEF = "#1f5f8b", "#c1440e"
GAP = "#b9b5aa"
# drawn 9.8 in wide and printed at 183 mm (scaled by 0.735)
FS_L, FS_TTL, FS_AX = 10.9, 9.2, 8.8
FS_VAL = FS_AX
MS = 8.5

pc = json.load(open(RESULTS_DIR / "raw_retention.json"))["classes"]
ck = json.load(open(RESULTS_DIR / "crop_aef.json"))

tes = {r["event_id"]: r for r in map(json.loads, open(RESULTS_DIR / "western_us_fire.jsonl"))
       if r.get("ok")}
aef = {r["event_id"]: r for r in map(json.loads, open(RESULTS_DIR / "western_us_fire_aef.jsonl"))
       if r.get("ok")}
both = sorted(set(tes) & set(aef))
off = {"tessera": [int(tes[e]["peak"] - tes[e]["year"]) for e in both],
       "aef": [int(aef[e]["peak"] - tes[e]["year"]) for e in both],
       "n": len(both)}

js.use()
fig = plt.figure(figsize=(9.8, 4.8), dpi=200)
gs = fig.add_gridspec(2, 2, width_ratios=[1.00, 1.10], height_ratios=[1.0, 0.90],
                      wspace=0.46, hspace=0.62)

def letter(ax, s, x0=-0.16, y0=1.03):
    ax.text(x0, y0, s, transform=ax.transAxes, fontsize=FS_L, fontweight="bold",
            color=INK, va="bottom")

def bare(ax, left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=FS_AX, colors=INK)

def gap(ax, x0, x1, y, c0=TES, c1=AEF):
    """Draw one Tessera and AlphaEarth pair on one row, joined by a line along the value axis."""
    ax.plot([x0, x1], [y, y], "-", color=GAP, lw=1.6, zorder=1,
            solid_capstyle="round")
    ax.plot([x0], [y], "o", ms=MS, color=c0, mec="white", mew=1.0, zorder=3)
    ax.plot([x1], [y], "o", ms=MS, color=c1, mec="white", mew=1.0, zorder=3)

# ---- a: raw-embedding comparison --------------------------------------------
ga = gs[:, 0].subgridspec(2, 1, hspace=1.05)
STRIPS = [("Small fast-recovery fires\nSE US wildfire",
           pc["SE US wildfire | Tessera"]["snr_med"], pc["SE US wildfire | AlphaEarth"]["snr_med"],
           "Raw signal / pre-event noise", 1.0, "noise floor"),
          ("Crop identity\nIowa corn vs soy",
           ck["k4"]["tessera"], ck["k4"]["aef"],
           "Corn–soy separation (normalized)", None, None)]
for si, (ttl, tv, av, xlab, ref, reflab) in enumerate(STRIPS):
    ax = fig.add_subplot(ga[si])
    if si == 0:
        letter(ax, "a", x0=-0.13, y0=1.34)
    if ref is not None:
        ax.axvline(ref, color=FAINT, lw=1.0, ls=":", zorder=0)
    gap(ax, tv, av, 0.0)
    for v, cl, nm in ((tv, TES, "Tessera"), (av, AEF, "AlphaEarth")):
        ax.text(v, 0.26, f"{v:.2f}", ha="center", va="bottom", fontsize=FS_VAL,
                color=cl)
        ax.text(v, -0.30, nm, ha="center", va="top", fontsize=FS_AX, color=cl)
    lo, hi = min(tv, av), max(tv, av)
    span = hi - lo
    ax.set_xlim(min(lo, ref if ref else lo) - 0.30 * span - 0.10,
                hi + 0.22 * span + 0.08)
    if ref is not None:
        ax.text(ref, 1.02, reflab, ha="center", va="bottom", fontsize=FS_AX,
                color=INK)
    ax.set_ylim(-0.62, 1.42)
    ax.set_yticks([])
    ax.set_xlabel(xlab, fontsize=FS_AX, labelpad=4)
    ax.text(0.0, 1.12, ttl, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS_TTL, color=INK, linespacing=1.35)
    bare(ax, left=False)

# ---- b: signal left two years later -----------------------------------------
axb = fig.add_subplot(gs[0, 1]); letter(axb, "b", x0=-0.36, y0=1.16)
PAIRS = [("CONUS megafires", "Western US fire | Tessera", "Western US fire | AlphaEarth"),
         ("SE US wildfires", "SE US wildfire | Tessera", "SE US wildfire | AlphaEarth"),
         ("SE US prescribed", "SE US prescribed fire | Tessera", "SE US prescribed fire | AlphaEarth")]
for i, (nm, kt, ka) in enumerate(PAIRS):
    y = len(PAIRS) - 1 - i
    t, a = pc[kt]["P_med"], pc[ka]["P_med"]
    gap(axb, t, a, y)
    axb.text(t - 0.028, y, f"{t:.2f}", ha="right", va="center", fontsize=FS_VAL,
             color=TES)
    axb.text(a + 0.028, y, f"{a:.2f}", ha="left", va="center", fontsize=FS_VAL,
             color=AEF)
    if i == 0:
        axb.text(t, y + 0.30, "Tessera", ha="center", va="bottom", fontsize=FS_AX,
                 color=TES)
        axb.text(a, y + 0.30, "AlphaEarth", ha="center", va="bottom",
                 fontsize=FS_AX, color=AEF)
axb.set_xlim(0.0, 0.85)
axb.set_ylim(-0.55, len(PAIRS) - 0.15)
axb.set_yticks(range(len(PAIRS)))
axb.set_yticklabels([nm for nm, _, _ in PAIRS][::-1], fontsize=FS_AX)
axb.set_xlabel("Signal remaining two years on\n(identical events and masks)",
               fontsize=FS_AX, linespacing=1.4)
bare(axb, left=False)
axb.grid(False)
axb.tick_params(axis="y", length=0)

# ---- c: peak year after the fire, on fires both models share ----------------
axc = fig.add_subplot(gs[1, 1]); letter(axc, "c", x0=-0.36, y0=1.06)
OFFS = list(range(-2, 6))
clip = lambda o: max(-2, min(5, o))
tof = Counter(clip(o) for o in off["tessera"])
aof = Counter(clip(o) for o in off["aef"])
axc.axvspan(0.5, 1.5, color="#dce8dc", zorder=0)
# shaded band: the expected window, the year after the fire
W = 0.38
for i, (c_, col, nm) in enumerate(((tof, TES, "Tessera"), (aof, AEF, "AlphaEarth"))):
    axc.bar([o + (i - 0.5) * W for o in OFFS], [c_.get(o, 0) for o in OFFS],
            width=W, color=col, alpha=0.9, label=nm, zorder=2)
axc.set_xticks(OFFS)
axc.set_xticklabels(["$\\leq-$2", "$-$1", "0", "+1", "+2", "+3", "+4",
                     "$\\geq$+5"], fontsize=FS_AX - 1.0)
axc.set_xlabel("Family peak, years after fire", fontsize=FS_AX)
axc.set_ylabel(f"Fires (of {off['n']})", fontsize=FS_AX)
axc.legend(fontsize=FS_AX - 0.6, frameon=False, loc="upper right",
           handlelength=1.0, handletextpad=0.5)
for s_ in ("top", "right"):
    axc.spines[s_].set_visible(False)
axc.tick_params(labelsize=FS_AX - 0.8, colors=INK)

# ---- numbers the caption and text quote -------------------------------------
NUM = FigureNumbers("fig6")
for k in ("SE US wildfire", "Western US fire", "SE US prescribed fire"):
    for m, tag in (("Tessera", "tessera"), ("AlphaEarth", "aef")):
        if f"{k} | {m}" in pc:
            NUM.put(f"fig6.persist.{tag}.{k}", round(pc[f"{k} | {m}"]["P_med"], 4),
                    "fraction remaining two years on")
            NUM.put(f"fig6.snr.{tag}.{k}", round(pc[f"{k} | {m}"]["snr_med"], 4),
                    "raw signal / pre-event noise")
NUM.put("fig6.crop.tessera", round(ck["k4"]["tessera"], 4))
NUM.put("fig6.crop.aef", round(ck["k4"]["aef"], 4))
NUM.put("fig6.offsets.n", off["n"])
NUM.put("fig6.offsets.tessera_in_window", sum(o == 1 for o in off["tessera"]))
NUM.put("fig6.offsets.aef_in_window", sum(o == 1 for o in off["aef"]))
# of the fires Tessera dates in the window and AlphaEarth misses, the share whose AlphaEarth peak is two or more years
# later
t_only = [e for e in both if tes[e]["dated"] and not aef[e]["dated"]]
NUM.put("fig6.offsets.aef_late_share",
        round(float(np.mean([aef[e]["peak"] - tes[e]["year"] >= 2 for e in t_only])), 4),
        f"share of the {len(t_only)} Tessera-dated, AlphaEarth-missed fires whose "
        "AlphaEarth peak lands 2+ years after the fire. Agreement on jointly dated "
        "fires is 1.0 by construction with a one-year window.")
NUM.save()

out = FIG_DIR / "fig6_crossmodel.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", out)
