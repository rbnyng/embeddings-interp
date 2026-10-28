"""Figure 4: when a disturbance family peaks, relative to the event.

a  event-dated fire settings: share of units that peak in the event year, by ignition month (Wilson 95% intervals,
   leaving out months with fewer than four units)
b  three windstorms: the storm date, the first April-October growing season after it and the most common peak year
   over held-out tiles
c  detection-dated settings: share of units by peak year, counted from the detection year

Writes figures/fig4_timing.png and .pdf, and results/fig4_numbers.json.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from scipy.stats import spearmanr

from lib.config import FIG_DIR, RESULTS_DIR
from lib.numbers import FigureNumbers
from lib.stats import wilson
from lib.style import js

INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
GS_BAND = "#cfe3cf"
df = pd.read_csv(RESULTS_DIR / "paper_events.csv")

# SE Australia is left out of panel a. Its October to March growing season spans the
# new year, so its pattern by month partly comes from the calendar itself. Its numbers
# are still written below.
STYLE = {"Western US fire": ("#2f6f9f", "o", "western US wildfire"),
         "SE US wildfire": ("#7a4fa3", "^", "SE US wildfire"),
         "SE US prescribed fire": ("#3f8f5c", "v", "SE US prescribed fire")}

# drawn 9.3 in wide and printed at 183 mm (scaled by 0.775)
FS_LETTER, FS_AXLAB, FS_TICK, FS_ANNOT, FS_LEG, FS_INBAR = 10.3, 9.0, 8.6, 8.8, 8.4, 8.0
FS_SUB = 7.9

def letter(ax, s, x0=-0.10, y0=1.02):
    ax.text(x0, y0, s, transform=ax.transAxes, fontsize=FS_LETTER,
            fontweight="bold", color=INK, va="bottom")

# ---- a: share peaking in the event year, by ignition month ------------------
def draw_curves(fig, ax):
    for region, (col, mk, lab) in STYLE.items():
        d = df[(df.region == region) & df.ig_month.notna()
               & (df.date_type == "event")].copy()
        d["y0"] = (d.peak_year == d.event_year).astype(int)
        tab = d.groupby(d.ig_month.astype(int)).agg(n=("y0", "size"),
                                                    p0=("y0", "mean"))
        # a share from three or fewer units is 0 or 1 almost by definition
        tab = tab[tab.n >= 4]
        if tab.empty:
            continue
        for mo, r in tab.iterrows():
            lo, hi = wilson(int(round(r.p0 * r.n)), int(r.n))
            ax.plot([mo, mo], [lo, hi], color=col, lw=1.0, alpha=0.35, zorder=1)
        ax.scatter(tab.index, tab.p0, s=22 + 11 * np.sqrt(tab.n), color=col,
                   marker=mk, ec="white", lw=0.9, zorder=3,
                   label=lab.replace(" wildfire", "\nwildfire")
                             .replace(" prescribed fire",
                                      "\nprescribed fire"))
    ax.set_xlabel("Ignition month", fontsize=FS_AXLAB, labelpad=4)
    ax.set_ylabel("Share peaking in the event year", fontsize=FS_AXLAB)
    ax.set_xlim(0.4, 12.6); ax.set_ylim(-0.03, 1.045)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0, 1.0)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                        "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=FS_TICK)
    ax.tick_params(labelsize=FS_TICK, colors=INK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return ax.get_legend_handles_labels()

# ---- b: three storms --------------------------------------------------------
LAB_X = 2016.72
# storm dates are fixed. Peak years and counts come from the held-out tiles in paper_events.csv.
STORM_DATES = [("Xavier", 2017 + 10.5 / 12, "10 Nov 2017"),
               ("Friederike", 2018 + 0.6 / 12, "18 Jan 2018"),
               ("Vaia", 2018 + 9.9 / 12, "28 Oct 2018")]
STORMS = []
for nm, ev, datestr in STORM_DATES:
    peaks = df[(df.region == "Windthrow") & (df.event == nm)].peak_year.astype(int)
    peak = int(peaks.mode()[0])
    STORMS.append((nm, ev, peak, int((peaks == peak).sum()), len(peaks), datestr))

def draw_storms(fig, ax):
    for i, (nm, ev, peak, k, n, datestr) in enumerate(STORMS):
        y = 2 - i
        season = int(ev) if ev % 1 < 3 / 12 else int(ev) + 1      # first April to October season after the storm
        ax.add_patch(Rectangle((season + 3 / 12, y - 0.26), 7 / 12, 0.52,
                               fc=GS_BAND, ec="none", zorder=1))
        ax.plot([ev], [y], marker="v", ms=9, color="#b3261e", mec="white",
                mew=0.8, zorder=4)
        ax.plot([peak + 0.5], [y], marker="*", ms=16, color="#1b7a4b",
                mec="white", mew=0.7, zorder=5)

        ax.text(LAB_X, y + 0.44, nm, fontsize=FS_ANNOT, color=INK,
                va="bottom")
        ax.text(LAB_X, y + 0.27, datestr, fontsize=FS_SUB, color=INK,
                va="bottom")
        ax.text(2020.62, y, f"{k}/{n}", fontsize=FS_ANNOT, color="#1b7a4b",
                va="center", ha="right", fontweight="bold")
    for yr in (2018, 2019, 2020):
        ax.axvline(yr, color=FAINT, lw=0.9, zorder=0)
    ax.set_xlim(LAB_X - 0.06, 2020.66); ax.set_ylim(-1.15, 2.78)
    ax.set_xticks([2017.5, 2018.5, 2019.5])
    ax.set_xticklabels(["2017", "2018", "2019"], fontsize=FS_TICK)
    ax.set_yticks([])
    ax.tick_params(labelsize=FS_TICK, colors=INK, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(FAINT)
    # two stacked legends (two entries, then one) keep the key inside the panel's width
    keys = [Line2D([], [], marker="v", ls="", mfc="#b3261e", mec="white",
                   ms=8, label="storm"),
            Line2D([], [], marker="s", ls="", mfc=GS_BAND, mec="none", ms=9,
                   label="first post-storm growing season"),
            Line2D([], [], marker="*", ls="", mfc="#1b7a4b", mec="white",
                   ms=12, label="observed peak year")]
    kw = dict(fontsize=FS_LEG, frameon=False, loc="upper left",
              handlelength=0.9, columnspacing=0.9, handletextpad=0.45,
              borderaxespad=0.0, bbox_transform=ax.get_yaxis_transform())
    row1 = ax.legend(handles=keys[:2], ncol=2, bbox_to_anchor=(0.0, -0.30),
                     **kw)
    ax.add_artist(row1)
    ax.legend(handles=keys[2:], ncol=1, bbox_to_anchor=(0.0, -0.66), **kw)

# ---- c: detection-dated settings --------------------------------------------
DET = ["Amazon arc clearing", "Europe harvest",
       "Europe wind/beetle", "Europe fire"]
CATS = [-1, 0, 1, 2, 3]
CATC = {0: "#1b7a4b", 1: "#2f6f9f", -1: "#d98a2b", 2: "#c0552b", 3: "#8c2d18"}

def draw_offsets(fig, ax):
    for yi, region in enumerate(DET[::-1]):
        d = df[df.region == region]
        off = (d.peak_year - d.event_year).clip(-1, 3)
        left = 0.0
        for c in CATS:
            frac = float((off == c).mean())
            if frac <= 0:
                continue
            ax.barh(yi, frac, left=left, height=0.66, color=CATC[c], ec="white",
                    lw=0.9)
            if frac > 0.10:
                ax.text(left + frac / 2, yi, f"{frac:.0%}", ha="center",
                        va="center", fontsize=FS_INBAR,
                        color="white" if c in (0, 1, 3) else INK)
            left += frac
        ax.text(-0.015, yi, f"{region}  ({len(d)})", fontsize=FS_ANNOT,
                color=INK, ha="right", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(DET) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("Share of units by peak offset", fontsize=FS_AXLAB)
    ax.tick_params(labelsize=FS_TICK, colors=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(FAINT)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc=CATC[c],
                              mec="none", ms=8, label=l) for c, l in
                       ((-1, "≤Y−1"), (0, "Y+0"), (1, "Y+1"), (2, "Y+2"),
                     (3, "≥Y+3"))],
              fontsize=FS_LEG, frameon=False, loc="lower center", ncol=5,
              handlelength=0.9, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))

# ---- layout -----------------------------------------------------------------
js.use()
fig = plt.figure(figsize=(9.3, 6.05), dpi=200)

gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.60],
                      width_ratios=[0.60, 1.0, 1.04], hspace=0.52, wspace=0.10)

ax_curves = fig.add_subplot(gs[0, 0:2]); letter(ax_curves, "a", x0=-0.105, y0=1.03)
ax_storms = fig.add_subplot(gs[0, 2]); letter(ax_storms, "b", x0=-0.045)
ax_off = fig.add_subplot(gs[1, 1:]); letter(ax_off, "c", x0=-0.285, y0=1.12)
h, l = draw_curves(fig, ax_curves)
ax_curves.legend(h, l, fontsize=FS_LEG, frameon=False, loc="center right",
                 bbox_to_anchor=(1.0, 0.55), handlelength=0.9,
                 labelspacing=0.85, handletextpad=0.5, borderaxespad=0.4)
draw_storms(fig, ax_storms)
draw_offsets(fig, ax_off)

# ---- numbers the caption and text quote -------------------------------------
NUM = FigureNumbers("fig4")
for region in ("Western US fire", "SE US prescribed fire", "SE US wildfire", "SE Australia fire"):
    d = df[(df.region == region) & df.ig_month.notna() & (df.date_type == "event")]
    if d.empty:
        continue
    y0 = d.peak_year == d.event_year
    rho, pv = spearmanr(d.ig_month, y0)
    NUM.put(f"timing.{region}.p_event_year", round(float(y0.mean()), 4),
            "P(peak in the event year), event-dated units only")
    NUM.put(f"timing.{region}.n", int(len(d)))
    NUM.put(f"timing.{region}.rho", round(float(rho), 3),
            "Spearman, ignition month vs peak in the event year")
    NUM.put(f"timing.{region}.rho_p", float(f"{pv:.2g}"))
for region in DET:
    d = df[df.region == region]
    NUM.put(f"detection.{region}.p_y0", round(float((d.peak_year == d.event_year).mean()), 4),
            "share of units peaking at Y+0, detection-dated (panel c)")
for nm, ev, peak, k, n, _ in STORMS:
    ev_year = int(df[(df.region == "Windthrow") & (df.event == nm)].event_year.iloc[0])
    NUM.put(f"storm.{nm.lower()}", {"n": n, "k": k, f"y{peak - ev_year}": round(k / n, 4)},
            "one family, screened on 12 Friederike tiles not counted here, scores all three "
            "storms; no Xavier or Vaia tile was screened")
NUM.save()

out = FIG_DIR / "fig4_timing.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", out)
