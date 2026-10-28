"""Figure 3: every unit on a world map, and each setting's validation rate next to its random-family null.

a  the units on an Equal Earth map. Color shows whether the peak year falls in the setting's own expected window, and
   if not, whether it came early or late. Filled dots are event-dated, open dots detection-dated, squares crop units.
b  each setting's validation rate on held-out units with its 95% interval, next to the median of its random-family null

Writes figures/fig3_scale.png and .pdf.
"""
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.config import FIG_DIR, GT_DIR, RESULTS_DIR
from lib.style import js

INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
LAND, COAST = "#eeece7", "#d8d5cc"
VERC = {"in": "#1b7a4b", "early": "#d98a2b", "late": "#a3341c"}
VERL = {"in": "Dated in window", "early": "Early", "late": "Late"}
CROP = "#6a3d9a"
LAT0, LAT1 = -47, 72


# Equal Earth (Savric, Patterson & Jenny 2018)
A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796


def ee(lon, lat):
    lam, phi = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * A4 * th ** 8 + 7 * A3 * th ** 6 + 3 * A2 * th ** 2 + A1))
    y = A4 * th ** 9 + A3 * th ** 7 + A2 * th ** 3 + A1 * th
    return x, y


df = pd.read_csv(RESULTS_DIR / "paper_events.csv")
df["off"] = (df.peak_year - df.event_year).clip(-1, 3)
_w0 = df.window.map({"Y+1": 1, "{Y,Y+1}": 0})
df["outcome"] = np.where(df.ok == True, "in", np.where(df.off < _w0, "early", "late"))
df.loc[df.kind == "management", "outcome"] = "crop"
land = json.load(open(GT_DIR / "ne_110m_land.geojson"))

_an = json.load(open(RESULTS_DIR / "all_nulls.json"))
NULLS = {k: v["null_median"] for k, v in _an.items() if not k.startswith("_")}
CI = {k: (v["obs_lo"], v["obs_hi"]) for k, v in _an.items() if not k.startswith("_")}

js.use()
# layout in mm from the figure's lower-left corner
MM = 1 / 25.4
FS = dict(letter=8.0, name=7.0, lab=6.5, axlab=6.0, stat=6.5, tick=6.0, leg=6.0)

X0, X1 = ee(-168, 0)[0], ee(180, -40)[0]      # cut off the empty Pacific west of 168W and east of New Zealand
Y0, Y1 = ee(0, LAT0)[1], ee(0, LAT1)[1]
W = 183.0
MAP_W = 165.0
MAP_X = W - MAP_W
MAP_H = MAP_W * (Y1 - Y0) / (X1 - X0)
H = MAP_H + 0.6
K = MAP_W / (X1 - X0)                         # mm per projected unit


fig = plt.figure(figsize=(W * MM, H * MM), dpi=300)
fx, fy = (lambda v: v / W), (lambda v: v / H)


def letter(s, x, y):
    fig.text(fx(x), fy(y), s, fontsize=FS["letter"], fontweight="bold", color=INK, va="baseline", ha="left")


# ---- a: map --------------------------------------------------------------------
ax = fig.add_axes([fx(MAP_X), 0, fx(MAP_W), fy(MAP_H)])
ax.set_facecolor("white")
for f in land["features"]:
    g = f["geometry"]
    for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
        xs, ys = ee(*zip(*poly[0]))
        ax.fill(xs, ys, color=LAND, ec=COAST, lw=0.25, zorder=1)
rng = np.random.default_rng(0)
px, py = ee(df.lon + rng.normal(0, 0.13, len(df)), df.lat + rng.normal(0, 0.13, len(df)))
mg = (df.kind == "management").values
ax.scatter(px[mg], py[mg], s=3, c=CROP, marker="s", lw=0, alpha=0.85, zorder=3)
for ver in ("early", "late", "in"):           # units dated in the window go last, so they sit on top
    for date_type in ("event", "detection"):
        k = (~mg) & (df.outcome == ver).values & (df.date_type == date_type).values
        if not k.any():
            continue
        det = date_type == "detection"
        ax.scatter(px[k], py[k], s=2.6 if ver == "in" else 3.4,
                   facecolors="none" if det else VERC[ver], edgecolors=VERC[ver] if det else "none",
                   linewidths=0.35 if det else 0.0, alpha=0.85, zorder=5 if ver == "in" else 4)
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

TOP_BASE = H - 3.0
letter("a", 0.4, TOP_BASE)
fig.text(fx(3.9), fy(TOP_BASE), f"{len(df):,} units in {df.region.nunique()} settings", fontsize=FS["name"],
         color=INK, va="baseline", ha="left")
handles = [Line2D([], [], marker="o", ls="", mfc=VERC[v], mec="none", ms=3.6, label=VERL[v]) for v in ("in", "early", "late")]
handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, mew=0.6, ms=3.6, label="Detection-dated"),
            Line2D([], [], marker="s", ls="", mfc=CROP, mec="none", ms=3.4, label="Crop")]
fig.legend(handles=handles, loc="upper left", ncol=1, fontsize=FS["leg"], frameon=False, handlelength=0.8,
           handletextpad=0.5, labelspacing=0.35, borderpad=0.0, borderaxespad=0.0,
           bbox_transform=fig.transFigure, bbox_to_anchor=(fx(3.6), fy(TOP_BASE - 2.4)))

# ---- b: validation rate against its null ----------------------------------------
g = (df[df.kind != "management"].groupby("region").agg(n=("ok", "size"), rate=("ok", "mean")).reset_index())
# where a setting has a null, take the rate and n over the same units the null used
for i, reg in g.region.items():
    if reg in _an:
        g.loc[i, "rate"] = _an[reg]["observed"]
        g.loc[i, "n"] = _an[reg]["n"]
g = g.sort_values("rate")
NSET = len(g)
ROW = 2.42                                    # mm per setting row
B_X0, B_X1, B_Y0 = 28.3, 48.5, 6.2            # axis left, right and bottom, fitted to the coastline
N_R = B_X1 + 5.2
B_H = ROW * NSET
axe = fig.add_axes([fx(B_X0), fy(B_Y0), fx(B_X1 - B_X0), fy(B_H)])
axe.patch.set_alpha(0)
N_X = 1.0 + (N_R - B_X1) / (B_X1 - B_X0)
yp = np.arange(NSET)
for yi, (_, r) in zip(yp, g.iterrows()):
    nul = NULLS.get(r.region)
    if nul is not None:
        axe.plot([nul], [yi], "o", ms=3.0, mfc="white", mec=INK2, mew=0.7, zorder=2, clip_on=False)
    if r.region in CI:                            # 95% interval of the observed rate
        axe.plot(CI[r.region], [yi, yi], "-", color="#1b7a4b", lw=1.3, solid_capstyle="butt", zorder=3, clip_on=False)
    axe.plot([r.rate], [yi], "o", ms=3.0, color="#1b7a4b", mec="white", mew=0.4, zorder=4, clip_on=False)
    axe.text(N_X, yi, f"{int(r.n):,}", fontsize=FS["stat"], color=INK, va="center", ha="right")
axe.set_yticks(yp)
SHORT = {"Windthrow": "European storms", "Europe wind/beetle": "Europe wind & beetle", "SE US wildfire": "SE wildfire",
         "SE US prescribed fire": "SE prescribed fire", "SE Australia fire": "Australia fire",
         "Landslides": "Rainfall landslides", "New water": "New permanent water"}
label = lambda r: (lambda s: s[0].upper() + s[1:])(SHORT.get(r, r))
axe.set_yticklabels([label(r) for r in g.region], fontsize=FS["lab"])
axe.set_xlim(0, 1.0)
axe.set_ylim(-0.6, NSET - 0.4)
axe.set_xticks([0, 0.5, 1.0])
axe.set_xticklabels(["0", "0.5", "1"])
axe.set_xticks([0.25, 0.75], minor=True)
axe.set_xlabel("Share of held-out units dated in window", fontsize=FS["axlab"], labelpad=1.5)
axe.tick_params(axis="x", labelsize=FS["tick"], colors=INK, length=1.5, width=0.5, pad=1.2)
axe.tick_params(axis="x", which="minor", colors=FAINT, length=1.0, width=0.4)
axe.tick_params(axis="y", length=0, pad=2, labelcolor=INK)
for s in ("top", "right", "left"):
    axe.spines[s].set_visible(False)
axe.spines["bottom"].set_color(FAINT)
axe.spines["bottom"].set_linewidth(0.5)
axe.grid(axis="x", which="both", color=FAINT, lw=0.3, alpha=0.6)
axe.set_axisbelow(True)
HEAD = B_Y0 + B_H + 1.0
fig.text(fx(N_R), fy(HEAD), "n", fontsize=FS["stat"], color=INK, ha="right", va="baseline")
letter("b", 0.4, HEAD)
fig.legend(handles=[Line2D([], [], marker="o", ls="-", color="#1b7a4b", lw=1.3, mec="white", mew=0.4, ms=3.0,
                           label="Observed, 95% CI"),
                    Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=0.7, ms=3.0, label="Random-family null")],
           fontsize=FS["leg"], loc="lower left", handlelength=1.3, frameon=False, ncol=2, columnspacing=0.6,
           handletextpad=0.4, borderpad=0.0, borderaxespad=0.0,
           bbox_transform=fig.transFigure, bbox_to_anchor=(fx(4.2), fy(HEAD - 0.45)))

out = FIG_DIR / "fig3_scale.png"
fig.savefig(out, dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print("wrote", out, f"({W:.0f} x {H:.1f} mm)")
