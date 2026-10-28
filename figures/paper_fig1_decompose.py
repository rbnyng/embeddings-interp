"""Figure 1: one Dixie Fire pixel's embedding split into dictionary features, and when the burn family peaks.

a  Sentinel-2 SWIR false color just after the fire, then four 2022 scenes going into the encoder
b  the 2022 Tessera embedding as its first three principal components, with a 10 m inset of the marked pixel
c  the dictionary applied to a stack of pixels: active features drawn as maps, with their burned minus unburned courses
d  dNBR, the burn-severity reference
e  the burn family's activation
f  burned minus unburned family activation by years since the fire, for Dixie and three held-out fires

Writes figures/fig1_decompose.png and .pdf, and results/fig1_numbers.json. The first run also caches the Sentinel-2
composites in data/groundtruth/.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import ConnectionPatch, Polygon, Rectangle
from scipy.ndimage import distance_transform_edt

from lib.config import (DIXIE, DNBR_BURN_THRESHOLD, FIG_DIR, FIRES, BURN_FAMILY, GT_DIR,
                        RESULTS_DIR, dnbr_path)
from lib.numbers import FigureNumbers
from lib.groundtruth import fetch_s2_rgb
from lib.tiles import load_tile
from lib.style import js

S = 2
TILE, YEAR = FIRES[DIXIE]["tile"], 2022
FAM = BURN_FAMILY
R0, C0, CS = 0, 0, 256              # crop origin and size, in stride-2 pixels (5.1 km)
INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
BURN, GREYL = "#c1440e", "#a3a3a3"
PX_M = 10.0

# ---- data -----------------------------------------------------------------------
x = load_tile(TILE, YEAR, stride=S)
h, w, _ = x.shape
F = np.load(RESULTS_DIR / "fig1_dixie_feats.npy", mmap_mode="r")[:h, :w]
v = np.asarray(F[:, :, FAM], dtype=np.float32).mean(-1)
dn = np.load(dnbr_path(DIXIE))[::S, ::S][:h, :w]
crop = lambda a: a[R0:R0 + CS, C0:C0 + CS]
cropfull = lambda a: a[R0 * S:(R0 + CS) * S, C0 * S:(C0 + CS) * S]

def stretch(a, lo=1, hi=99):
    a0, a1 = np.nanpercentile(a, lo), np.nanpercentile(a, hi)
    return ((a - a0) / (a1 - a0)).clip(0, 1)

def fill_nearest(a):
    bad = np.isnan(a).any(-1)
    if not bad.any():
        return a
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]

swir = cropfull(fetch_s2_rgb(TILE, FIRES[DIXIE]["post"],
                             GT_DIR / "s2swir_dixie2021_2021post_display.npy",
                             bands=("B12", "B8A", "B04"), clear_classes=(2, 3, 4, 5, 6, 7, 11)))
rgb_a = fill_nearest(stretch(swir) ** 0.9)
S2_2022 = (("03", "2022-03-01/2022-04-15"), ("06", "2022-06-01/2022-06-30"),
           ("08", "2022-08-01/2022-08-31"), ("10", "2022-10-01/2022-10-31"))
scenes = [fill_nearest(stretch(cropfull(fetch_s2_rgb(TILE, win, GT_DIR / f"s2rgb_dixie2021_2022{m}.npy",
                                                     max_cloud=30)), 2, 98) ** 0.8)
          for m, win in S2_2022]

flat = x.reshape(-1, 128)
mu = flat.mean(0)
rng = np.random.default_rng(0)
_, _, Vt = np.linalg.svd(flat[rng.choice(len(flat), 60000, replace=False)] - mu,
                         full_matrices=False)
pcs = ((flat - mu) @ Vt[:3].T).reshape(h, w, 3)
# two features outside the family. Their labels are checked in the numbers section below.
from matplotlib.colors import LinearSegmentedColormap
NAMED = [(594, "south-facing slopes", "#b07d1a"), (332, "meadow", "#4f7f3f")]
OTHER = [k for k, _, _ in NAMED]
fmaps = [crop(np.asarray(F[:, :, k], dtype=np.float32)) for k in OTHER]
_fc = np.asarray(F[R0:R0 + CS, C0:C0 + CS][:, :, FAM], dtype=np.float32)
BURN4 = [FAM[i] for i in np.argsort(-(_fc > 0).mean((0, 1)))[:4]]
bmaps = [crop(np.asarray(F[:, :, k], dtype=np.float32)) for k in BURN4]
_pc = [crop(pcs[..., i]) for i in range(3)]
_lohi = [np.percentile(p, [2, 98]) for p in _pc]
rgb_pca = np.stack([((p - lo) / (hi - lo)).clip(0, 1) for p, (lo, hi) in zip(_pc, _lohi)], -1)
# the traced pixel, in stride-2 crop coordinates
PR, PC = 121, 205
ZW = 11                                       # zoom window, in 10 m pixels
_xf = load_tile(TILE, YEAR, stride=1)
_r0, _c0 = (R0 + PR) * S - ZW // 2, (C0 + PC) * S - ZW // 2
_z = (_xf[_r0:_r0 + ZW, _c0:_c0 + ZW].reshape(-1, 128) - mu) @ Vt[:3].T
rgb_zoom = np.stack([((_z[:, i] - lo) / (hi - lo)).clip(0, 1)
                     for i, (lo, hi) in enumerate(_lohi)], -1).reshape(ZW, ZW, 3)
del _xf

# ---- style ----------------------------------------------------------------------
js.use()
plt.rcParams["hatch.linewidth"] = 0.35
plt.rcParams["hatch.color"] = "#9a9a9a"
FS_LET, FS_ANN, FS_TICK = 8.0, 6.5, 6.0

# ---- geometry (inches) ----------------------------------------------------------
FIGW = 7.20
GS = 0.58
GC = 0.20
WC = 2.20
M_L = 0.05
M = (FIGW - 2 * M_L - GS - GC - WC) / 2
CAPH, GR, M_B, M_T = 0.0, 0.20, 0.03, 0.15
FIGH = M_B + 2 * (M + CAPH) + GR + M_T
XL, XR = M_L, M_L + M + GS
XC = XR + M + GC
Y2 = M_B + CAPH
Y1 = Y2 + M + CAPH + GR

fig = plt.figure(figsize=(FIGW, FIGH), dpi=300)

def box(x0, y0, wd, ht):
    return fig.add_axes([x0 / FIGW, y0 / FIGH, wd / FIGW, ht / FIGH])

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax

def letter(x0, y0, s):
    fig.text(x0 / FIGW, y0 / FIGH, s, fontsize=FS_LET, fontweight="bold", color=INK,
             ha="left", va="bottom")

ov = fig.add_axes([0, 0, 1, 1], zorder=5); ov.set_xlim(0, FIGW); ov.set_ylim(0, FIGH)
ov.axis("off"); ov.patch.set_alpha(0)

def arr(p0, p1, color=INK2, lw=0.8, ls="-", style="-|>", ms=7, **k):
    ov.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                linestyle=ls, mutation_scale=ms, shrinkA=0, shrinkB=0, **k))

# ---- a: fire --------------------------------------------------------------------
axa = clean(box(XL, Y1, M, M)); letter(XL - 0.02, Y1 + M + 0.03, "a")
axa.imshow(rgb_a, interpolation="nearest")

# ---- between a and b: imagery going into the encoder ------------------------------
OFF = 0.032
TH = GS - 3 * OFF - 0.05
sx0 = XL + M + 0.025
ey = Y1 + M / 2 - 0.30
sy0 = ey + 0.36
fx, fy = (PC + 0.5) / CS, 1 - (PR + 0.5) / CS
marks = []
for k, sc in enumerate(scenes):
    cx0, cy0 = sx0 + OFF * (3 - k), sy0 + OFF * (3 - k)
    ax = clean(box(cx0, cy0, TH, TH))
    ax.imshow(sc, interpolation="nearest")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor("white")
    marks.append((cx0 + fx * TH, cy0 + fy * TH))
for (mx, my) in marks:
    ov.add_patch(Rectangle((mx - 0.011, my - 0.011), 0.022, 0.022, fill=False, ec="white",
                           lw=0.7, zorder=6))
ov.plot([marks[0][0], marks[-1][0]], [marks[0][1], marks[-1][1]], color="white", lw=0.7,
        zorder=6)
ex0, ex1 = XL + M + 0.06, XR - 0.07
ov.add_patch(Polygon([(ex0, ey - 0.20), (ex1, ey - 0.07), (ex1, ey + 0.07), (ex0, ey + 0.20)],
                     closed=True, fc="#e7e4dc", ec=INK2, lw=0.6))
arr((sx0 + (TH + 3 * OFF) / 2, sy0 - 0.03), (sx0 + (TH + 3 * OFF) / 2, ey + 0.17), ms=5)
arr((ex1 + 0.005, ey), (XR - 0.01, ey), ms=5)

# ---- b: embedding, with a 10 m zoom on the traced pixel --------------------------
axb = clean(box(XR, Y1, M, M)); letter(XR - 0.02, Y1 + M + 0.03, "b")
axb.imshow(rgb_pca, interpolation="nearest")
pxl = (XR + (PC + 0.5) / CS * M, Y1 + M - (PR + 0.5) / CS * M)   # inches
zb = ZW * PX_M / (CS * S * PX_M) * M                                # inches
ov.add_patch(Rectangle((pxl[0] - zb / 2, pxl[1] - zb / 2), zb, zb, fill=False, ec="white", lw=0.6))
ZS = 0.56
zx, zy = XR + M - ZS - 0.04, Y1 + M - ZS - 0.04
axz = clean(box(zx, zy, ZS, ZS)); axz.set_zorder(6)
axz.imshow(rgb_zoom, interpolation="nearest")
for sp in axz.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.8); sp.set_edgecolor("white")
axz.add_patch(Rectangle((ZW // 2 - 0.5, ZW // 2 - 0.5), 1, 1, fill=False, ec="white", lw=0.9))
for cx, cy in ((-1, -1), (1, -1)):
    ov.plot([pxl[0] + cx * zb / 2, zx + (0 if cx < 0 else ZS)],
            [pxl[1] + zb / 2, zy], color="white", lw=0.4)

# ---- c: decomposition -------------------------------------------------------------
letter(XC - 0.02, Y1 + M + 0.03, "c")
TG, BO = 0.09, 0.035
TW = (WC - 0.12 - 2 * TG) / 3
CH = 0.10
GAPC = 0.24
_top = 0.15 / 2 + 6 * 0.026
_bot = CH / 2 + GAPC + TW + 3 * BO + 0.36
yc = Y1 + M / 2 + (_bot - _top) / 2
PCARD, POFF, NCARD = 0.15, 0.026, 7
xi0 = XC + 0.01
xc0, xc1 = XC + 0.46, XC + WC - 0.48
xo0 = xc1 + 0.19
NC = 26
CW = (xc1 - xc0) / NC
GREY_C = {14: 1, 25: 2}                      # code-bar cell and the map column it points to
GREY_X = [9, 20]                             # active cells not drawn as maps
CELL_COL = {i: NAMED[k - 1][2] for i, k in GREY_C.items()}
BURN_C = [1, 2, 5, 6]
# card colors: the traced pixel in front, then k-means centers of the colors in b
from sklearn.cluster import KMeans
_flat = rgb_pca.reshape(-1, 3)
_cand = _flat[np.random.default_rng(1).choice(len(_flat), 20000, replace=False)]
_km = KMeans(NCARD - 1, n_init=4, random_state=0).fit(_cand).cluster_centers_
cols = [rgb_zoom[ZW // 2, ZW // 2]] + list(_km[np.argsort(_km.sum(1))])
def pcards(x0):
    for k in range(NCARD - 1, -1, -1):
        ov.add_patch(Rectangle((x0 + k * POFF, yc - PCARD / 2 + k * POFF), PCARD, PCARD,
                               fc=cols[k], ec="white", lw=0.6, zorder=3))
pcards(xi0)
pcards(xo0)
for a0, a1, grow in ((xi0 + PCARD + (NCARD - 1) * POFF + 0.02, xc0 - 0.02, True),
                     (xc1 + 0.02, xo0 - 0.02, False)):
    s0, s1 = (0.045, 0.10) if grow else (0.10, 0.045)
    ov.add_patch(Polygon([(a0, yc - s0), (a1, yc - s1), (a1, yc + s1), (a0, yc + s0)],
                         closed=True, fc="#e7e4dc", ec=INK2, lw=0.6, zorder=3))
for i in range(NC):
    if i == 18:
        ov.text(xc0 + (i + 0.4) * CW, yc, "…", ha="center", va="center", fontsize=6)
        continue
    on_g = i in GREY_C or i in GREY_X
    fc = BURN if i in BURN_C else CELL_COL.get(i, "#7d7d7d") if on_g else "#eeece6"
    ov.add_patch(Rectangle((xc0 + i * CW, yc - CH / 2), CW * 0.8, CH, fc=fc, ec="none", zorder=3))
_a = ConnectionPatch(xyA=(ZW // 2 + 0.5, ZW // 2), coordsA=axz.transData,
                     xyB=(xi0 - 0.01, yc), coordsB=ov.transData, arrowstyle="-|>",
                     mutation_scale=6, color="white", lw=0.9, zorder=20)
_a.set_path_effects([pe.Stroke(linewidth=1.9, foreground="#1a1a1a", alpha=0.55), pe.Normal()])
fig.add_artist(_a)
cell = lambda i: xc0 + (i + 0.4) * CW
tx = lambda k: XC + 0.12 + k * (TW + TG)
yt = yc - CH / 2 - GAPC - TW

def tile(x0, y0, fm, cm="Greys", edge="#bdbdbd", lw=0.4):
    ax = clean(box(x0, y0, TW, TW))
    ax.imshow(np.ma.masked_less_equal(fm, 0), cmap=cm, vmin=0,
              vmax=np.percentile(fm[fm > 0], 99), interpolation="nearest")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw); sp.set_edgecolor(edge)

for (i, k), fm, (_, _, col) in zip(GREY_C.items(), fmaps[:2], NAMED):
    tile(tx(k), yt, fm, cm=LinearSegmentedColormap.from_list("", ["#f4efe4", col]),
         edge=col, lw=0.6)
    arr((cell(i), yc - CH / 2 - 0.01), (tx(k) + TW / 2, yt + TW + 0.01), color=col,
        lw=0.6, ms=3.5)
bx0, by0 = tx(0), yt
for j, bm in enumerate(bmaps[::-1]):
    o = BO * (3 - j)
    tile(bx0 + o - 3 * BO, by0 + o - 3 * BO, bm, cm="Oranges", edge=BURN,
         lw=0.9 if j == 3 else 0.5)
for i in BURN_C:
    arr((cell(i), yc - CH / 2 - 0.01), (bx0 + TW / 2 - 1.5 * BO, by0 + TW + 0.01), color=BURN,
        lw=0.6, ms=3.5)
ov.text((xc0 + xc1) / 2, yc + CH / 2 + 0.03, "1,024 features, 16 active per pixel",
        ha="center", va="bottom", fontsize=5.0, color=INK2, zorder=6)   # Nature's minimum size
_V = json.load(open(RESULTS_DIR / "fire_trajectories.json"))["dixie2021"]
_ages = [y - _V["fire_year"] for y in _V["years"]]
def course(feats):
    return [float(np.array(_V["mean_burn"][str(y)])[feats].mean()
                  - np.array(_V["mean_unburn"][str(y)])[feats].mean()) for y in _V["years"]]
courses = [(bx0 - 3 * BO, course(list(FAM)), BURN, 1.1)] + \
          [(tx(k), course([OTHER[j]]), NAMED[j][2], 0.9) for j, k in enumerate((1, 2))]
_lo = min(min(c) for _, c, _, _ in courses); _hi = max(max(c) for _, c, _, _ in courses)
SH = 0.15
ys = yt - 3 * BO - 0.05 - SH
for x0, c, col, lw in courses:
    axs = clean(box(x0, ys, TW, SH))
    axs.plot(_ages, c, color=col, lw=lw, solid_capstyle="round")
    axs.axvline(0, color=INK2, lw=0.5, ls=(0, (1.5, 1.5)))
    axs.axhline(0, color=FAINT, lw=0.4, zorder=0)
    axs.set_xlim(min(_ages) - 0.2, max(_ages) + 0.2)
    axs.set_ylim(_lo - 0.05 * (_hi - _lo), _hi + 0.05 * (_hi - _lo))
    axs.patch.set_alpha(0)
for (x0, _, col, _), lab in zip(courses, ["burn family"] + [n for _, n, _ in NAMED]):
    ov.text(x0 + TW / 2, ys - 0.03, lab, ha="center", va="top", fontsize=FS_TICK - 0.6,
            color=col, zorder=6, fontweight="bold" if col == BURN else "normal")
arr((XC + 0.10, ys - 0.14), (XR + M - 0.30, Y2 + M + 0.06), color=BURN, lw=1.1, ms=7,
    connectionstyle="arc3,rad=-0.35")

# ---- d: dNBR -----------------------------------------------------------------------
axd = clean(box(XL, Y2, M, M)); letter(XL - 0.02, Y2 + M + 0.03, "d")
axd.add_patch(Rectangle((0, 0), 1, 1, transform=axd.transAxes, fc="white", lw=0,
                        hatch="//////", zorder=0))
axd.imshow(np.ma.masked_invalid(crop(dn)), cmap="Greys", vmin=0,
           vmax=np.nanpercentile(dn, 97), interpolation="nearest", zorder=1)

# ---- e: burn family -------------------------------------------------------------
axe = clean(box(XR, Y2, M, M)); letter(XR - 0.02, Y2 + M + 0.03, "e")
axe.imshow(np.ma.masked_less_equal(crop(v), 0), cmap="Oranges", vmin=0,
           vmax=np.percentile(v[v > 0], 99), interpolation="nearest")
arr((XL + M + 0.10, Y2 + M / 2), (XR - 0.10, Y2 + M / 2), style="<|-|>", lw=0.7,
    ls=(0, (2.0, 1.6)), ms=5)

# one scale bar for all maps, in the gap between d and e
SB_KM = 1
sbw = SB_KM * 1000 / (CS * S * PX_M) * M
sbx = XL + M + 0.07
ov.plot([sbx, sbx], [Y2 + M, Y2 + M - sbw], color=INK, lw=1.6, solid_capstyle="butt")
ov.text(sbx + 0.06, Y2 + M - sbw / 2, f"{SB_KM} km", ha="left", va="center",
        fontsize=FS_TICK, color=INK)

# ---- f: timing -------------------------------------------------------------------
XF, WF, HF = XC + 0.42, WC - 0.42, M - 0.30
yf = Y2 + 0.28
axf = box(XF, yf, WF, HF); letter(XC - 0.02, Y2 + M + 0.03, "f")
V = json.load(open(RESULTS_DIR / "fire_trajectories.json"))
for fk in ("northcx2020", "caldor2021", "mosquito2022", "dixie2021"):
    tr = V[fk]; fy = tr["fire_year"]
    ages = [y - fy for y in tr["years"]]
    c = [float(np.array(tr["mean_burn"][str(y)])[FAM].mean()
               - np.array(tr["mean_unburn"][str(y)])[FAM].mean()) for y in tr["years"]]
    if fk == DIXIE:
        axf.plot(ages, c, "-o", color=BURN, lw=1.8, ms=3.0, mec="white", mew=0.5, zorder=3)
        axf.text(1.25, max(c), "Dixie", ha="left", va="center", fontsize=FS_TICK,
                 color=BURN, fontweight="bold")
    else:
        axf.plot(ages, c, "-o", color=GREYL, lw=0.9, ms=2.0, mec="white", mew=0.4, zorder=2)
axf.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
axf.set_xlim(-3.6, 4.6); axf.set_xticks(range(-2, 5, 2))
axf.set_xlabel("Years since fire", fontsize=FS_TICK, labelpad=1.5)
axf.set_ylabel("Burned − unburned\nactivation", fontsize=FS_TICK, labelpad=1.5,
               linespacing=1.1)
axf.tick_params(labelsize=FS_TICK, colors=INK, pad=1.2, length=2)
for s in ("top", "right"):
    axf.spines[s].set_visible(False)

# ---- numbers the caption quotes ----------------------------------------------------
# computed over the full tile, not the mapped crop
from sklearn.metrics import roc_auc_score
NUM = FigureNumbers("fig1")

# compare e with d, over pixels that have a dNBR value. The "inside" shares are the
# fraction of pixels in an activation class that dNBR calls burned.
_cov = ~np.isnan(dn)
_burned = dn[_cov] > DNBR_BURN_THRESHOLD
NUM.put("fig1.family_size", len(FAM))
NUM.put("fig1.dnbr_coverage_of_tile", round(float(_cov.mean()), 4))
NUM.put("fig1.auc_vs_perimeter", round(float(roc_auc_score(_burned.astype(int), v[_cov])), 4))
NUM.put("fig1.inside_any_activation",
        round(float((dn[(v > 0) & _cov] > DNBR_BURN_THRESHOLD).mean()), 4))
NUM.put("fig1.inside_above_median_activation",
        round(float((dn[(v > np.median(v[v > 0])) & _cov] > DNBR_BURN_THRESHOLD).mean()), 4),
        "share above the dNBR burn threshold among pixels with above-median family activation")

# check the labels in c: the share of a feature's active pixels in a reference class, next to the tile's share
_nb = np.load(GT_DIR / "gt_terrain_dixie.npz")["northness"][::S, ::S][:h, :w]
_wc = np.load(GT_DIR / "gt_worldcover_dixie.npy")[::S, ::S][:h, :w]
_on = lambda k: np.asarray(F[:, :, k]) > 0
_s, _ok = _on(594), ~np.isnan(_nb)
NUM.put("fig1.south_facing.active_share", round(float((_nb[_s & _ok] < 0).mean()), 3),
        "#594: share of its active pixels facing south (northness < 0)")
NUM.put("fig1.south_facing.tile_share", round(float((_nb[_ok] < 0).mean()), 3))
_m, _g = _on(332), _wc == 30
NUM.put("fig1.meadow.active_share", round(float(_g[_m].mean()), 3),
        "#332: share of its active pixels that are ESA WorldCover grassland")
NUM.put("fig1.meadow.tile_share", round(float(_g.mean()), 3))
# how often #332 is active by distance (m) to the meadow-forest edge. Positive is meadow, negative is forest.
_d = np.where(_g, distance_transform_edt(_g), -distance_transform_edt(~_g)) * 20.0
_inmap = _g | (_wc == 10)
for lab, lo, hi in (("forest_gt300m", -1e9, -300), ("forest_edge_0_40m", -40, 0),
                    ("meadow_edge_0_40m", 0, 40), ("meadow_gt100m", 100, 1e9)):
    _b = (_d > lo) & (_d <= hi) & _inmap
    NUM.put(f"fig1.meadow.active_rate.{lab}", round(float(_m[_b].mean()), 3),
            "#332 active rate by distance to the meadow-forest boundary")
NUM.save()

out = FIG_DIR / "fig1_decompose.png"
fig.savefig(out, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print("wrote", out, fig.get_size_inches())
