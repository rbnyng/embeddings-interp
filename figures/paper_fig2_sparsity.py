"""Figure 2: one raw embedding dimension next to one dictionary feature, on an Iowa Corn Belt tile.

a  the reference classes from USDA CDL 2020
b  pixels in the top and bottom 2% of one raw dimension, colored by CDL class
c  pixels where the corn feature is active, colored by CDL class
d  the mix of CDL classes along the raw dimension (20 equal-count bins), and where the feature is off or active
e  share of pixels where the corn and soy features are active in 2020 and 2021, on corn-to-soy and corn-to-corn pixels

Writes figures/fig2_sparsity.png and .pdf.
"""
import json

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lib.config import FIG_DIR, RESULTS_DIR
from lib.style import js

D = np.load(RESULTS_DIR / "fig2_data.npz")
J = json.load(open(RESULTS_DIR / "fig2_data.json"))
NAMES = J["names"]
CCOL = ["#e0a33c", "#3f7a42", "#9c6b4f", "#3b7ba8", "#bcd08a", "#1e5a2e",
        "#8fa9cf"]
INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
CORN, SOY = "#d68910", "#2e7d32"

h, w = D["shape"]
lab20, mask, poly = D["lab20"], D["mask"], J["poly"]

# ---- layout constants, all in inches ---------------------------------------
FS_LETTER, FS_CAP, FS_TICK = 11.5, 8.8, 8.2

L, RGT, TOP, BOT = 0.30, 0.30, 0.34, 0.58
GAP = 0.30
MW = 3.10
MH = MW / (w / h)

CAP_DY = 0.15
CAP_LINE = 0.20
BAND = 0.86
BH = 2.10

W = L + 3 * MW + 2 * GAP + RGT
H = TOP + MH + CAP_DY + BAND + BH + BOT

MAP_TOP = H - TOP
MAP_BOT = MAP_TOP - MH
CAP_Y = MAP_BOT - CAP_DY
BROW_TOP = CAP_Y - BAND
BROW_BOT = BOT

MX = [L + i * (MW + GAP) for i in range(3)]

YLAB_W = 0.74
D_X = L + YLAB_W
D_W = MX[1] + MW - D_X
E_X = MX[2] + 0.62
E_W = MW - 0.62
E_DROP = 0.24

js.use()
fig = plt.figure(figsize=(W, H), dpi=200)

def rect(x, y, wd, ht):
    """Turn a rectangle in inches, with y measured from the figure bottom, into figure fractions."""
    return [x / W, y / H, wd / W, ht / H]

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax

def letter(x, y, s):
    fig.text(x / W, y / H, s, fontsize=FS_LETTER, fontweight="bold",
             color=INK, ha="left", va="bottom")

def cap(xc, y, s, col=INK):
    fig.text(xc / W, y / H, s, ha="center", va="top", fontsize=FS_CAP,
             color=col, linespacing=1.5)

def hexrgb(c):
    return [int(c[j:j + 2], 16) / 255 for j in (1, 3, 5)]

def key(xc, y, handles, ncol=1, **kw):
    """Place a legend by its top center, given in inches."""
    kw.setdefault("handlelength", 0.9)
    kw.setdefault("labelspacing", 0.35)
    return fig.legend(handles=handles, fontsize=FS_CAP, frameon=False,
                      loc="upper center", ncol=ncol,
                      bbox_to_anchor=(xc / W, y / H),
                      bbox_transform=fig.transFigure, handletextpad=0.5,
                      borderpad=0.0, **kw)

# ---- a: CDL reference, and the class key ------------------------------------
axa = clean(fig.add_axes(rect(MX[0], MAP_BOT, MW, MH)))
letter(MX[0], MAP_TOP + 0.06, "a")
ref = np.ones((h, w, 3)) * 0.965
for i, c in enumerate(CCOL):
    ref[lab20 == i] = hexrgb(c)
axa.imshow(ref, interpolation="nearest", aspect="auto")
cap(MX[0] + MW / 2, CAP_Y, "USDA CDL, 2020")
key(W / 2, CAP_Y - CAP_LINE - 0.08,
    [Patch(fc=c, ec="none", label=n) for n, c in zip(NAMES, CCOL)],
    ncol=len(NAMES), handlelength=0.9, columnspacing=2.0)

# ---- b: both tails of one raw dimension -------------------------------------
axb = clean(fig.add_axes(rect(MX[1], MAP_BOT, MW, MH)))
letter(MX[1], MAP_TOP + 0.06, "b")
pv = D["poly_full"]
lo, hi = np.percentile(pv[mask], 2), np.percentile(pv[mask], 98)
tails = np.ones((h, w, 3)) * 0.965
tail = ((pv >= hi) | (pv <= lo)) & mask
for i, c in enumerate(CCOL):
    tails[tail & (lab20 == i)] = hexrgb(c)
axb.imshow(tails, interpolation="nearest", aspect="auto")
cap(MX[1] + MW / 2, CAP_Y,
    f"raw embedding dim {poly['d']}, top and bottom 2%")


# ---- d: class mix along the raw dimension (left) and for the feature (right).
# Block widths are shares of pixels.
DGAP = 0.30
DW1 = (D_W - DGAP) * 0.62
DW2 = D_W - DGAP - DW1
DH = BROW_TOP - BROW_BOT
axd = fig.add_axes(rect(D_X, BROW_BOT, DW1, DH))
axd2 = fig.add_axes(rect(D_X + DW1 + DGAP, BROW_BOT, DW2, DH))
letter(L, BROW_TOP + 0.06, "d")
labm = lab20[mask]
K = len(NAMES)
def mix(idx):
    return np.bincount(labm[idx], minlength=K)[:K] / max(len(idx), 1)
def stack(ax, x0, wd, comp):
    y0 = 0.0
    for i, v in enumerate(comp):
        ax.bar(x0, v, width=wd, bottom=y0, align="edge", color=CCOL[i],
               ec="white", lw=0.5)
        y0 += v
NB = 20
order_ = np.argsort(D["poly_full"][mask], kind="stable")
for j, idx in enumerate(np.array_split(order_, NB)):
    stack(axd, j / NB, 1 / NB, mix(idx))
on = D["corn20"][mask] > 0
f_on = on.mean()
c_off, c_on = mix(np.flatnonzero(~on)), mix(np.flatnonzero(on))
stack(axd2, 0.0, 1 - f_on, c_off)
stack(axd2, 1 - f_on, f_on, c_on)
ci = NAMES.index("corn")
axd2.text(1 - f_on / 2, c_on[ci] / 2, f"corn\n{c_on[ci]:.0%}", ha="center",
          va="center", fontsize=FS_CAP, color=INK, linespacing=1.3)
c_first = mix(np.array_split(order_, NB)[0])
c_last = mix(np.array_split(order_, NB)[-1])
di = NAMES.index("developed")
axd.text(0.5 / NB, c_first[ci] / 2, f"corn {c_first[ci]:.0%}", rotation=90,
         ha="center", va="center", fontsize=FS_CAP, color=INK)
axd.text(1 - 0.5 / NB, c_last[:di].sum() + c_last[di] / 2,
         f"developed {c_last[di]:.0%}", rotation=90, ha="center",
         va="center", fontsize=FS_CAP, color="white")
for ax, ttl in ((axd, f"raw dim {poly['d']}"),
                (axd2, f"feature {J['corn']['f']}")):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 1.03, ttl, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=FS_CAP, color=INK)
    ax.tick_params(labelsize=FS_TICK, colors=INK, length=0)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
axd.set_yticks([0, 0.5, 1.0])
axd.set_yticklabels(["0", "0.5", "1"])
for ax in (axd, axd2):
    ax.tick_params(axis="x", pad=7)
axd.set_ylabel("Class share", fontsize=FS_CAP, color=INK)
axd.set_xticks([0.025, 0.975])
axd.set_xticklabels(["lowest 5%", "highest 5%"])
axd.set_xlabel(f"Pixels sorted by dim {poly['d']}, 20 equal bins",
               fontsize=FS_CAP, color=INK)
axd2.set_yticks([])
axd2.set_xticks([(1 - f_on) / 2, 1 - f_on / 2])
axd2.set_xticklabels([f"off  {1 - f_on:.0%}", f"active  {f_on:.0%}"])
axd2.set_xlabel("Share of pixels", fontsize=FS_CAP, color=INK)

# ---- c: where one feature is active. e: corn and soy features over two years.
axc = clean(fig.add_axes(rect(MX[2], MAP_BOT, MW, MH)))
letter(MX[2], MAP_TOP + 0.06, "c")
feat = np.ones((h, w, 3)) * 0.965
act = D["corn20"] > 0
for i, c in enumerate(CCOL):
    feat[act & (lab20 == i)] = hexrgb(c)
axc.imshow(feat, interpolation="nearest", aspect="auto")
cap(MX[2] + MW / 2, CAP_Y, f"decomposed feature {J['corn']['f']}")

sw = J["swap"]
letter(MX[2], BROW_TOP + 0.06, "e")
key(MX[2] + MW / 2, BROW_TOP + 0.02,
    [Line2D([], [], color=CORN, lw=2.4, label="corn feature"),
     Line2D([], [], color=SOY, lw=2.4, label="soy feature")],
    ncol=2, handlelength=1.6, columnspacing=1.4)
E_TOP = BROW_TOP - E_DROP
GX = 2.35
axe = fig.add_axes(rect(E_X, BROW_BOT, E_W, E_TOP - BROW_BOT))
groups = [(0.0, "corn → soy",
           (sw["corn_feat_on_corn2020"], sw["corn_feat_on_same_px_2021"]),
           (sw["soy_feat_on_rot_2020"], sw["soy_feat_on_same_px_2021"])),
          (GX, "corn → corn",
           (sw["corn_feat_stays_2020"], sw["corn_feat_stays_2021"]),
           (sw["soy_feat_stays_2020"], sw["soy_feat_stays_2021"]))]
for x0, ttl, cvals, svals in groups:
    for vals, col in ((cvals, CORN), (svals, SOY)):
        axe.plot([x0, x0 + 1], vals, "-o", color=col, lw=2.4, ms=7,
                 mec="white", mew=1.0, zorder=3)
        for xi, v in zip((x0, x0 + 1), vals):
            up = v > 0.5
            axe.text(xi, v + (0.05 if up else -0.05), f"{v:.2f}", ha="center",
                     va="bottom" if up else "top", fontsize=FS_TICK, color=col)
    axe.annotate(ttl, xy=(x0 + 0.5, 0), xycoords=("data", "axes fraction"),
                 xytext=(0, -20), textcoords="offset points", ha="center",
                 va="top", fontsize=FS_CAP, color=INK)
axe.set_xlim(-0.45, GX + 1.45); axe.set_ylim(-0.17, 1.10)
axe.set_xticks([0, 1, GX, GX + 1])
axe.set_xticklabels(["2020", "2021", "2020", "2021"])
axe.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axe.set_ylabel("Fraction of pixels active", fontsize=FS_CAP)
axe.tick_params(labelsize=FS_TICK, colors=INK)
for s in ("top", "right"):
    axe.spines[s].set_visible(False)

out = FIG_DIR / "fig2_sparsity.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.05, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05,
            facecolor="white")
print("wrote", out, f"{W:.2f} x {H:.2f} in")
