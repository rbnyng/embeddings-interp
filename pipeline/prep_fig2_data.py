"""Compute the Figure 2 data: dictionary features against raw embedding dimensions on one Iowa tile, with CDL labels.

The tile is grid_-93.55_42.05 in 2020 and 2021 at stride 2, encoded with the Iowa dictionary. Pixels are labeled with
seven USDA CDL classes: corn, soy, developed, water, grass, forest and wetland. The polysemantic axis is the raw
dimension whose top and bottom 2% tails have different majority classes, with the largest sum of the two majority
shares. The corn and soy features have the best F1 against their class in 2020, among features active on more than
500 pixels. The swap test follows both features on pixels that rotated from corn in 2020 to soy in 2021.

Usage: python pipeline/prep_fig2_data.py
Writes results/fig2_data.npz and fig2_data.json.
"""
import json

import numpy as np

from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR
from lib.tiles import load_tile
from lib.sae import encode_array, load_sae

TILE, S = "grid_-93.55_42.05", 2
CLS = {"corn": [1], "soy": [5], "developed": [121, 122, 123, 124],
       "water": [111], "grass": [176], "forest": [141, 142, 143],
       "wetland": [190, 195]}
NAMES = list(CLS)
cdl_all = dict(np.load(GT_DIR / "gt_cdl.npz"))

sae = load_sae(path=CKPT_DIR / "sae_iowa.pt")

def year(y):
    x = load_tile(TILE, y, stride=S)
    h, w, D = x.shape
    cdl = cdl_all[str(y)][::S, ::S][:h, :w]
    lab = np.full((h, w), -1, np.int8)
    for i, codes in enumerate(CLS.values()):
        lab[np.isin(cdl, codes)] = i
    F = np.zeros((h, w, 1024), np.float32)
    for r0 in range(0, h, 128):
        r1 = min(r0 + 128, h)
        F[r0:r1] = encode_array(sae, x[r0:r1])
    return x, F, lab

x20, F20, lab20 = year(2020)
x21, F21, lab21 = year(2021)
h, w = lab20.shape
M20 = lab20 >= 0
L20 = lab20[M20]
base = np.array([(L20 == i).mean() for i in range(len(CLS))])
print(f"tile {TILE}  {h}x{w} at stride {S}; labelled {M20.mean():.1%}")
print("base rates:", {n: round(float(b), 3) for n, b in zip(NAMES, base)})

ff20 = F20.reshape(-1, 1024)[M20.ravel()]
xx20 = x20.reshape(-1, 128)[M20.ravel()]
freq = (ff20 > 0).mean(0)
alive = np.where((freq > 0.01) & (freq < 0.5))[0]
p = float(np.median(freq[alive]))
print(f"{len(alive)} live features; matched active rate p = {p:.3f}")

def comp_from_idx(idx):
    q = np.array([(L20[idx] == i).mean() for i in range(len(CLS))])
    return q, float((q / base).max()), int(q.argmax())

def top_idx(v, rate, bottom=False):
    k = max(int(rate * v.size), 50)
    return np.argpartition(v, k)[:k] if bottom else np.argpartition(v, -k)[-k:]

# ---- each live feature and raw-dimension tail: the class mix of an active set of matched rate ----
feat_rows, dim_rows = [], []
for k in alive:
    q, lift, arg = comp_from_idx(np.flatnonzero(ff20[:, k] > 0))
    ent = -(q[q > 0] * np.log(q[q > 0])).sum() / np.log(len(CLS))
    feat_rows.append((float(lift), float(1 - ent), int(k), int(arg)))
for d in range(128):
    for bot in (False, True):
        q, lift, arg = comp_from_idx(top_idx(xx20[:, d], p, bottom=bot))
        ent = -(q[q > 0] * np.log(q[q > 0])).sum() / np.log(len(CLS))
        dim_rows.append((float(lift), float(1 - ent), int(d), int(arg), bool(bot)))

best_dim = max(dim_rows, key=lambda r: r[0])
print(f"best-enriched raw dim overall: dim {best_dim[2]} "
      f"({'bottom' if best_dim[4] else 'top'} tail) lift {best_dim[0]:.1f} "
      f"-> {NAMES[best_dim[3]]}, purity {best_dim[1]:.2f}")

# The panel compares the corn feature with the raw dimension that best detects corn. The most enriched dimension
# overall, found above, can pick out a rare class that is invisible on a map.
corn_dims = []
for d in range(128):
    for bot in (False, True):
        idx = top_idx(xx20[:, d], p, bottom=bot)
        q, _, _ = comp_from_idx(idx)
        ent = -(q[q > 0] * np.log(q[q > 0])).sum() / np.log(len(CLS))
        corn_dims.append((float(q[0] / base[0]), float(1 - ent), int(d),
                          bool(bot), q.tolist()))
dim_corn = max(corn_dims, key=lambda r: r[0])
print(f"strongest CORN dim: dim {dim_corn[2]} "
      f"({'bottom' if dim_corn[3] else 'top'} tail) corn enrichment "
      f"{dim_corn[0]:.1f}x, corn share {dim_corn[4][0]:.2f}, "
      f"purity {dim_corn[1]:.2f}")

# ---- the polysemantic axis: the dimension whose two tails disagree most ----
two = []
for d in range(128):
    qt, _, at = comp_from_idx(top_idx(xx20[:, d], 0.02))
    qb, _, ab = comp_from_idx(top_idx(xx20[:, d], 0.02, bottom=True))
    if at != ab:
        two.append((float(qt[at] + qb[ab]), int(d), int(at), int(ab),
                    qt.tolist(), qb.tolist()))
two.sort(reverse=True)
poly = two[0]
print(f"most polysemantic axis: dim {poly[1]} — top tail {NAMES[poly[2]]} "
      f"{poly[4][poly[2]]:.2f}, bottom tail {NAMES[poly[3]]} {poly[5][poly[3]]:.2f}")

# ---- the corn and soy features, selected on 2020 ----------------------------
# Chosen by F1, not purity. Purity alone picks very sparse features that fire in one year and not the next.
def f1(a, C):
    i = int(np.count_nonzero(a & C))
    return 2 * i / (int(a.sum()) + int(C.sum())) if i else 0.0

def best_feature(cls_i):
    C = (lab20 == cls_i).ravel()
    sc = [(f1(F20.reshape(-1, 1024)[:, k] > 0, C), k) for k in range(1024)
          if (F20.reshape(-1, 1024)[:, k] > 0).sum() > 500]
    return max(sc)[1] if sc else None

f_corn, f_soy = best_feature(0), best_feature(1)
Cc20, Cc21 = (lab20 == 0).ravel(), (lab21 == 0).ravel()
Cs20, Cs21 = (lab20 == 1).ravel(), (lab21 == 1).ravel()
stab = dict(
    corn_f1_2020=f1(F20.reshape(-1, 1024)[:, f_corn] > 0, Cc20),
    corn_f1_2021=f1(F21.reshape(-1, 1024)[:, f_corn] > 0, Cc21),
    soy_f1_2020=f1(F20.reshape(-1, 1024)[:, f_soy] > 0, Cs20),
    soy_f1_2021=f1(F21.reshape(-1, 1024)[:, f_soy] > 0, Cs21))
print(f"corn feature F{f_corn}: F1 {stab['corn_f1_2020']:.3f} (2020) -> "
      f"{stab['corn_f1_2021']:.3f} (2021, unchanged feature)")
print(f"soy  feature F{f_soy}: F1 {stab['soy_f1_2020']:.3f} (2020) -> "
      f"{stab['soy_f1_2021']:.3f} (2021, unchanged feature)")

qc, _, _ = comp_from_idx(np.flatnonzero(ff20[:, f_corn] > 0))
qs, _, _ = comp_from_idx(np.flatnonzero(ff20[:, f_soy] > 0))
qd, _, _ = comp_from_idx(top_idx(xx20[:, best_dim[2]], p, bottom=best_dim[4]))

# ---- the swap: do the features chosen on 2020 follow the 2021 rotation? ----
rot_cs = (lab20 == 0) & (lab21 == 1)      # corn in 2020, soy in 2021
rot_sc = (lab20 == 1) & (lab21 == 0)
stay_c = (lab20 == 0) & (lab21 == 0)
a20c, a21c = F20[:, :, f_corn] > 0, F21[:, :, f_corn] > 0
a20s, a21s = F20[:, :, f_soy] > 0, F21[:, :, f_soy] > 0
swap = dict(
    n_corn_to_soy=int(rot_cs.sum()), n_soy_to_corn=int(rot_sc.sum()),
    n_corn_stays=int(stay_c.sum()),
    corn_feat_on_corn2020=float(a20c[rot_cs].mean()),
    corn_feat_on_same_px_2021=float(a21c[rot_cs].mean()),
    soy_feat_on_same_px_2021=float(a21s[rot_cs].mean()),
    soy_feat_on_rot_2020=float(a20s[rot_cs].mean()),
    soy_feat_stays_2020=float(a20s[stay_c].mean()),
    soy_feat_stays_2021=float(a21s[stay_c].mean()),
    corn_feat_stays_2020=float(a20c[stay_c].mean()),
    corn_feat_stays_2021=float(a21c[stay_c].mean()),
)
print("\nswap test (pixels that rotated corn 2020 -> soy 2021, "
      f"n={swap['n_corn_to_soy']:,}):")
print(f"  corn feature active in 2020: {swap['corn_feat_on_corn2020']:.3f}")
print(f"  corn feature active in 2021: {swap['corn_feat_on_same_px_2021']:.3f}")
print(f"  soy  feature active in 2021: {swap['soy_feat_on_same_px_2021']:.3f}")
print(f"  (control) corn feature on pixels that STAYED corn, 2021: "
      f"{swap['corn_feat_stays_2021']:.3f}")

np.savez_compressed(
    RESULTS_DIR / "fig2_data.npz",
    lab20=lab20, lab21=lab21,
    dim_map=xx20[:, best_dim[2]].astype(np.float32),
    dim_corn_full=x20[:, :, dim_corn[2]].astype(np.float32),
    corn20=F20[:, :, f_corn], corn21=F21[:, :, f_corn],
    soy20=F20[:, :, f_soy], soy21=F21[:, :, f_soy],
    mask=M20, shape=np.array([h, w]),
    dim_full=x20[:, :, best_dim[2]].astype(np.float32),
    poly_full=x20[:, :, poly[1]].astype(np.float32),
    soy21_map=F21[:, :, f_soy], corn21_map=F21[:, :, f_corn])
json.dump(dict(names=NAMES, stability=stab, base=base.tolist(), rate=p,
               best_dim=dict(d=best_dim[2], bottom=best_dim[4], lift=best_dim[0],
                             purity=best_dim[1], cls=NAMES[best_dim[3]],
                             comp=qd.tolist()),
               dim_corn=dict(d=dim_corn[2], bottom=dim_corn[3],
                             lift=dim_corn[0], purity=dim_corn[1],
                             comp=dim_corn[4]),
               poly=dict(d=poly[1], top_cls=NAMES[poly[2]],
                         bot_cls=NAMES[poly[3]], top=poly[4], bot=poly[5]),
               corn=dict(f=f_corn, comp=qc.tolist()),
               soy=dict(f=f_soy, comp=qs.tolist()),
               feat_rows=feat_rows, dim_rows=dim_rows, swap=swap),
          open(RESULTS_DIR / "fig2_data.json", "w"))
print("\nsaved results/fig2_data.{npz,json}")
