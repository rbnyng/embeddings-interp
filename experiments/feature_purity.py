"""Feature purity at scale: test whether single features pick out one land-cover class, while raw dimensions mix
classes at their two ends.

sample takes up to 10,000 pixels from each crop-atlas tile at stride 2, labeled by USDA CDL 2022 (a crop-atlas held-out
year) grouped into the classes in GROUPS. analyze encodes them with the crop-atlas dictionary and compares four bases.
A feature or tail is single-class when its most common class makes up at least 75% of its pixels.
  SAE features    each feature's active pixels, for features active on at least 200 pixels
  raw dimensions  the top and bottom 2% of each of the 128 embedding dimensions
  PCA components  the same two tails of each principal component
  random          the same two tails of 128 random orthonormal directions
best and fixed score corn and soy by F1 on the 100 Corn Belt tiles in 2020, with the Iowa dictionary. A single feature
(active or not) is compared with a raw dimension, end and threshold (one of 49 quantiles from 2% to 98%).
  best   picks the feature and the raw rule on each tile
  fixed  uses the Fig. 2 features (904 corn, 114 soy) and one raw rule picked on the Fig. 2 Iowa tile

Usage: python experiments/feature_purity.py sample|analyze|best|fixed   (run sample before analyze)
Writes results/feature_purity.json, feature_purity_best.json and feature_purity_fixed.json.
"""
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import DATA_DIR, RESULTS_DIR, workers
from lib.sae import encode_array, load_sae
from lib.tiles import load_tile

YEAR = 2022
N_PER_TILE = 10_000
TAIL = 0.02
MIN_ACTIVE = 200
PURE = 0.75
SAMPLE = DATA_DIR / "intermediate" / "feature_purity_sample.npz"
OUT = RESULTS_DIR / "feature_purity.json"

# CDL codes grouped into land-cover classes. Any code not listed is "other".
GROUPS = {
    "corn": (1,), "soy": (5,), "winter wheat": (24,), "rice": (3,), "cotton": (2,),
    "perennial crops": (66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 204, 210, 211, 212, 217, 218,
                        220, 223),
    "fallow": (61,),
    "other crops": tuple(c for c in list(range(4, 61)) + list(range(225, 255))
                         if c not in (5, 24)),
    "developed": (121, 122, 123, 124),
    "forest": (63, 141, 142, 143),
    "grass/pasture": (62, 176),
    "shrub": (64, 152),
    "water": (83, 111),
    "wetland": (87, 190, 195),
    "barren": (65, 131),
}
NAMES = list(GROUPS) + ["other"]


def group(codes):
    out = np.full(codes.shape, len(GROUPS), dtype=np.int8)
    for i, cs in enumerate(GROUPS.values()):
        out[np.isin(codes, cs)] = i
    return out


def _tile(tile):
    import experiments.crop_atlas as ca
    try:
        cdl = ca.cdl_stack(tile, [YEAR])[YEAR]
        x = load_tile(tile, YEAR, stride=ca.S)
        H, W = min(x.shape[0], cdl.shape[0]), min(x.shape[1], cdl.shape[1])
        x, cdl = x[:H, :W].reshape(-1, 128), cdl[:H, :W].ravel()
        ok = (cdl > 0) & (np.abs(x).sum(1) > 0)
        idx = np.flatnonzero(ok)
        rng = np.random.default_rng(zlib.crc32(tile.encode()))
        pick = rng.choice(idx, min(N_PER_TILE, idx.size), replace=False)
        return tile, x[pick].astype(np.float32), group(cdl[pick])
    except Exception as e:
        print("skip", tile, repr(e), flush=True)
        return tile, None, None


def phase_sample():
    import experiments.crop_atlas as ca
    tiles = ca.TILES
    X, Y, T = [], [], []
    with ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for tile, x, y in ex.map(_tile, tiles):
            if x is not None:
                X.append(x); Y.append(y); T += [tile] * len(y)
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SAMPLE, X=np.concatenate(X), y=np.concatenate(Y), tile=np.array(T))
    print(f"{len(X)} tiles, {sum(len(y) for y in Y):,} pixels", flush=True)


def _purity(labels, base):
    """For the pixels of one feature or tail, return the most common class, its share, and that share over the
    class's base rate."""
    counts = np.bincount(labels, minlength=len(NAMES))
    c = int(counts.argmax())
    share = counts[c] / counts.sum()
    return c, float(share), float(share / base[c])


def _tails(scores, y, base):
    """Return _purity for the low TAIL and the high TAIL of a score."""
    lo, hi = np.quantile(scores, [TAIL, 1 - TAIL])
    return _purity(y[scores <= lo], base), _purity(y[scores >= hi], base)


def phase_analyze():
    import experiments.crop_atlas as ca
    z = np.load(SAMPLE)
    X, y = z["X"], z["y"].astype(np.int64)
    base = np.bincount(y, minlength=len(NAMES)) / len(y)
    sae = load_sae(path=ca.CKPT)
    torch.set_num_threads(32)
    F = np.concatenate([encode_array(sae, X[i:i + 200_000].reshape(1, -1, 128))[0]
                        for i in range(0, len(X), 200_000)])

    out = dict(n_pixels=int(len(y)), n_tiles=int(len(set(z["tile"]))), tail=TAIL,
               min_active=MIN_ACTIVE, pure=PURE,
               base_rates={n: round(float(b), 4) for n, b in zip(NAMES, base)})

    act = F > 0
    n_act = act.sum(0)
    live = np.flatnonzero(n_act >= MIN_ACTIVE)
    sae_units = [_purity(y[act[:, j]], base) for j in live]
    out["sae"] = dict(n_features=int(F.shape[1]), n_scored=int(len(live)),
                      mean_active_share=round(float(act.mean()), 4),
                      share=[round(u[1], 4) for u in sae_units],
                      lift=[round(u[2], 3) for u in sae_units],
                      cls=[NAMES[u[0]] for u in sae_units],
                      n_active=[int(n_act[j]) for j in live])

    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc[np.random.default_rng(0).choice(len(Xc), 200_000, replace=False)],
                             full_matrices=False)
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(128, 128)))
    bases = {"raw": np.eye(128), "pca": Vt.T, "random": Q}
    for name, B in bases.items():
        S = Xc @ B
        tails = [_tails(S[:, d], y, base) for d in range(B.shape[1])]
        out[name] = dict(
            share=[round(t[k][1], 4) for t in tails for k in (0, 1)],
            lift=[round(t[k][2], 3) for t in tails for k in (0, 1)],
            split_ends=float(np.mean([t[0][0] != t[1][0] for t in tails])))

    def frac(v):
        return round(float(np.mean(np.asarray(v) >= PURE)), 4)
    w = np.asarray(out["sae"]["n_active"], float)
    out["summary"] = dict(
        sae_pure=frac(out["sae"]["share"]),
        sae_pure_weighted=round(float(np.sum(w * (np.asarray(out["sae"]["share"]) >= PURE)) / w.sum()), 4),
        sae_median_share=round(float(np.median(out["sae"]["share"])), 4),
        **{f"{b}_pure": frac(out[b]["share"]) for b in bases},
        **{f"{b}_median_share": round(float(np.median(out[b]["share"])), 4) for b in bases},
        **{f"{b}_split_ends": round(out[b]["split_ends"], 4) for b in bases})
    json.dump(out, open(OUT, "w"))
    print(json.dumps(out["summary"], indent=1))
    print("base rates:", out["base_rates"])




# ---- best: on each Corn Belt tile, for corn and soy, the best single feature (active or not) against the best raw
# dimension, end and threshold. All are chosen per tile to maximize F1, which favors the raw dimensions.
BEST_OUT = RESULTS_DIR / "feature_purity_best.json"
QS = np.linspace(0.02, 0.98, 49)


def _f1(pred, truth):
    tp = np.sum(pred & truth)
    return 0.0 if tp == 0 else 2 * tp / (pred.sum() + truth.sum())


def _best_tile(tile):
    import experiments.crop_atlas as ca
    from lib.config import CKPT_DIR
    try:
        torch.set_num_threads(2)
        cdl = ca.cdl_stack(tile, [2020])[2020]
        x = load_tile(tile, 2020, stride=ca.S)
        H, W = min(x.shape[0], cdl.shape[0]), min(x.shape[1], cdl.shape[1])
        x, cdl = x[:H, :W], cdl[:H, :W].ravel()
        sae = load_sae(path=CKPT_DIR / "sae_iowa.pt")
        F = encode_array(sae, x).reshape(-1, 1024) > 0
        X = x.reshape(-1, 128)
        ok = (cdl > 0) & (np.abs(X).sum(1) > 0)
        F, X, cdl = F[ok], X[ok], cdl[ok]
        out = dict(tile=tile)
        for name, code in (("corn", 1), ("soy", 5)):
            t = cdl == code
            fs = [_f1(F[:, j], t) for j in range(1024)]
            j = int(np.argmax(fs))
            best_raw = 0.0
            for d in range(128):
                qs = np.quantile(X[:, d], QS)
                for q in qs:
                    best_raw = max(best_raw, _f1(X[:, d] >= q, t), _f1(X[:, d] <= q, t))
            out[name] = dict(base=float(t.mean()), sae_f1=float(fs[j]), sae_feature=j,
                             sae_precision=float((F[:, j] & t).sum() / max(F[:, j].sum(), 1)),
                             raw_f1=float(best_raw))
        return out
    except Exception as e:
        return dict(tile=tile, error=repr(e))


def phase_best():
    tiles = json.load(open(RESULTS_DIR / "cornbelt_selection.json"))
    with ProcessPoolExecutor(max_workers=workers(50)) as ex:
        rows = list(ex.map(_best_tile, tiles))
    rows = [r for r in rows if "error" not in r]
    summ = {}
    for c in ("corn", "soy"):
        s = np.array([r[c]["sae_f1"] for r in rows]); w = np.array([r[c]["raw_f1"] for r in rows])
        summ[c] = dict(n_tiles=len(rows), sae_f1_median=round(float(np.median(s)), 4),
                       raw_f1_median=round(float(np.median(w)), 4),
                       sae_better_share=round(float(np.mean(s > w)), 4),
                       sae_precision_median=round(float(np.median([r[c]["sae_precision"] for r in rows])), 4))
    json.dump(dict(summary=summ, rows=rows), open(BEST_OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    iowa = [r for r in rows if r["tile"] == "grid_-93.55_42.05"]
    if iowa:
        print("Iowa tile (Fig. 2):", {c: (round(iowa[0][c]["sae_f1"], 3), iowa[0][c]["sae_feature"]) for c in ("corn", "soy")})


# ---- fixed: the Fig. 2 features (FEAT, active or not) against a raw rule (dimension, end, threshold) chosen once
# on the Fig. 2 Iowa tile. Both are applied unchanged to all 100 Corn Belt tiles.
FIXED_OUT = RESULTS_DIR / "feature_purity_fixed.json"
IOWA = "grid_-93.55_42.05"
FEAT = dict(corn=904, soy=114)


def _tile_arrays(tile):
    import experiments.crop_atlas as ca
    from lib.config import CKPT_DIR
    torch.set_num_threads(2)
    cdl = ca.cdl_stack(tile, [2020])[2020]
    x = load_tile(tile, 2020, stride=ca.S)
    H, W = min(x.shape[0], cdl.shape[0]), min(x.shape[1], cdl.shape[1])
    x, cdl = x[:H, :W], cdl[:H, :W].ravel()
    sae = load_sae(path=CKPT_DIR / "sae_iowa.pt")
    F = encode_array(sae, x).reshape(-1, 1024)
    X = x.reshape(-1, 128)
    ok = (cdl > 0) & (np.abs(X).sum(1) > 0)
    return F[ok], X[ok], cdl[ok]


def _fixed_tile(args):
    tile, rules = args
    try:
        F, X, cdl = _tile_arrays(tile)
        out = dict(tile=tile)
        for c, code in (("corn", 1), ("soy", 5)):
            t = cdl == code
            d, side, thr = rules[c]
            raw = X[:, d] >= thr if side == "high" else X[:, d] <= thr
            out[c] = dict(sae_f1=float(_f1(F[:, FEAT[c]] > 0, t)), raw_f1=float(_f1(raw, t)))
        return out
    except Exception as e:
        return dict(tile=tile, error=repr(e))


def _best_raw_rule(X, t):
    best, n_pos = (0.0, None), t.sum()
    for d in range(X.shape[1]):
        # Sort each dimension once, then read true positives at every threshold off cumulative sums.
        o = np.argsort(X[:, d], kind="stable")
        v, tp_cum = X[o, d], np.cumsum(t[o])
        for thr in np.quantile(X[:, d], QS):
            k_le = np.searchsorted(v, thr, side="right")
            k_lt = np.searchsorted(v, thr, side="left")
            tp_low = tp_cum[k_le - 1] if k_le else 0
            tp_high = n_pos - (tp_cum[k_lt - 1] if k_lt else 0)
            for side, tp, npred in (("low", tp_low, k_le), ("high", tp_high, len(v) - k_lt)):
                f = 0.0 if tp == 0 else 2 * tp / (npred + n_pos)
                if f > best[0]:
                    best = (float(f), (d, side, float(thr)))
    return best


def phase_fixed():
    F, X, cdl = _tile_arrays(IOWA)
    rules = {}
    for c, code in (("corn", 1), ("soy", 5)):
        t = cdl == code
        best = _best_raw_rule(X, t)
        rules[c] = best[1]
        print(f"{c}: Iowa-tile rule raw dim {best[1][0]} {best[1][1]} end, threshold {best[1][2]:.3f} "
              f"(F1 {best[0]:.3f}); feature {FEAT[c]} F1 {_f1(F[:, FEAT[c]] > 0, t):.3f}", flush=True)
    tiles = json.load(open(RESULTS_DIR / "cornbelt_selection.json"))
    # Use spawn, not fork. This process has already run torch, and forking after torch starts its threads deadlocks the
    # workers.
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers(50), mp_context=mp.get_context("spawn")) as ex:
        rows = [r for r in ex.map(_fixed_tile, [(t, rules) for t in tiles]) if "error" not in r]
    summ = {}
    for c in ("corn", "soy"):
        s = np.array([r[c]["sae_f1"] for r in rows]); w = np.array([r[c]["raw_f1"] for r in rows])
        summ[c] = dict(n_tiles=len(rows), rule=rules[c], sae_f1_median=round(float(np.median(s)), 4),
                       raw_f1_median=round(float(np.median(w)), 4),
                       sae_better_share=round(float(np.mean(s > w)), 4),
                       raw_f1_p10=round(float(np.percentile(w, 10)), 4),
                       sae_f1_p10=round(float(np.percentile(s, 10)), 4))
    json.dump(dict(summary=summ, rows=rows), open(FIXED_OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"sample": phase_sample, "analyze": phase_analyze, "best": phase_best,
     "fixed": phase_fixed}[sys.argv[1]]()
