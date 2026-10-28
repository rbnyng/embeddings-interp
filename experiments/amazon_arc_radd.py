"""Test the Amazon arc clearing family (screened on 2018 Hansen units) plot by plot against RADD alert dates.

Reference product: RADD forest-disturbance alerts. On each Amazon arc tile, high-confidence alert pixels (confidence 3)
dated 2020-01-01 to 2023-12-31 are grouped into connected plots of at least 13 stride-2 pixels (about 0.5 ha). A plot's
year Y is the year of its median alert date. Its peak year is the year in 2017-2025 when the family's mean activation
over the plot is highest. A tile needs AlphaEarth for at least 7 of those years, including 2024. `verdicts` reports:
  size bands  for plots under 1, 1-3 and at least 3 ha, the share that peak in Y (exact) and in Y or Y+1 (window)
  sub-annual  among plots of at least 0.5 ha, the share that peak in Y for plots dated January-June minus plots dated
              July-December, with a 95% interval from resampling tiles

Usage: python experiments/amazon_arc_radd.py fetch|screen|verdicts
Writes results/amazon_arc_radd.jsonl and amazon_arc_radd.json, and caches RADD in data/groundtruth/radd_arc.
Run amazon_arc.py first.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.alphaearth import compute_scale, load_aef
from lib.groundtruth import fetch_radd, radd_decode
from lib.sae import encode_array, load_sae

YEARS = list(range(2017, 2026))
S = 2
M = 512
HIGH_CONF = 3
DATE_LO, DATE_HI = np.datetime64("2020-01-01"), np.datetime64("2023-12-31")
MIN_PX = 13                      # in stride-2 pixels, 13 of 0.04 ha each is about 0.5 ha
HA_PER_PX = 0.04                 # 20 m pixels
RADD_DIR = GT_DIR / "radd_arc"
JSONL = RESULTS_DIR / "amazon_arc_radd.jsonl"
# Empty until amazon_arc.py has written its selection and family, so the module still imports on a fresh checkout.
_SEL_PATH, _ARC_PATH = RESULTS_DIR / "amazon_arc_selection.json", RESULTS_DIR / "amazon_arc.json"
SEL = json.load(open(_SEL_PATH)) if _SEL_PATH.exists() else []
TILES = sorted({m["tile"]: m for m in SEL})
FAMILY = [int(f) for f in json.load(open(_ARC_PATH))["family"]] if _ARC_PATH.exists() else []

def granule(tile):
    lon, lat = (float(v) for v in tile.split("_")[1:])
    lo = int(np.floor(lon / 10) * 10)
    la = int(np.ceil(lat / 10) * 10)
    return (f"{abs(la):02d}{'S' if la < 0 else 'N'}_"
            f"{abs(lo):03d}{'W' if lo < 0 else 'E'}")

def radd_path(tile):
    return RADD_DIR / f"{tile}.npz"

def get_radd(tile):
    p = radd_path(tile)
    if p.exists():
        return np.load(p)["radd"]
    arr = fetch_radd(tile, granule(tile))[::S, ::S].astype(np.uint32)
    RADD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, radd=arr)
    return arr

def phase_fetch():
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            get_radd(t)
            return None
        except Exception as e:
            return f"{t}: {e!r}"

    errs = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(one, TILES), 1):
            if r:
                errs.append(r)
            if i % 20 == 0:
                print(f"  [{i}/{len(TILES)}] ({len(errs)} errors)", flush=True)
    print(f"RADD fetch done ({len(errs)} errors)", flush=True)
    for e in errs[:8]:
        print(" ", e)

def run_tile(args):
    tile, scale = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from scipy import ndimage
        radd = get_radd(tile)
        conf, dates = radd_decode(radd)
        alert = (radd > 0) & (conf == HIGH_CONF) & (dates >= DATE_LO) \
            & (dates <= DATE_HI)
        if int(alert.sum()) < MIN_PX:
            return dict(ok=False, tile=tile, error="no alerts")
        lab, n = ndimage.label(alert)
        sizes = np.bincount(lab.ravel())
        keep = [p for p in range(1, n + 1) if sizes[p] >= MIN_PX]
        if not keep:
            return dict(ok=False, tile=tile, error="no plots >= floor")
        day = dates.astype("datetime64[D]").astype(int)
        meta = {}
        for p in keep:
            m = lab == p
            d = np.datetime64(int(np.median(day[m])), "D")
            meta[p] = dict(px=int(sizes[p]), year=int(str(d)[:4]),
                           month=int(str(d)[5:7]))
        fam_t = np.zeros((len(YEARS), n + 1))
        got = []
        sae = load_sae(path=CKPT_DIR / "amazon_arc" / "sae_arc_aef.pt", d=64, m=M, k=8)
        for yi, y in enumerate(YEARS):
            try:
                x = np.nan_to_num(load_aef(tile, y, stride=S)) * scale
            except Exception:
                continue
            got.append(y)
            H, W = x.shape[:2]
            for r0 in range(0, H, 192):
                r1 = min(r0 + 192, H)
                f = encode_array(sae, x[r0:r1])[:, :, FAMILY]
                ll = lab[r0:r1, :W].ravel()
                np.add.at(fam_t[yi], ll, f.mean(-1).ravel())
            del x
        if len(got) < 7 or 2024 not in got:
            return dict(ok=False, tile=tile, error=f"years missing: {got}")
        for p in keep:
            fam_t[:, p] /= sizes[p]
        yr = np.array(YEARS)
        out = []
        for p in keep:
            pf = int(yr[np.argmax(fam_t[:, p])])
            out.append(dict(px=meta[p]["px"], ha=round(meta[p]["px"] * HA_PER_PX, 2),
                            year=meta[p]["year"], month=meta[p]["month"], peak=pf))
        lon, lat = (float(v) for v in tile.split("_")[1:])
        return dict(ok=True, tile=tile, lon=lon, lat=lat, plots=out)
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

def phase_screen():
    from concurrent.futures import ProcessPoolExecutor
    scale = compute_scale()
    print(f"SCALE x{scale:.1f}; family {len(FAMILY)} features", flush=True)
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                done.add(json.loads(line)["tile"])
            except Exception:
                pass
    todo = [t for t in TILES if t not in done]
    print(f"{len(todo)} tiles to screen", flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for i, r in enumerate(ex.map(run_tile, [(t, scale) for t in todo]), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    verdicts()

def clustered_p(by_tile, stat, B=5000, seed=0):
    """Return `stat` over all plots and its 95% interval from resampling tiles. Not a p-value, despite the name."""
    rng = np.random.default_rng(seed)
    tiles = list(by_tile)
    obs = stat([p for t in tiles for p in by_tile[t]])
    null = np.empty(B)
    for b in range(B):
        pick = rng.choice(tiles, len(tiles), replace=True)
        null[b] = stat([p for t in pick for p in by_tile[t]])
    lo, hi = np.percentile(null, [2.5, 97.5])
    return obs, lo, hi

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    P = [dict(p, tile=r["tile"]) for r in rows for p in r["plots"]]
    print(f"\n{len(rows)} tiles with plots; {len(P)} plots total")

    def acc(sub, win=(0, 1)):
        return float(np.mean([p["peak"] - p["year"] in win for p in sub])) \
            if sub else float("nan")

    def exact(sub):
        return acc(sub, win=(0,))

    bands = [("<1 ha", [p for p in P if p["ha"] < 1]),
             ("1-3 ha", [p for p in P if 1 <= p["ha"] < 3]),
             (">=3 ha", [p for p in P if p["ha"] >= 3])]
    for n, s in bands:
        print(f"{n}: peak in Y {exact(s):.3f}, in Y or Y+1 {acc(s):.3f} (n={len(s)})")

    half = [p for p in P if p["ha"] >= 0.5]
    early = [p for p in half if p["month"] <= 6]
    late = [p for p in half if p["month"] >= 7]
    by_tile = defaultdict(list)
    for p in half:
        by_tile[p["tile"]].append(p)

    def diffstat(sub):
        e = [p for p in sub if p["month"] <= 6]
        l = [p for p in sub if p["month"] >= 7]
        if not e or not l:
            return np.nan
        return exact(e) - exact(l)

    obs, lo, hi = clustered_p(by_tile, diffstat)
    print(f"peak in Y, plots of at least 0.5 ha: January-June {exact(early):.3f} (n={len(early)}), "
          f"July-December {exact(late):.3f} (n={len(late)}); difference {obs:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    json.dump(dict(n_plots=len(P), n_tiles=len(rows),
                   size_bands={n: dict(n=len(s), exact=exact(s), window=acc(s)) for n, s in bands},
                   subannual_diff=obs, subannual_ci=[lo, hi]),
              open(RESULTS_DIR / "amazon_arc_radd.json", "w"), default=float)
    print("\nsaved results/amazon_arc_radd.json")

if __name__ == "__main__":
    {"fetch": phase_fetch, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
