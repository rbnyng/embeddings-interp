"""Look for a precursor: can the embedding tell European forest about to be disturbed from forest that is not?

Units are tile-years (Y = 2021 or 2022) on Europe-atlas tiles with EFDA wind or bark beetle damage (agent 1), encoded
with the Europe-atlas dictionary. Groups are at stride 2. Targets need at least 150 pixels, the control pool 1,000.
  pre  forest with EFDA wind or bark beetle damage in Y
  ctl  forest with no EFDA disturbance from Y-5 through 2023, kept within the 10th-90th percentile of the targets'
       distance to disturbance in Y-5 to Y-1 (the whole pool if fewer than 150 pixels are kept)
A gap is the family's mean activation on pre minus ctl, a set number of years (depth) before Y.
  family  inverted screen: features whose median Y-2 gap on the 2021 units clears the anchored bar (5 tiles or more)
  q2      on the 2022 units, the share of tiles with a positive Y-2 gap, and the median gap with a 95% interval from
          resampling tiles (q2_median, q2_ci)
  q5      the median 2022 gap at Y-4 through Y-0

Usage: python experiments/precursor.py screen|verdicts
Writes results/precursor.jsonl and results/precursor.json.
"""
import json
import os
import sys

import numpy as np
import torch

from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.tiles import load_tile, tile_grid
import europe_atlas
from lib.sae import group_sums, load_sae
from lib.screen import ANCHOR
YEARS = list(range(2017, 2026))
S = 2
M = 1024
CKPT = CKPT_DIR / "europe_atlas" / "sae_europe.pt"
JSONL = RESULTS_DIR / "precursor.jsonl"
EFDA = GT_DIR / "efda"
EVENT_YEARS = (2021, 2022)          # the embeddings go back to Y-4 for both years
AGENTS = {"wind_beetle": 1}
MIN_PX = 150

def tile_layers(tile):
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    countries = sorted({m["country"] for m in europe_atlas.TILE_META[tile]})
    lon, lat = (float(v) for v in tile.split("_")[1:])
    bb = (lon - 0.055, lat - 0.055, lon + 0.055, lat + 0.055)
    dst_t, dst_crs, H, W = tile_grid(tile)
    yrs = list(range(2017, 2024))
    DYRS = list(range(2012, 2024))
    agent = np.zeros((len(yrs), H, W), np.uint8)
    dist = {y: np.zeros((H, W), bool) for y in DYRS}
    forest = np.zeros((H, W), bool)
    for c in countries:
        try:
            with rasterio.open(EFDA / f"disturbance_agent_1985_2023_{c}.tif") as src:
                wb = transform_bounds("EPSG:4326", src.crs, *bb)
                win = src.window(*wb)
                if win.width < 2 or win.height < 2:
                    continue
                ag = src.read([y - 1984 for y in yrs], window=win)
                wt, crs = src.window_transform(win), src.crs
            with rasterio.open(EFDA / f"annual_disturbances_1985_2023_{c}.tif") as src:
                ad_win = src.read([y - 1984 for y in DYRS], window=win) == 1
            with rasterio.open(EFDA / f"forest_mask_{c}.tif") as src:
                fo = src.read(1, window=win) == 1
        except Exception:
            continue

        def grid(a):
            d = np.zeros((H, W))
            reproject(a.astype(float), d, src_transform=wt, src_crs=crs,
                      dst_transform=dst_t, dst_crs=dst_crs,
                      resampling=Resampling.nearest)
            return d

        for i in range(len(yrs)):
            g = grid(ag[i])
            agent[i] = np.maximum(agent[i], (g * (g < 250)).astype(np.uint8))
        for i, y in enumerate(DYRS):
            dist[y] |= grid(ad_win[i]) > 0.5
        forest |= grid(fo) > 0.5
    return agent, dist, forest, yrs

def run_tile(tile):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        agent, dist, forest, yrs = tile_layers(tile)
        yi = {y: i for i, y in enumerate(yrs)}
        from scipy import ndimage
        groups = {}
        meta = {}
        for Y in EVENT_YEARS:
            # Disturbance before Y-5 is allowed, so controls share the targets' history and are not untouched forest.
            near = np.zeros_like(forest)
            for y in range(Y - 5, 2024):
                if y in dist:
                    near |= dist[y]
            pool = forest & ~near
            if pool[::S, ::S].sum() < 1000:
                continue
            for name, code in AGENTS.items():
                tgt = (agent[yi[Y]] == code) & forest
                if tgt[::S, ::S].sum() < MIN_PX:
                    continue
                prior = np.zeros_like(forest)
                for y in range(Y - 5, Y):
                    if y in dist:
                        prior |= dist[y]
                prior &= ~tgt
                d2p = (ndimage.distance_transform_edt(~prior) if prior.any()
                       else np.full(forest.shape, 1e4, np.float32))
                dsel = d2p[tgt]
                lo, hi = np.percentile(dsel, [10, 90])
                matched = pool & (d2p >= lo) & (d2p <= hi)
                ok_match = bool(matched[::S, ::S].sum() >= MIN_PX)
                groups[f"{name}_{Y}_pre"] = tgt
                groups[f"{name}_{Y}_ctl"] = matched if ok_match else pool
                meta[f"{name}_{Y}_matched"] = ok_match
                meta[f"{name}_{Y}_n"] = int(tgt[::S, ::S].sum())
                meta[f"{name}_{Y}_meddist"] = float(np.median(dsel))
                meta[f"{name}_{Y}_ctl_meddist"] = float(np.median(d2p[matched])) \
                    if ok_match else None
        if not groups:
            return dict(ok=False, tile=tile, error="no disturbed groups")

        sae = load_sae(path=CKPT)
        gs = {k: v[::S, ::S] for k, v in groups.items()}
        T = {}
        for y in YEARS:
            try:
                x = load_tile(tile, y, stride=S)
            except Exception:
                continue
            s, n = group_sums(sae, x, gs)
            T[y] = {k: (s[k] / n[k]).tolist() for k in gs if n[k] > 0}
            del x
        return dict(ok=True, tile=tile, lat=float(tile.split("_")[2]),
                    meta=meta, tables={str(y): T[y] for y in T})
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

def phase_screen():
    from concurrent.futures import ProcessPoolExecutor
    rows = [r for r in map(json.loads, open(RESULTS_DIR / "europe_atlas.jsonl"))
            if r.get("ok")]
    tiles = sorted({r["tile"] for r in rows
                    if any(f"wind_beetle_{Y}" in r["tables"].get("2017", {})
                           for Y in EVENT_YEARS)})
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                done.add(json.loads(line)["tile"])
            except Exception:
                pass
    todo = [t for t in tiles if t not in done]
    print(f"{len(todo)} tiles to screen (of {len(tiles)} with beetle tile-years)",
          flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(40)) as ex:
        for i, r in enumerate(ex.map(run_tile, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    verdicts()

def contrast(r, Y, agent, depth, fam):
    y = str(Y - depth)
    T = r["tables"].get(y, {})
    a, b = f"{agent}_{Y}_pre", f"{agent}_{Y}_ctl"
    if a not in T or b not in T:
        return None
    return float(np.mean([T[a][f] for f in fam]) - np.mean([T[b][f] for f in fam]))

def inverted_screen(rows, Y=2021, agent="wind_beetle", depth=2):
    per = []
    for r in rows:
        T = r["tables"]
        a, b = f"{agent}_{Y}_pre", f"{agent}_{Y}_ctl"
        ys = [str(Y - d) for d in (depth, depth + 1, depth + 2)]
        if not all(y in T and a in T[y] and b in T[y] for y in ys):
            continue
        per.append(np.mean([[T[y][a][f] - T[y][b][f] for f in range(M)]
                            for y in ys[:1]], axis=0))
    if len(per) < 5:
        return [], len(per)
    A = np.median(per, axis=0)
    act = np.abs(A) > 0
    bar = ANCHOR * np.median(np.abs(A[act])) if act.any() else np.inf
    return [int(f) for f in np.where(A > bar)[0]], len(per)

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    print(f"\n{len(rows)} tiles screened")
    fam, nscr = inverted_screen(rows, Y=2021, agent="wind_beetle", depth=2)
    print(f"inverted screen at Y-2 on the 2021 wind/beetle units ({nscr} tiles): {len(fam)} features")
    if not fam:
        json.dump(dict(family=[], outcome="no precursor family at depth 2",
                       n_screen=nscr),
                  open(RESULTS_DIR / "precursor.json", "w"))
        print("\nNo precursor family found.")
        return

    def tile_vals(depth):
        """Return the per-tile gaps on the 2022 wind/beetle units, `depth` years before the event."""
        return [c for c in (contrast(r, 2022, "wind_beetle", depth, fam) for r in rows)
                if c is not None]

    v = tile_vals(2)
    q2 = float(np.mean([x > 0 for x in v])) if v else float("nan")

    # Bootstrap over tiles. The share of positive tiles says nothing about the size of the effect.
    rng = np.random.default_rng(0)
    va = np.array(v)
    boot = np.array([np.median(rng.choice(va, len(va), replace=True))
                     for _ in range(5000)]) if len(va) else np.array([np.nan])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"2022 units, Y-2 gap: positive in {q2:.3f} of {len(v)} tiles; "
          f"median {np.median(va):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")

    prof = {d: float(np.median(tile_vals(d) or [np.nan])) for d in (4, 3, 2, 1, 0)}
    print("median 2022 gap by depth: " +
          "  ".join(f"Y-{d}:{prof[d]:+.4f}" for d in (4, 3, 2, 1, 0)))
    json.dump(dict(family=fam, n_screen=nscr, q2=q2,
                   q2_ci=[float(lo), float(hi)], q2_median=float(np.median(va)), q5=prof),
              open(RESULTS_DIR / "precursor.json", "w"), default=float)
    print("\nsaved results/precursor.json")

if __name__ == "__main__":
    {"screen": phase_screen, "verdicts": verdicts}[sys.argv[1]]()
