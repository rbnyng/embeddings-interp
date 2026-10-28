"""Crop rotation: find corn and soy features in each Corn Belt tile's own dictionary and test them on held-out years.

The tiles are IOWA_TILE and those with the highest corn plus soy share in USDA CDL 2020, checked every 0.2 degrees
across the central Corn Belt, up to 100 in all. Each tile's dictionary (1,024 features, k = 16) is trained on 90,000
pixels per year, 2018-2024. At stride 2, corn features have a corn-minus-soy mean activation above 0.15 in every year
2018-2021, and soy features below -0.15. Each year a corn or soy pixel is called corn when its summed activation of
the tile's top 5 corn features is above that of the top 5 soy features, and soy otherwise. A tile's held-out accuracy
(held_acc) is the median of that accuracy over 2022-2024.

Usage: python experiments/crop_rotation.py select|fetch|screen|verdicts
Writes results/cornbelt_selection.json, crop_rotation.jsonl/.json and the dictionaries in data/checkpoints/cornbelt.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

from lib.config import (IOWA_TILE, CDL_CORN, CDL_SOY, GT_DIR,
                                   CKPT_DIR, RESULTS_DIR)
from lib.config import workers
from lib.tiles import (download_tile, download_landmask, load_tile,
                                       tile_grid, sample_pixels)
from lib.sae import encode_array, load_sae, train_and_save

YEARS = list(range(2018, 2025))
SCREEN_YEARS = [2018, 2019, 2020, 2021]
HELD_YEARS = [2022, 2023, 2024]
S = 2
N_TILES = 100
BOX = (-96.5, -88.5, 40.0, 44.0)
CDL_DIR = GT_DIR / "cdl"
SEL_PATH = RESULTS_DIR / "cornbelt_selection.json"
JSONL = RESULTS_DIR / "crop_rotation.jsonl"

def cdl_path(year):
    hits = (glob.glob(str(CDL_DIR / f"{year}_30m_cdls.*")) +
            glob.glob(str(CDL_DIR / f"**/{year}_30m_cdls.*"), recursive=True))
    hits = [h for h in hits if h.endswith((".img", ".tif"))]
    if not hits:
        raise FileNotFoundError(f"CDL {year}")
    return hits[0]

def read_cdl(tile, year):
    """Read the CDL classes around the tile and resample them onto the tile grid (nearest neighbor)."""
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    lon, lat = (float(v) for v in tile.split("_")[1:])
    bb = (lon - 0.055, lat - 0.055, lon + 0.055, lat + 0.055)
    dst_t, dst_crs, H, W = tile_grid(tile)
    with rasterio.open(cdl_path(year)) as src:
        wb = transform_bounds("EPSG:4326", src.crs, *bb)
        win = src.window(*wb)
        arr = src.read(1, window=win)
        wt = src.window_transform(win)
        crs = src.crs
    dst = np.zeros((H, W), np.float64)
    reproject(arr.astype(np.float64), dst, src_transform=wt, src_crs=crs,
              dst_transform=dst_t, dst_crs=dst_crs, resampling=Resampling.nearest)
    return dst.astype(np.uint8)

def phase_select():
    import rasterio
    from rasterio.warp import transform_bounds
    from lib.tiles import snap, tile_for_point
    lons = np.arange(BOX[0], BOX[1], 0.2)
    lats = np.arange(BOX[2], BOX[3], 0.2)
    scores = []
    with rasterio.open(cdl_path(2020)) as src:
        for lo in lons:
            for la in lats:
                t = tile_for_point(lo, la)
                bb = (snap(lo) - 0.05, snap(la) - 0.05, snap(lo) + 0.05, snap(la) + 0.05)
                try:
                    wb = transform_bounds("EPSG:4326", src.crs, *bb)
                    win = src.window(*wb)
                    a = src.read(1, window=win)
                except Exception:
                    continue
                if a.size < 1000:
                    continue
                frac = float(np.isin(a, (CDL_CORN, CDL_SOY)).mean())
                scores.append((frac, t))
    scores.sort(reverse=True)
    keep = [t for _, t in scores[:N_TILES - 1] if t != IOWA_TILE][:N_TILES - 1]
    tiles = [IOWA_TILE] + keep
    print("selected:", tiles)
    print("top fractions:", [round(f, 3) for f, _ in scores[:N_TILES]])
    json.dump(tiles, open(SEL_PATH, "w"))

def phase_fetch():
    tiles = json.load(open(SEL_PATH))
    jobs = [(t, y) for t in tiles for y in YEARS]
    def one(j):
        t, y = j
        try:
            download_tile(t, y)
            return None
        except Exception as e:
            return f"{t} {y}: {e!r}"
    with ThreadPoolExecutor(max_workers=24) as ex:
        for t in tiles:
            download_landmask(t)
        errs = [r for r in ex.map(one, jobs) if r]
    print(f"Tessera fetch done ({len(errs)} errors)")
    for e in errs[:8]:
        print(" ", e)

def run_tile(tile):
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    try:
        cdl = {y: read_cdl(tile, y) for y in YEARS}
        ck = CKPT_DIR / "cornbelt" / f"sae_{tile}.pt"
        if ck.exists():
            sae = load_sae(path=ck)
        else:
            X = sample_pixels([(tile, y) for y in YEARS], n_per=90_000, seed=0)
            sae = train_and_save(X, ck, threads=None, epochs=40, seed=0, log=False)
            sae.eval()
            del X

        F = {y: encode_array(sae, load_tile(tile, y, stride=S)) for y in YEARS}
        h, w = F[YEARS[0]].shape[:2]
        C = {y: cdl[y][::S, ::S][:h, :w] for y in YEARS}

        def screen(Fd):
            cp = np.array([Fd[y][C[y] == CDL_CORN].mean(0) for y in SCREEN_YEARS])
            sp = np.array([Fd[y][C[y] == CDL_SOY].mean(0) for y in SCREEN_YEARS])
            d = cp - sp
            return (np.where((d > 0.15).all(0))[0], np.where((d < -0.15).all(0))[0],
                    d.mean(0))

        corn, soy, d = screen(F)
        res = dict(ok=True, tile=tile, n_corn=int(corn.size), n_soy=int(soy.size))
        if corn.size and soy.size:
            tc = corn[np.argsort(d[corn])[::-1]][:5]
            ts = soy[np.argsort(-d[soy])[::-1]][:5]
            acc = {}
            for y in YEARS:
                m = np.isin(C[y], (CDL_CORN, CDL_SOY))
                pred = np.where(F[y][..., tc].sum(-1) > F[y][..., ts].sum(-1),
                                CDL_CORN, CDL_SOY)
                acc[str(y)] = float((pred[m] == C[y][m]).mean())
            res["acc"] = acc
            res["held_acc"] = float(np.median([acc[str(y)] for y in HELD_YEARS]))
        return res
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

def phase_screen():
    tiles = json.load(open(SEL_PATH))
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                done.add(json.loads(line)["tile"])
            except Exception:
                pass
    todo = [t for t in tiles if t not in done]
    print(f"{len(todo)} tiles to run", flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(50)) as ex:
        for r in ex.map(run_tile, todo):
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(" ", {k: v for k, v in r.items() if k not in ("acc",)}, flush=True)
    out.close()
    verdicts()

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    both = sum(r["n_corn"] > 0 and r["n_soy"] > 0 for r in rows)
    hac = [r["held_acc"] for r in rows if "held_acc" in r]
    print(f"\n{len(rows)} tiles, {both} with both corn and soy features")
    print(f"median held-out (2022-24) accuracy: {np.median(hac) if hac else float('nan'):.3f}")
    json.dump(dict(rows=rows), open(RESULTS_DIR / "crop_rotation.json", "w"), default=float)
    print("saved results/crop_rotation.json")

if __name__ == "__main__":
    {"select": phase_select, "fetch": phase_fetch,
     "screen": phase_screen, "verdicts": verdicts}[sys.argv[1]]()
