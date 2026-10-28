"""Encode Europe-atlas tiles with the Dixie dictionary, for the specificity test on harvest.

Tiles come from three boxes with full Tessera coverage for 2017-2025, 12 per box: the Sumava box ranked by EFDA wind/bark
beetle share in 2018-2020, the Vaia belt by FORWIND Vaia damage share, and lowland Germany by EFDA harvest share in
2018-2022. Units are tile-years on the EFDA forest mask. Each table is every feature's mean activation over each pixel
group, for each year 2017-2025. Groups need at least 800 pixels at stride 2 (2,000 for intact).
  harvest_Y  EFDA harvest agent in year Y, 2018-2022
  intact     forest with no EFDA disturbance in 1985-2023

Usage: python experiments/europe_agents_dixie.py coverage|select|masks|screen
Writes results/efda_tile_candidates.json, efda_tile_selection.json and europe_agents_dixie.json, and caches the masks
in data/groundtruth/efda_masks.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import GT_DIR, RESULTS_DIR, workers
from lib.tiles import tile_grid
from lib.sae import load_sae
from lib.trajectories import group_tables

EFDA = GT_DIR / "efda"
MASKS = GT_DIR / "efda_masks"
YEARS = list(range(2017, 2026))
S = 2
_SEL_PATH = RESULTS_DIR / "efda_tile_selection.json"
SEL = json.load(open(_SEL_PATH)) if _SEL_PATH.exists() else []
BOX_COUNTRIES = {"sumava_beetle": ["czechia", "germany", "austria"],
                 "vaia_belt": ["italy", "austria"],
                 "lowland_managed": ["germany"]}
BAND = lambda y: y - 1984
HARVEST_YEARS = range(2018, 2023)

def tiles_of(box):
    return [d["tile"] for d in SEL[box]]

def build_masks(job):
    """Save one tile's pixel-group masks, at full tile resolution, to an npz cache."""
    box, tile = job
    out = MASKS / f"{box}_{tile}.npz"
    if out.exists():
        return f"{tile} cached"
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    dst_t, dst_crs, H, W = tile_grid(tile)
    lon, lat = (float(v) for v in tile.split("_")[1:])
    bb4326 = (lon - 0.055, lat - 0.055, lon + 0.055, lat + 0.055)
    agent = np.zeros((len(range(2017, 2024)), H, W), np.uint8)   # years 2017-2023
    dist_any = np.zeros((H, W), bool)
    forest = np.zeros((H, W), bool)
    got = False
    for c in BOX_COUNTRIES[box]:
        try:
            with rasterio.open(EFDA / f"disturbance_agent_1985_2023_{c}.tif") as src:
                wb = transform_bounds("EPSG:4326", src.crs, *bb4326)
                win = src.window(*wb)
                if win.width < 2 or win.height < 2:
                    continue
                ag = src.read([BAND(y) for y in range(2017, 2024)], window=win)
                wt = src.window_transform(win)
                src_crs = src.crs
            with rasterio.open(EFDA / f"annual_disturbances_1985_2023_{c}.tif") as src:
                anyd = (src.read(window=win) == 1).any(axis=0).astype(np.uint8)
            with rasterio.open(EFDA / f"forest_mask_{c}.tif") as src:
                fo = src.read(1, window=win)
        except Exception:
            continue
        for i in range(ag.shape[0]):
            d = np.zeros((H, W), np.float64)
            reproject(ag[i].astype(np.float64), d, src_transform=wt, src_crs=src_crs,
                      dst_transform=dst_t, dst_crs=dst_crs,
                      resampling=Resampling.nearest)
            agent[i] = np.maximum(agent[i], d.astype(np.uint8) * (d < 250))
        for srcarr, dstarr in ((anyd, dist_any), (fo == 1, forest)):
            d = np.zeros((H, W), np.float64)
            reproject(srcarr.astype(np.float64), d, src_transform=wt, src_crs=src_crs,
                      dst_transform=dst_t, dst_crs=dst_crs,
                      resampling=Resampling.nearest)
            dstarr |= d > 0.5
        got = True
    if not got:
        return f"{tile} NO EFDA DATA"

    yi = {y: i for i, y in enumerate(range(2017, 2024))}
    groups = {"intact": forest & ~dist_any}
    for y in HARVEST_YEARS:
        groups[f"harvest_{y}"] = (agent[yi[y]] == 3) & forest
    MASKS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: v for k, v in groups.items()})
    sizes = {k: int(v[::S, ::S].sum()) for k, v in groups.items()}
    return f"{tile} {sizes}"

def phase_masks():
    jobs = [(box, t) for box in SEL for t in tiles_of(box)]
    with ProcessPoolExecutor(max_workers=workers(36)) as ex:
        for msg in ex.map(build_masks, jobs):
            print(" ", msg, flush=True)
    print("masks done")

def tile_tables(job):
    box, tile = job
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    z = np.load(MASKS / f"{box}_{tile}.npz")
    groups = {k: z[k] for k in ["intact"] + [f"harvest_{y}" for y in HARVEST_YEARS] if k in z.files}
    groups = {k: v for k, v in groups.items() if v[::S, ::S].sum() >= (2000 if k == "intact" else 800)}
    if "intact" not in groups or len(groups) < 2:
        return None
    sae = load_sae()
    T = group_tables(sae, tile, YEARS, groups, stride=S)
    return dict(box=box, tile=tile,
                tables={str(y): {g: T[y][g].tolist() for g in T[y]} for y in YEARS})

def phase_screen():
    jobs = [(box, t) for box in SEL for t in tiles_of(box)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers(72)) as ex:
        for i, r in enumerate(ex.map(tile_tables, jobs), 1):
            if r:
                rows.append(r)
            if i % 12 == 0:
                print(f"  [{i}/{len(jobs)}]", flush=True)
    json.dump(rows, open(RESULTS_DIR / "europe_agents_dixie.json", "w"))
    for box in SEL:
        print(f"{box:16s} {sum(r['box'] == box for r in rows)} tile-tables")
    print(f"{len(rows)} tile-tables saved to results/europe_agents_dixie.json", flush=True)

def phase_select():
    import geopandas as gpd
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box as sbox
    EFDA = GT_DIR / "efda"
    CAND = json.load(open(RESULTS_DIR / "efda_tile_candidates.json"))
    BAND = lambda y: y - 1984
    COUNTRIES = {"sumava_beetle": ["czechia", "germany"],
                 "vaia_belt": ["italy", "austria"],
                 "lowland_managed": ["germany"]}

    def tile_lonlat_bbox(tile):
        lon, lat = (float(v) for v in tile.split("_")[1:])
        return lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05

    def agent_fraction(tile, countries, agent, years):
        bb = tile_lonlat_bbox(tile)
        best = 0.0
        for c in countries:
            try:
                with rasterio.open(f"{EFDA}/disturbance_agent_1985_2023_{c}.tif") as src:
                    l, b, r, t = transform_bounds("EPSG:4326", src.crs, *bb)
                    win = src.window(l, b, r, t)
                    a = src.read([BAND(y) for y in years], window=win)
                with rasterio.open(f"{EFDA}/forest_mask_{c}.tif") as src:
                    f = src.read(1, window=win)
                forest = (f == 1)
                if forest.sum() < 1000:
                    continue
                hit = np.any(a == agent, axis=0) & forest
                best = max(best, float(hit.sum() / forest.sum()))
            except Exception:
                continue
        return best

    print("loading FORWIND (Vaia window)...", flush=True)
    fw = gpd.read_file(GT_DIR / "forwind" / "FORWIND_v2.shp")
    datecol = next(c for c in fw.columns if "date" in c.lower() or "Date" in c)
    fw["d"] = gpd.pd.to_datetime(fw[datecol], errors="coerce")
    vaia = fw[(fw["d"] >= "2018-10-25") & (fw["d"] <= "2018-11-05")].to_crs("EPSG:4326")
    vaia.geometry = vaia.geometry.make_valid()
    print(f"Vaia polygons: {len(vaia)}", flush=True)

    def vaia_fraction(tile):
        bb = tile_lonlat_bbox(tile)
        cell = sbox(*bb)
        hits = vaia[vaia.geometry.intersects(cell)]
        if len(hits) == 0:
            return 0.0
        from shapely.ops import unary_union
        u = unary_union(hits.geometry.values)
        return float(u.intersection(cell).area / cell.area)

    selection = {}
    for boxname, tiles in CAND.items():
        scores = []
        for t in tiles:
            if boxname == "vaia_belt":
                s = vaia_fraction(t)
            elif boxname == "sumava_beetle":
                s = agent_fraction(t, COUNTRIES[boxname], 1, range(2018, 2021))
            else:
                s = agent_fraction(t, COUNTRIES[boxname], 3, range(2018, 2023))
            scores.append((s, t))
        scores.sort(reverse=True)
        selection[boxname] = [dict(tile=t, score=round(s, 4)) for s, t in scores[:12]]
        print(f"\n{boxname}: top tiles")
        for s, t in scores[:12]:
            print(f"   {t}  {s:.4f}")

    json.dump(selection, open(RESULTS_DIR / "efda_tile_selection.json", "w"))
    print("\nsaved results/efda_tile_selection.json")

def phase_coverage():
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from lib.tiles import embedding_url
    YEARS = list(range(2017, 2026))
    BOXES = {"sumava_beetle": (13.0, 14.0, 48.6, 49.2),
             "vaia_belt": (11.3, 13.7, 46.1, 46.8),
             "lowland_managed": (13.2, 14.5, 51.5, 52.2)}

    from lib.tiles import tile_for_point

    def head(job):
        t, y = job
        try:
            r = requests.head(embedding_url(t, y), timeout=30,
                              allow_redirects=True)
            return t, y, r.status_code == 200
        except Exception:
            return t, y, False

    out = {}
    for name, (lo0, lo1, la0, la1) in BOXES.items():
        tiles = sorted({tile_for_point(lo, la)
                        for lo in np.arange(lo0, lo1, 0.1)
                        for la in np.arange(la0, la1, 0.1)})
        cov = {t: set() for t in tiles}
        with ThreadPoolExecutor(max_workers=24) as ex:
            for t, y, ok in ex.map(head, [(t, y) for t in tiles for y in YEARS]):
                if ok:
                    cov[t].add(y)
        full = [t for t, ys in cov.items() if len(ys) == len(YEARS)]
        out[name] = full
        print(f"{name:16s} {len(full)}/{len(tiles)} tiles full {YEARS[0]}-{YEARS[-1]}",
              flush=True)

    json.dump(out, open(RESULTS_DIR / "efda_tile_candidates.json", "w"))
    print("saved results/efda_tile_candidates.json")

if __name__ == "__main__":
    {"coverage": phase_coverage, "select": phase_select, "masks": phase_masks,
     "screen": phase_screen}[sys.argv[1]]()
