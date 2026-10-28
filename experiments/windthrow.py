"""Windthrow: find a family on Friederike tiles and test it on held-out tiles from three FORWIND storms.

Reference product: FORWIND v2 storm-damage polygons (event-dated). Each unit is a tile with at least 25 ha of one
storm's damage, keeping the 60 largest tiles per storm. Disturbed pixels are in that storm's polygons. Undisturbed
pixels are EFDA forest with no disturbance in 1985-2023, outside every FORWIND polygon. Groups are at stride 2 and need
at least 300 and 2,000 pixels. Storm, FORWIND date, storm year Y and the countries whose EFDA forest maps are read:
  Friederike  2018-01-18  Y = 2018  Germany
  Xavier      2017-11-10  Y = 2017  Germany
  Vaia        2018-10-28  Y = 2018  Italy, Austria, Slovenia, Switzerland
One dictionary is trained on all selected tiles. The family is screened on 12 random Friederike tiles with the
anchored bar and expected window 2018-2019. All other tiles are held out. A held-out tile validates when the family's
mean damaged minus undisturbed gap peaks in Y or Y+1.

Usage: python experiments/windthrow.py select|fetch|train|screen|verdicts
Writes results/windthrow_selection.json, windthrow.jsonl and windthrow.json.
Caches the dictionary at data/checkpoints/windthrow/sae_windthrow.pt.
"""
import json
import os
import sys

import numpy as np
import torch

from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.tiles import load_tile, tile_grid, sample_pixels
from lib.screen import anchored_family
from lib.sae import group_sums, load_sae, train_and_save

YEARS = list(range(2017, 2026))
S = 2
M = 1024
CKPT = CKPT_DIR / "windthrow" / "sae_windthrow.pt"
JSONL = RESULTS_DIR / "windthrow.jsonl"
_SEL_PATH = RESULTS_DIR / "windthrow_selection.json"
# Empty until `select` has run, so that phase still works on a fresh checkout.
SEL = json.load(open(_SEL_PATH)) if _SEL_PATH.exists() else []
EVENT_YEAR = dict(friederike=2018, xavier=2017, vaia=2018)
COUNTRIES = dict(friederike=("germany",), xavier=("germany",),
                 vaia=("italy", "austria", "slovenia", "switzerland"))
EFDA = GT_DIR / "efda"

rng0 = np.random.default_rng(0)
FRIED = sorted(r["tile"] for r in SEL if r["storm"] == "friederike")
SCREEN_TILES = set(rng0.choice(FRIED, size=12, replace=False)) if FRIED else set()

def phase_fetch():
    from concurrent.futures import ThreadPoolExecutor
    from lib.tiles import download_tile, download_landmask
    def one(t):
        try:
            download_landmask(t)
            for y in YEARS:
                download_tile(t, y)
            return t, ""
        except Exception as e:
            return t, repr(e)
    tiles = sorted({r["tile"] for r in SEL})
    with ThreadPoolExecutor(8) as ex:
        bad = [(t, e) for t, e in ex.map(one, tiles) if e]
    print(f"{len(tiles) - len(bad)}/{len(tiles)} tiles complete; missing: {[t for t, _ in bad]}")

def phase_train():
    if CKPT.exists():
        print("cached"); return
    rng = np.random.default_rng(0)
    tiles = sorted({r["tile"] for r in SEL})
    X = np.concatenate([sample_pixels([(t, y) for y in YEARS], n_per=9000, seed=0)
                        for t in tiles])
    rng.shuffle(X)
    print(f"storm-belt corpus {X.shape}", flush=True)
    train_and_save(X, CKPT, epochs=40, seed=0, log=True)
    print("trained", flush=True)

def tile_masks(rec):
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from rasterio.warp import reproject, Resampling, transform_bounds
    tile = rec["tile"]
    lon, lat = rec["lon"], rec["lat"]
    t, crs, H, W = tile_grid(tile)
    fw = gpd.read_file(GT_DIR / "forwind" / "FORWIND_v2.shp",
                       bbox=(lon - 0.06, lat - 0.06, lon + 0.06, lat + 0.06))
    mine = fw[fw.StormName.astype(str).str.lower() == rec["storm"]]
    if not len(mine):
        raise RuntimeError("no storm polygons in bbox")
    dmg = features.rasterize(((g, 1) for g in mine.to_crs(crs).geometry),
                             out_shape=(H, W), transform=t, fill=0,
                             dtype=np.uint8).astype(bool)
    anyfw = features.rasterize(((g, 1) for g in fw.to_crs(crs).geometry),
                               out_shape=(H, W), transform=t, fill=0,
                               dtype=np.uint8).astype(bool)
    bb = (lon - 0.055, lat - 0.055, lon + 0.055, lat + 0.055)
    forest = np.zeros((H, W), bool)
    dist = np.zeros((H, W), bool)
    for c in COUNTRIES[rec["storm"]]:         # a tile can straddle a border
        try:
            with rasterio.open(EFDA / f"forest_mask_{c}.tif") as src:
                wb = transform_bounds("EPSG:4326", src.crs, *bb)
                win = src.window(*wb)
                if win.width < 2 or win.height < 2:
                    continue
                fo = src.read(1, window=win) == 1
                wt, scrs = src.window_transform(win), src.crs
            with rasterio.open(EFDA / f"annual_disturbances_1985_2023_{c}.tif") as src:
                ad = (src.read(window=win) == 1).any(0)
        except Exception:
            continue
        if fo.size == 0 or ad.size == 0:      # the tile barely touches this country's raster
            continue
        for sarr, darr in ((fo, forest), (ad, dist)):
            d = np.zeros((H, W))
            reproject(sarr.astype(float), d, src_transform=wt, src_crs=scrs,
                      dst_transform=t, dst_crs=crs, resampling=Resampling.nearest)
            darr |= d > 0.5
    intact = forest & ~dist & ~anyfw
    return dmg[::S, ::S], intact[::S, ::S]

def run_tile(rec):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        dmg, intact = tile_masks(rec)
        if dmg.sum() < 300 or intact.sum() < 2000:
            return dict(ok=False, tile=rec["tile"], error="mask too small")
        sae = load_sae(path=CKPT)
        td, ti = {}, {}
        for y in YEARS:
            try:
                x = load_tile(rec["tile"], y, stride=S)
            except Exception:
                continue
            s, n = group_sums(sae, x, {"dmg": dmg, "ref": intact})
            td[y] = (s["dmg"] / n["dmg"]).tolist()
            ti[y] = (s["ref"] / n["ref"]).tolist()
            del x
        ey = EVENT_YEAR[rec["storm"]]
        if not {ey, ey + 1} <= set(td):
            return dict(ok=False, tile=rec["tile"], error="gap at window")
        return dict(ok=True, tile=rec["tile"], storm=rec["storm"],
                    ha=rec["ha"], screen=rec["tile"] in SCREEN_TILES,
                    dmg={str(y): v for y, v in td.items()},
                    ref={str(y): v for y, v in ti.items()})
    except Exception as e:
        return dict(ok=False, tile=rec["tile"], storm=rec["storm"], error=repr(e))

def phase_screen():
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                done.add(json.loads(line)["tile"])
            except Exception:
                pass
    todo = [r for r in SEL if r["tile"] not in done]
    print(f"{len(todo)} tiles to screen", flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(41)) as ex:
        for i, r in enumerate(ex.map(run_tile, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    verdicts()

def contrast_traj(r, fam):
    return {int(y): float(np.mean([r["dmg"][y][f] for f in fam])
                          - np.mean([r["ref"][y][f] for f in fam]))
            for y in r["dmg"]}

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    scr = [r for r in rows if r["screen"]]
    held = [r for r in rows if not r["screen"]]

    # Screen on the 12 screening tiles. The expected window is 2018-2019.
    A = {y: np.median([np.array(r["dmg"][str(y)]) - np.array(r["ref"][str(y)])
                       for r in scr], axis=0) for y in YEARS}
    fam = anchored_family({y - 2018: A[y] for y in YEARS})
    print(f"\n{len(scr)} screening tiles, {len(held)} held out; family {len(fam)} features {fam}")
    if not fam:
        return
    for storm, ey in EVENT_YEAR.items():
        pk = [max(t, key=t.get) for t in (contrast_traj(r, fam) for r in held if r["storm"] == storm)]
        if pk:
            print(f"  {storm}: validation rate {{{ey},{ey + 1}}} {np.mean([p - ey in (0, 1) for p in pk]):.3f} "
                  f"(n={len(pk)})")

    json.dump(dict(family=fam, n_screen=len(scr)), open(RESULTS_DIR / "windthrow.json", "w"), default=float)
    print("\nsaved results/windthrow.json")

def phase_select():
    import warnings
    warnings.filterwarnings("ignore")
    import geopandas as gpd
    MIN_AREA_HA = 25.0
    CAP = 60
    fw = gpd.read_file(GT_DIR / "forwind" / "FORWIND_v2.shp")
    fw = fw.to_crs(4326)
    from lib.tiles import tile_for_point
    out = []
    for storm in ("Friederike", "Xavier", "Vaia"):
        sub = fw[fw.StormName.astype(str) == storm].copy()
        dates = sorted(sub.EventDate.astype(str).str[:10].value_counts().items(),
                       key=lambda kv: -kv[1])
        print(f"{storm}: {len(sub)} polygons; EventDate top: {dates[:3]}")
        cen = sub.geometry.centroid
        sub["tile"] = [tile_for_point(x, y) for x, y in zip(cen.x, cen.y)]
        # FORWIND v2's Area column is in hectares
        ha = sub.Area.astype(float)
        agg = sub.assign(ha=ha).groupby("tile").agg(
            ha=("ha", "sum"), n=("ha", "size")).reset_index()
        agg = agg[agg.ha >= MIN_AREA_HA].sort_values("ha", ascending=False)[:CAP]
        for _, r in agg.iterrows():
            lon, lat = (float(v) for v in r.tile.split("_")[1:])
            out.append(dict(tile=r.tile, storm=storm.lower(), ha=round(float(r.ha), 1),
                            n_poly=int(r.n), lon=lon, lat=lat))
        print(f"  kept {len(agg)} tiles, damage {agg.ha.min():.0f}-{agg.ha.max():.0f} ha, "
              f"lon {agg.tile.str.split('_').str[1].astype(float).min():.1f}.."
              f"{agg.tile.str.split('_').str[1].astype(float).max():.1f}")

    json.dump(out, open(RESULTS_DIR / "windthrow_selection.json", "w"))
    tiles = sorted({o["tile"] for o in out})
    print(f"\n{len(out)} rows, {len(tiles)} distinct tiles")

if __name__ == "__main__":
    {"select": phase_select, "train": phase_train, "screen": phase_screen,
     "fetch": phase_fetch, "verdicts": verdicts}[sys.argv[1]]()
