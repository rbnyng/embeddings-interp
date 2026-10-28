"""Screen boreal Canada fires in AlphaEarth with a local dictionary.

Reference product: NBAC (Canadian National Burned Area Composite), event-dated. A unit is one NBAC fire from 2018-2023
of at least 10,000 ha, on the 0.1-degree tile it covers most, with at least 15% of that tile burned. Disturbed pixels
are inside the fire polygon (at least 300 at stride 2). Undisturbed pixels are the rest of the tile. Contrasts cover
2017-2024. The dictionary (512 features, k = 8) is trained on 1,500 pixels per tile-year, 2017-2023, from 60 random
tiles. The family is screened with the anchored bar on the 2018 fires and scored on the 2019-2023 fires. The expected
window is Y+1, the year after the fire.

Usage: python experiments/boreal_fire.py inventory|fetch|screen|verdicts
Writes results/nbac_inventory.json, nbac_geoms.geojson, boreal_fetch_errors.json, boreal.jsonl and boreal.json, and the
dictionary to data/checkpoints/aef_boreal.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.alphaearth import compute_scale, fetch_aef, load_aef
from lib.tiles import download_landmask, tile_grid
from lib.sae import group_sums, load_sae, train_and_save
from lib.screen import anchored_family
YEARS = list(range(2017, 2025))          # through Y+1 for the 2023 fires
S = 2
MIN_HA = 10_000
FIRE_YEARS = range(2018, 2024)
INV_PATH = RESULTS_DIR / "nbac_inventory.json"
GEO_PATH = RESULTS_DIR / "nbac_geoms.geojson"
CKPT = CKPT_DIR / "aef_boreal" / "sae_aef_boreal.pt"
JSONL = RESULTS_DIR / "boreal.jsonl"

def phase_inventory():
    import geopandas as gpd
    from shapely.geometry import box as sbox
    shp = next((GT_DIR / "nbac").glob("*.shp"))
    print("reading", shp.name, flush=True)
    g = gpd.read_file(shp)
    print("columns:", list(g.columns), flush=True)
    ycol = next(c for c in g.columns if c.upper() in ("YEAR", "FIRE_YEAR", "NBAC_YR"))
    acol = next((c for c in g.columns if "HA" in c.upper()
                 and g[c].dtype.kind in "if"), None)
    print(f"year col {ycol}, area col {acol}", flush=True)
    g[ycol] = gpd.pd.to_numeric(g[ycol], errors="coerce")
    sel = g[g[ycol].isin(list(FIRE_YEARS))].to_crs("EPSG:4326")
    if acol:
        sel = sel[gpd.pd.to_numeric(sel[acol], errors="coerce") >= MIN_HA]
    print(f"{len(sel)} fires {FIRE_YEARS.start}-{FIRE_YEARS.stop-1} >= {MIN_HA} ha",
          flush=True)

    from lib.tiles import snap
    recs, geoms = [], []
    date_cols = [c for c in sel.columns if "DATE" in c.upper()]
    for i, r in sel.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        lonmin, latmin, lonmax, latmax = geom.bounds
        best, bf = None, 0.0
        for lo in np.arange(snap(lonmin), snap(lonmax) + 0.1, 0.1):
            for la in np.arange(snap(latmin), snap(latmax) + 0.1, 0.1):
                cell = sbox(lo - 0.05, la - 0.05, lo + 0.05, la + 0.05)
                try:
                    fr = geom.intersection(cell).area / cell.area
                except Exception:
                    continue
                if fr > bf:
                    best, bf = (round(lo, 2), round(la, 2)), fr
        if best is None or bf < 0.15:
            continue
        uid = int(i)
        dates = {c: str(r[c])[:10] for c in date_cols if r[c] is not None}
        recs.append(dict(uid=uid, year=int(r[ycol]),
                         area_ha=float(r[acol]) if acol else None,
                         tile=f"grid_{best[0]}_{best[1]}", burn_frac=round(bf, 3),
                         dates=dates))
        geoms.append(dict(type="Feature", properties=dict(uid=uid),
                          geometry=geom.__geo_interface__))
    by_year = {}
    for r in recs:
        by_year[r["year"]] = by_year.get(r["year"], 0) + 1
    print(f"tile-dominant: {len(recs)} fires on {len({r['tile'] for r in recs})} "
          f"tiles; by year {dict(sorted(by_year.items()))}", flush=True)
    json.dump(recs, open(INV_PATH, "w"))
    json.dump(dict(type="FeatureCollection", features=geoms), open(GEO_PATH, "w"))
    print("saved inventory")

def phase_fetch():
    recs = json.load(open(INV_PATH))
    tiles = sorted({r["tile"] for r in recs})
    jobs = [(t, y) for t in tiles for y in YEARS]
    print(f"{len(tiles)} tiles x {len(YEARS)} yr = {len(jobs)} "
          f"(~{len(jobs)*0.077:.0f} GB)", flush=True)

    def one(j):
        t, y = j
        try:
            if y == YEARS[0]:
                download_landmask(t)
            fetch_aef(t, y)
            return None
        except Exception as e:
            return f"{t} {y}: {e!r}"
    errs = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            if r:
                errs.append(r)
            if i % 150 == 0:
                print(f"  [{i}/{len(jobs)}] ({len(errs)} errors)", flush=True)
    print(f"AlphaEarth fetch done ({len(errs)} errors)", flush=True)
    json.dump(errs, open(RESULTS_DIR / "boreal_fetch_errors.json", "w"))

def fire_mask(rec, geoms_cache={}):
    from shapely.geometry import shape
    from rasterio import features as rfeat
    import geopandas as gpd
    if not geoms_cache:
        geoms_cache.update({f["properties"]["uid"]: f["geometry"]
                            for f in json.load(open(GEO_PATH))["features"]})
    g = shape(geoms_cache[rec["uid"]])
    if not g.is_valid:
        g = g.buffer(0)
    t, crs, H, W = tile_grid(rec["tile"])
    gs = gpd.GeoSeries([g], crs="EPSG:4326").to_crs(crs)
    m = rfeat.rasterize([(gs.iloc[0], 1)], out_shape=(H, W), transform=t,
                        fill=0, dtype=np.uint8).astype(bool)
    return m[::S, ::S]

def get_local_sae(scale):
    if CKPT.exists():
        sae = load_sae(path=CKPT, d=64, m=512, k=8)
        return sae
    rng = np.random.default_rng(0)
    recs = json.load(open(INV_PATH))
    tiles = sorted({r["tile"] for r in recs})
    pick = rng.choice(tiles, size=min(60, len(tiles)), replace=False)
    out = []
    for t in pick:
        for y in YEARS[:7]:
            try:
                x = np.nan_to_num(load_aef(t, y, stride=1)) * scale
            except Exception:
                continue
            H, W, _ = x.shape
            idx = rng.choice(H * W, 1500, replace=False)
            out.append(x.reshape(-1, 64)[idx])
            del x
    X = np.concatenate(out).astype(np.float32)
    print(f"boreal corpus {X.shape}", flush=True)
    sae = train_and_save(X, CKPT, epochs=40, seed=0, log=True, d=64, m=512, k=8)
    sae.eval()
    return sae

def screen_fire(args):
    rec, scale = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae = load_sae(path=CKPT, d=64, m=512, k=8)
        b = fire_mask(rec)
        traj = {}
        for y in YEARS:
            try:
                x = np.nan_to_num(load_aef(rec["tile"], y, stride=S)) * scale
            except Exception:
                continue
            H, W, _ = x.shape
            bb = b[:H, :W]; uu = ~bb
            if bb.sum() < 300:
                return dict(ok=False, uid=rec["uid"], error="few px")
            s, n = group_sums(sae, x, {"burn": bb, "unburn": uu})
            traj[y] = ((s["burn"] / n["burn"]) - (s["unburn"] / max(n["unburn"], 1))).tolist()
        fy = rec["year"]
        if not {fy, fy + 1} <= set(traj):
            return dict(ok=False, uid=rec["uid"], error="gap")
        return dict(ok=True, uid=rec["uid"], which="local", year=fy,
                    traj={str(y): v for y, v in traj.items()})
    except Exception as e:
        return dict(ok=False, uid=rec.get("uid"), error=repr(e))

def phase_screen():
    scale = compute_scale()
    get_local_sae(scale)
    recs = json.load(open(INV_PATH))
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                r = json.loads(line)
                done.add((r["uid"], r["which"]))
            except Exception:
                pass
    jobs = [(r, scale) for r in recs if (r["uid"], "local") not in done]
    print(f"{len(jobs)} screen jobs", flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(70)) as ex:
        for i, r in enumerate(ex.map(screen_fire, jobs), 1):
            out.write(json.dumps(r) + "\n"); out.flush()
            if i % 40 == 0:
                print(f"  [{i}/{len(jobs)}]", flush=True)
    out.close()
    verdicts()

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    loc = [r for r in rows if r["which"] == "local"]

    # The screening units are the 2018 fires.
    scr = [r for r in loc if r["year"] == 2018]
    held = [r for r in loc if r["year"] != 2018]
    ys = sorted({int(y) for r in loc for y in r["traj"]})
    A = np.stack([np.median([r["traj"][str(y)] for r in scr], axis=0) for y in ys])
    fam = anchored_family({y - 2018: A[i] for i, y in enumerate(ys)}, window=(1,))
    print(f"{len(scr)} screening fires (2018), {len(held)} held out; family {len(fam)} features {fam}")

    def peak(r):
        t = {int(y): float(np.mean([r["traj"][str(y)][f] for f in fam]))
             for y in r["traj"]}
        return max(t, key=t.get)

    if fam:
        rate = float(np.mean([peak(r) - r["year"] == 1 for r in held]))
        print(f"held-out validation rate, peak at Y+1: {rate:.3f} (n={len(held)})")
    json.dump(dict(family=fam, n_screen=len(scr), n_held=len(held)),
              open(RESULTS_DIR / "boreal.json", "w"))
    print("\nsaved results/boreal.json")

if __name__ == "__main__":
    {"inventory": phase_inventory, "fetch": phase_fetch, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
