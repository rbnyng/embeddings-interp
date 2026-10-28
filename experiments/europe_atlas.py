"""Europe atlas: find one family per EFDA disturbance agent (harvest, wind/bark beetle, fire) and test it on held-out
tile-years. Detection-dated.

Tiles are picked from EFDA alone, on a 0.2 degree scan of each country. A tile qualifies for an agent when it has at
least 4,000 EFDA forest pixels and the agent disturbed at least 2% of them in 2018-2022. The 6 tiles with the highest
share are kept per country and agent. Units are tile-years of one agent, Y = 2018-2022, and the expected window is Y
or Y+1. Pixel groups lie on the EFDA forest mask, at stride 2:
  <agent>_Y  forest the agent disturbed in year Y, at least 600 pixels
  intact     forest with no EFDA disturbance in 1985-2023, at least 2,000 pixels
One pan-European dictionary is trained on up to 120 random tiles, 4,500 pixels from each year 2017-2025. Each agent's
family is screened on its 2018 units (at least 5 tiles) and scored on its 2019-2022 units.

Usage: python experiments/europe_atlas.py select|train|screen|verdicts
Writes results/europe_selection.json, europe_atlas.jsonl and europe_atlas.json, and the dictionary checkpoint.
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.tiles import tile_grid, sample_pixels
from lib.trajectories import group_tables
from lib.screen import anchored_family
from lib.sae import load_sae, train_and_save

EFDA = GT_DIR / "efda"
YEARS = list(range(2017, 2026))
EVENT_YEARS = range(2018, 2023)
S = 2
BAND = lambda y: y - 1984
AGENT_CODE = {"wind_beetle": 1, "fire": 2, "harvest": 3}
CKPT = CKPT_DIR / "europe_atlas" / "sae_europe.pt"
JSONL = RESULTS_DIR / "europe_atlas.jsonl"
_SEL_PATH = RESULTS_DIR / "europe_selection.json"
# Empty until `select` has run, so that phase still works on a fresh checkout.
SEL = json.load(open(_SEL_PATH)) if _SEL_PATH.exists() else []
TILE_META = defaultdict(list)
for m in SEL:
    TILE_META[m["tile"]].append(m)
TILES = sorted(TILE_META)

def phase_train():
    if CKPT.exists():
        print("cached"); return
    rng = np.random.default_rng(0)
    pick = rng.choice(TILES, size=min(120, len(TILES)), replace=False)
    X = np.concatenate([sample_pixels([(t, y) for y in YEARS], n_per=4500, seed=0)
                        for t in pick])
    print(f"pan-European corpus {X.shape}", flush=True)
    train_and_save(X, CKPT, epochs=40, seed=0, log=True)
    print("trained", flush=True)

def tile_groups(tile):
    """Return full-resolution masks for intact forest and each agent-year group of at least 600 pixels at stride 2."""
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    countries = sorted({m["country"] for m in TILE_META[tile]})
    lon, lat = (float(v) for v in tile.split("_")[1:])
    bb = (lon - 0.055, lat - 0.055, lon + 0.055, lat + 0.055)
    dst_t, dst_crs, H, W = tile_grid(tile)
    agent = np.zeros((len(list(EVENT_YEARS)), H, W), np.uint8)
    anyd = np.zeros((H, W), bool)
    forest = np.zeros((H, W), bool)
    for c in countries:
        try:
            with rasterio.open(EFDA / f"disturbance_agent_1985_2023_{c}.tif") as src:
                wb = transform_bounds("EPSG:4326", src.crs, *bb)
                win = src.window(*wb)
                if win.width < 2 or win.height < 2:
                    continue
                ag = src.read([BAND(y) for y in EVENT_YEARS], window=win)
                wt, crs = src.window_transform(win), src.crs
            with rasterio.open(EFDA / f"annual_disturbances_1985_2023_{c}.tif") as src:
                ad = (src.read(window=win) == 1).any(0)
            with rasterio.open(EFDA / f"forest_mask_{c}.tif") as src:
                fo = src.read(1, window=win) == 1
        except Exception:
            continue
        for i in range(ag.shape[0]):
            d = np.zeros((H, W))
            reproject(ag[i].astype(float), d, src_transform=wt, src_crs=crs,
                      dst_transform=dst_t, dst_crs=dst_crs,
                      resampling=Resampling.nearest)
            agent[i] = np.maximum(agent[i], (d * (d < 250)).astype(np.uint8))
        for sarr, darr in ((ad, anyd), (fo, forest)):
            d = np.zeros((H, W))
            reproject(sarr.astype(float), d, src_transform=wt, src_crs=crs,
                      dst_transform=dst_t, dst_crs=dst_crs,
                      resampling=Resampling.nearest)
            darr |= d > 0.5
    groups = {"intact": forest & ~anyd}
    for i, y in enumerate(EVENT_YEARS):
        for name, code in AGENT_CODE.items():
            m = (agent[i] == code) & forest
            if m[::S, ::S].sum() >= 600:
                groups[f"{name}_{y}"] = m
    return groups

def run_tile(tile):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        groups = tile_groups(tile)
        if "intact" not in groups or groups["intact"][::S, ::S].sum() < 2000 \
                or len(groups) < 2:
            return dict(ok=False, tile=tile, error="no disturbed groups")
        sae = load_sae(path=CKPT)
        T = group_tables(sae, tile, YEARS, groups, stride=S)
        return dict(ok=True, tile=tile,
                    countries=sorted({m["country"] for m in TILE_META[tile]}),
                    lat=float(tile.split("_")[2]),
                    tables={str(y): {g: T[y][g].tolist() for g in T[y]}
                            for y in YEARS})
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

def phase_screen():
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
    with ProcessPoolExecutor(max_workers=workers(64)) as ex:
        for i, r in enumerate(ex.map(run_tile, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    verdicts()

def blind_family(rows, agent):
    """Return the family screened on the median contrast over the agent's 2018 units, and the number of those units."""
    g = f"{agent}_2018"
    rs = [r for r in rows if g in r["tables"]["2018"]]
    if len(rs) < 5:
        return [], 0
    M = 1024
    A = np.array([np.median([[r["tables"][str(y)][g][f]
                              - r["tables"][str(y)]["intact"][f] for f in range(M)]
                             for r in rs], axis=0) for y in YEARS])
    fam = anchored_family({y - 2018: A[i] for i, y in enumerate(YEARS)})
    return fam, len(rs)

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    print(f"\n{len(rows)} tiles screened cleanly")
    out = {}
    allv = []
    for agent in AGENT_CODE:
        fam, nscr = blind_family(rows, agent)
        print(f"{agent}: family {len(fam)} features "
              f"(screened on {nscr} tiles' 2018 units)")
        if not fam:
            out[agent] = dict(family=[], n=0)
            continue
        v = []
        for r in rows:
            for y in EVENT_YEARS:
                if y == 2018:
                    continue
                g = f"{agent}_{y}"
                t = {int(yy): float(np.mean([r["tables"][str(yy)][g][f] for f in fam])
                                    - np.mean([r["tables"][str(yy)]["intact"][f]
                                               for f in fam]))
                     for yy in r["tables"] if g in r["tables"][str(yy)]}
                if not t:
                    continue
                pk = max(t, key=t.get)
                v.append(dict(agent=agent, tile=r["tile"], lat=r["lat"],
                              countries=r["countries"], year=y, peak=pk,
                              off=pk - y, ok=bool(pk - y in (0, 1))))
        allv += v
        rate = float(np.mean([x["ok"] for x in v])) if v else float("nan")
        out[agent] = dict(family=fam, n=len(v), rate=rate)
        print(f"   held-out validation rate {{Y,Y+1}}: {rate:.3f}  (n={len(v)})")

    pooled = float(np.mean([x["ok"] for x in allv])) if allv else float("nan")
    print(f"\npooled held-out validation rate {pooled:.3f} (n={len(allv)})")

    json.dump(dict(agents=out, pooled=pooled, outcomes=allv),
              open(RESULTS_DIR / "europe_atlas.json", "w"), default=float)
    print("\nsaved results/europe_atlas.json")


# ------------------------------------------------ selection --

def phase_select():
    """Scan each EFDA country raster every 0.2 degrees and keep the tiles with the highest share of forest each agent
    disturbed in 2018-2022."""
    import rasterio
    from rasterio.warp import transform_bounds
    EFDA = GT_DIR / "efda"
    YEARS = range(2018, 2023)
    BAND = lambda y: y - 1984
    AGENTS = {1: "wind_beetle", 2: "fire", 3: "harvest"}
    MIN_FRAC = 0.02
    PER_COUNTRY_AGENT = 6
    STEP = 0.2
    from lib.tiles import snap, tile_for_point
    countries = sorted(p.stem.replace("disturbance_agent_1985_2023_", "")
                       for p in EFDA.glob("disturbance_agent_1985_2023_*.tif"))

    print(f"{len(countries)} EFDA countries", flush=True)
    cands = defaultdict(list)
    for ci, c in enumerate(countries, 1):
        try:
            ag_path = EFDA / f"disturbance_agent_1985_2023_{c}.tif"
            fm_path = EFDA / f"forest_mask_{c}.tif"
            with rasterio.open(ag_path) as src, rasterio.open(fm_path) as fsrc:
                w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                for lo in np.arange(np.floor(w / STEP) * STEP, e, STEP):
                    for la in np.arange(np.floor(s / STEP) * STEP, n, STEP):
                        t = tile_for_point(lo, la)
                        bb = (snap(lo) - 0.05, snap(la) - 0.05,
                              snap(lo) + 0.05, snap(la) + 0.05)
                        try:
                            wb = transform_bounds("EPSG:4326", src.crs, *bb)
                            win = src.window(*wb)
                            if win.width < 4 or win.height < 4:
                                continue
                            a = src.read([BAND(y) for y in YEARS], window=win)
                            f = fsrc.read(1, window=win)
                        except Exception:
                            continue
                        forest = f == 1
                        nf = int(forest.sum())
                        if nf < 4000:
                            continue
                        for code, name in AGENTS.items():
                            frac = float((np.any(a == code, 0) & forest).sum() / nf)
                            if frac >= MIN_FRAC:
                                cands[(c, name)].append((frac, t, nf))
        except Exception as ex:
            print(f"  {c}: {ex!r}", flush=True)
        if ci % 5 == 0:
            print(f"  [{ci}/{len(countries)}] {sum(len(v) for v in cands.values())} "
                  f"candidates so far", flush=True)

    meta = []
    for (c, agent), lst in cands.items():
        lst.sort(reverse=True)
        for frac, t, nf in lst[:PER_COUNTRY_AGENT]:
            meta.append(dict(tile=t, country=c, agent=agent, frac=round(frac, 4),
                             forest_px=nf))

    tiles = sorted({m["tile"] for m in meta})
    by_agent = defaultdict(set)
    for m in meta:
        by_agent[m["agent"]].add(m["tile"])

    print(f"\nselected {len(tiles)} distinct tiles across "
          f"{len({m['country'] for m in meta})} countries")

    for a, s_ in by_agent.items():
        print(f"  {a}: {len(s_)} tiles")

    json.dump(meta, open(RESULTS_DIR / "europe_selection.json", "w"))
    print("saved results/europe_selection.json")

if __name__ == "__main__":
    {"select": phase_select, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
