"""Find a forest-clearing family in AlphaEarth across the Amazon arc and test it on held-out tile-years.

Reference product: Hansen Global Forest Change (lossyear, treecover2000), detection-dated. Forest is treecover2000 of
at least 60. `select` picks 160 tiles from Hansen alone (criteria in phase_select). A unit is a tile-year of clearing
in forest, Y = 2018-2022, with at least 600 pixels at stride 2. Undisturbed pixels are the tile's forest with no
recorded loss (at least 2,000 at stride 2). One dictionary (512 features, k = 8) is trained on up to 9,000 pixels per
tile-year from 60 random tiles. The family is screened with the anchored bar on the 2018 units and scored on the
2019-2022 units. A unit validates when the family's mean cleared-minus-undisturbed contrast over 2017-2025 peaks in
the expected window, Y or Y+1.

Usage: python experiments/amazon_arc.py select|train|screen|verdicts
Writes results/amazon_arc_selection.json, amazon_arc.jsonl/.json and the dictionary in data/checkpoints/amazon_arc.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from lib.config import ARC_GRANULES, CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.alphaearth import compute_scale, load_aef
from lib.tiles import tile_grid
from lib.screen import anchored_family
from lib.sae import group_sums, load_sae, train_and_save

ARC = GT_DIR / "hansen_arc"
YEARS = list(range(2017, 2026))
EVENT_YEARS = range(2018, 2023)
S = 2
M = 512
CKPT = CKPT_DIR / "amazon_arc" / "sae_arc_aef.pt"
JSONL = RESULTS_DIR / "amazon_arc.jsonl"
_SEL_PATH = RESULTS_DIR / "amazon_arc_selection.json"
# Empty until `select` has run, so that phase still works on a fresh checkout.
SEL = json.load(open(_SEL_PATH)) if _SEL_PATH.exists() else []
TILE_META = {m["tile"]: m for m in SEL}
TILES = sorted(TILE_META)

def _available(tiles):
    """Return the tiles that have a readable 2018 AlphaEarth raster."""
    ok = []
    for t in tiles:
        try:
            load_aef(t, 2018, stride=8)
            ok.append(t)
        except Exception:
            pass
    return ok

def phase_train():
    if CKPT.exists():
        print("cached"); return
    rng = np.random.default_rng(0)
    scale = compute_scale()
    print(f"SCALE x{scale:.1f}", flush=True)
    avail = _available(TILES)
    print(f"{len(avail)}/{len(TILES)} tiles available in the AlphaEarth archive", flush=True)
    pick = rng.choice(avail, size=min(60, len(avail)), replace=False)
    Xs = []
    for t in pick:
        for y in YEARS[::2]:                      # 2017, 19, 21, 23, 25
            try:
                x = np.nan_to_num(load_aef(t, y, stride=8)).reshape(-1, 64)
            except Exception:
                continue
            idx = rng.choice(len(x), size=min(9000, len(x)), replace=False)
            Xs.append(x[idx])
    X = (np.concatenate(Xs) * scale).astype(np.float32)
    rng.shuffle(X)          # the validation split is the last rows, so they must not all come from one tile
    print(f"Amazon arc AlphaEarth corpus {X.shape} (scaled)", flush=True)
    train_and_save(X, CKPT, epochs=40, seed=0, log=True, d=64, m=M, k=8)
    print("trained", flush=True)

def tile_groups(tile):
    """Return masks on the tile's 10 m grid, keyed by group: forest cleared in each event year, and intact forest."""
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    meta = TILE_META[tile]
    lon, lat = (float(v) for v in tile.split("_")[1:])
    bb = (lon - 0.055, lat - 0.055, lon + 0.055, lat + 0.055)
    dst_t, dst_crs, H, W = tile_grid(tile)
    with rasterio.open(ARC / f"lossyear_{meta['granule']}.tif") as ls, \
            rasterio.open(ARC / f"treecover2000_{meta['granule']}.tif") as ts:
        wb = transform_bounds("EPSG:4326", ls.crs, *bb)
        win = ls.window(*wb)
        L = ls.read(1, window=win)
        T = ts.read(1, window=win)
        wt, crs = ls.window_transform(win), ls.crs

    def to_grid(src_arr):
        d = np.zeros((H, W))
        reproject(src_arr.astype(float), d, src_transform=wt, src_crs=crs,
                  dst_transform=dst_t, dst_crs=dst_crs,
                  resampling=Resampling.nearest)
        return d

    forest = to_grid(T >= 60) > 0.5
    Lg = to_grid(L)
    groups = {"intact": forest & (Lg < 0.5)}
    for y in EVENT_YEARS:
        m = forest & (np.abs(Lg - (y - 2000)) < 0.5)
        if m[::S, ::S].sum() >= 600:
            groups[f"clear_{y}"] = m
    return groups

def group_tables_aef(sae, tile, groups, scale):
    """Mean activation of every feature over each group, by year, from stride-S AlphaEarth embeddings."""
    gs = {k: v[::S, ::S] for k, v in groups.items()}
    out = {}
    for y in YEARS:
        try:
            x = np.nan_to_num(load_aef(tile, y, stride=S)) * scale
        except Exception:
            continue
        s, n = group_sums(sae, x, gs)
        out[y] = {k: s[k] / n[k] for k in gs if n[k] > 0}
        del x
    return out

def run_tile(args):
    tile, scale = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        groups = tile_groups(tile)
        if "intact" not in groups or groups["intact"][::S, ::S].sum() < 2000 \
                or len(groups) < 2:
            return dict(ok=False, tile=tile, error="no disturbed groups")
        sae = load_sae(path=CKPT, d=64, m=M, k=8)
        T = group_tables_aef(sae, tile, groups, scale)
        meta = TILE_META[tile]
        return dict(ok=True, tile=tile,
                    lon=float(tile.split("_")[1]), lat=float(tile.split("_")[2]),
                    frac=meta["frac"], granule=meta["granule"],
                    tables={str(y): {g: T[y][g].tolist() for g in T[y]} for y in T})
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

def phase_screen():
    from concurrent.futures import ProcessPoolExecutor
    scale = compute_scale()
    print(f"SCALE x{scale:.1f}", flush=True)
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
        for i, r in enumerate(ex.map(run_tile, [(t, scale) for t in todo]), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    verdicts()

def blind_family(rows):
    """Screen the family with the anchored bar on the median contrast over the 2018 units. Returns (family, n units)."""
    g = "clear_2018"
    rs = [r for r in rows if g in r["tables"].get("2018", {})]
    if len(rs) < 5:
        return [], 0
    A = np.array([np.median([[r["tables"][str(y)][g][f]
                              - r["tables"][str(y)]["intact"][f] for f in range(M)]
                             for r in rs if str(y) in r["tables"]
                             and g in r["tables"][str(y)]], axis=0) for y in YEARS])
    fam = anchored_family({y - 2018: A[i] for i, y in enumerate(YEARS)})
    return fam, len(rs)

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    print(f"\n{len(rows)} tiles screened cleanly")
    fam, nscr = blind_family(rows)
    print(f"clearing family: {len(fam)} features (screened on {nscr} tiles' 2018 units)")
    if not fam:
        json.dump(dict(family=[], n=0), open(RESULTS_DIR / "amazon_arc.json", "w"))
        return
    V = []
    for r in rows:
        for y in EVENT_YEARS:
            if y == 2018:
                continue
            g = f"clear_{y}"
            t = {int(yy): float(np.mean([r["tables"][yy][g][f] for f in fam])
                                - np.mean([r["tables"][yy]["intact"][f] for f in fam]))
                 for yy in r["tables"] if g in r["tables"][yy]}
            if not t:
                continue
            pk = max(t, key=t.get)
            V.append(dict(tile=r["tile"], lon=r["lon"], lat=r["lat"], frac=r["frac"],
                          year=y, peak=pk, off=pk - y, ok=bool(pk - y in (0, 1))))
    pooled = float(np.mean([v["ok"] for v in V]))
    print(f"held-out validation rate {{Y,Y+1}}: {pooled:.3f} (n={len(V)})")
    rel = [v for v in V if v["ok"]]
    y0 = float(np.mean([v["off"] == 0 for v in rel]))
    print(f"share of dated units that peak in Y: {y0:.3f} (n={len(rel)})")

    off = defaultdict(int)
    for v in V:
        off[int(np.clip(v["off"], -1, 3))] += 1
    print("   peak-offset histogram:",
          {k: round(off[k] / len(V), 3) for k in sorted(off)})

    json.dump(dict(family=fam, n=len(V), pooled=pooled, y0_share=y0, outcomes=V),
              open(RESULTS_DIR / "amazon_arc.json", "w"), default=float)
    print("\nsaved results/amazon_arc.json")


# ------------------------------------------------ selection --

def phase_select():
    """Pick tiles from Hansen alone and write results/amazon_arc_selection.json.

    Tiles are checked every 0.2 degrees over the arc granules. A tile needs at least 20,000 forest and 20,000 intact
    forest Hansen pixels, and at least 3% of its forest lost in 2018-2022. The 160 tiles with the most cleared pixels
    are kept.
    """
    import rasterio
    ARC = GT_DIR / "hansen_arc"

    YEARS = range(18, 23)
    MIN_FRAC = 0.03
    N_KEEP = 160
    STEP = 0.2
    from lib.tiles import snap, tile_for_point
    scores = []
    for gi, g in enumerate(ARC_GRANULES, 1):
        lp, tp = ARC / f"lossyear_{g}.tif", ARC / f"treecover2000_{g}.tif"
        if not lp.exists():
            continue
        with rasterio.open(lp) as ls, rasterio.open(tp) as ts:
            w, s, e, n = ls.bounds.left, ls.bounds.bottom, ls.bounds.right, ls.bounds.top
            for lo in np.arange(np.floor(w / STEP) * STEP, e, STEP):
                for la in np.arange(np.floor(s / STEP) * STEP, n, STEP):
                    bb = (snap(lo) - 0.05, snap(la) - 0.05,
                          snap(lo) + 0.05, snap(la) + 0.05)
                    try:
                        win = ls.window(*bb)
                        if win.width < 4 or win.height < 4:
                            continue
                        L = ls.read(1, window=win)
                        T = ts.read(1, window=win)
                    except Exception:
                        continue
                    forest = T >= 60
                    nf = int(forest.sum())
                    if nf < 20000:
                        continue
                    lost = np.isin(L, list(YEARS)) & forest
                    # Also require intact forest. Tiles that lost nearly all their forest have no undisturbed pixels.
                    intact = forest & (L == 0)
                    if int(intact.sum()) < 20000:
                        continue
                    frac = float(lost.sum() / nf)
                    if frac >= MIN_FRAC:
                        # rank by the number of cleared pixels (the group size), not by the fraction
                        scores.append((int(lost.sum()), tile_for_point(lo, la),
                                       nf, g, frac, int(intact.sum())))
        print(f"  [{gi}/{len(ARC_GRANULES)}] {g}: {len(scores)} candidates so far", flush=True)

    scores.sort(reverse=True)
    keep = scores[:N_KEEP]
    meta = [dict(tile=t, lost_px=n, forest_px=nf, granule=g, frac=round(fr, 4),
                 intact_px=ip) for n, t, nf, g, fr, ip in keep]

    lons = [float(m["tile"].split("_")[1]) for m in meta]
    lats = [float(m["tile"].split("_")[2]) for m in meta]
    print(f"\nselected {len(meta)} tiles; loss fraction "
          f"{min(m['frac'] for m in meta):.3f}-{max(m['frac'] for m in meta):.3f}; "
          f"clearing px {min(m['lost_px'] for m in meta):,}-{max(m['lost_px'] for m in meta):,}; "
          f"intact px >= {min(m['intact_px'] for m in meta):,}")

    print(f"  lon {min(lons):.1f}..{max(lons):.1f}  lat {min(lats):.1f}..{max(lats):.1f}")
    json.dump(meta, open(RESULTS_DIR / "amazon_arc_selection.json", "w"))
    print("saved results/amazon_arc_selection.json")

if __name__ == "__main__":
    {"select": phase_select, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
