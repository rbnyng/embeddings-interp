"""Rainfall landslides: find a family on three inventories mapped by hand after one rainstorm each, and test it on
held-out tiles. Event-dated.

  emilia   RER2023, Emilia-Romagna, May 2023
  beijing  RLBJ, western Beijing, late July 2023 (Typhoon Doksuri)
  zixing   RLZX-LIM, Zixing, Hunan, 26-28 July 2024
Each unit is a 0.1 degree tile with at least 15 ha of one event's landslides. A polygon belongs to the tile that holds
one of its interior points. Up to 60 tiles per event are kept, largest first, and each needs Tessera for Y-1, Y and Y+1.
The expected window is the event year Y or Y+1. Pixel groups use every second pixel, up to 5,000 per tile-year:
  disturbed    every mapped landslide polygon on the tile, at least 300 pixels
  undisturbed  WorldCover land that is not built-up, water or snow, more than 50 m from any landslide and at least as
               steep as the 25th percentile of landslide slope (Copernicus GLO-30), at least 2,000 pixels
One dictionary is trained on all three events' tiles. The family is screened on 15 random Emilia-Romagna tiles (seed 0)
and tested on all other tiles.

Usage: python experiments/landslides.py select|fetch|train|screen|verdicts
Writes results/landslide_selection.json, landslides.jsonl and landslides.json, and the dictionary checkpoint.
"""
import json
import sys
import zlib
from functools import lru_cache

import numpy as np

from lib.config import CKPT_DIR, DATA_DIR, GT_DIR, RESULTS_DIR
from lib.tiles import available_years, download_landmask, tile_bbox, tile_for_point, tile_grid
from lib.sae import load_sae, train_and_save
from lib.sampling import corpus, mean_activation, sample_tile_years
from lib.screen import aggregate_contrasts, anchored_family, peak_outcome, row_contrasts, row_units, outcome_summary

YEARS = list(range(2017, 2026))
S, WINDOW = 2, (0, 1)
CAP, N_TRAIN = 5000, 5000
MIN_HA, CAP_TILES, SCREEN_N = 15.0, 60, 15
LS = GT_DIR / "landslides"
EVENTS = {"emilia": dict(path=LS / "rer2023" / "RER2023_V1.shp", year=2023),
          "beijing": dict(path=LS / "beijing2023" / "Polygon_RLBJ.shp", year=2023),
          "zixing": dict(path=LS / "zixing2024" / "RLZX-LIM.shp", year=2024)}
PIX = DATA_DIR / "intermediate" / "landslides"
CKPT = CKPT_DIR / "landslides" / "sae_landslides.pt"
SEL_PATH = RESULTS_DIR / "landslide_selection.json"
JSONL = RESULTS_DIR / "landslides.jsonl"
SEL = json.load(open(SEL_PATH)) if SEL_PATH.exists() else []


def _npz(rec):
    return PIX / f"{rec['uid'].replace(':', '_')}.npz"


@lru_cache(None)
def _polygons(event):
    import geopandas as gpd
    g = gpd.read_file(EVENTS[event]["path"], on_invalid="fix")    # one Beijing polygon ring is not closed
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g.set_geometry(g.geometry.make_valid()).to_crs(4326)[["geometry"]]


def phase_select():
    out = []
    for event, e in EVENTS.items():
        g = _polygons(event)
        area = g.to_crs(g.estimate_utm_crs()).area / 1e4                    # ha
        cen = g.geometry.representative_point()
        tiles = np.array([tile_for_point(x, y) for x, y in zip(cen.x, cen.y)])
        ha = {t: float(area[tiles == t].sum()) for t in np.unique(tiles)}
        big = sorted([t for t, v in ha.items() if v >= MIN_HA], key=lambda t: -ha[t])[:CAP_TILES]
        have = available_years(big, YEARS)
        Y = e["year"]
        keep = [t for t in big if {Y - 1, Y, Y + 1} <= have[t]]
        out += [dict(uid=f"{event}:{t}", event=event, tile=t, year=Y, ha=round(ha[t], 1), years=sorted(have[t])) for t in keep]
        print(f"{event}: {len(g)} landslides, {sum(v >= MIN_HA for v in ha.values())} tiles >= {MIN_HA} ha, "
              f"{len(big)} kept by the cap, {len(keep)} with the years needed", flush=True)
    emilia = sorted(r["uid"] for r in out if r["event"] == "emilia")
    screen = set(np.random.default_rng(0).choice(emilia, min(SCREEN_N, len(emilia)), replace=False).tolist())
    for r in out:
        r["screen"] = r["uid"] in screen
    json.dump(out, open(SEL_PATH, "w"), indent=1)
    print(f"{len(out)} units; {len(screen)} screening (Emilia-Romagna), {len(out) - len(screen)} held out", flush=True)


def _tile_masks(rec):
    """Return the disturbed and undisturbed masks on the tile grid, at stride S."""
    from rasterio import features
    from scipy.ndimage import distance_transform_edt
    from lib.groundtruth import fetch_worldcover, fetch_terrain
    t, crs, H, W = tile_grid(rec["tile"])
    x0, y0, x1, y1 = tile_bbox(rec["tile"], 0.01)
    g = _polygons(rec["event"]).cx[x0:x1, y0:y1]
    ls = features.rasterize(((x, 1) for x in g.to_crs(crs).geometry), out_shape=(H, W), transform=t, fill=0,
                            dtype=np.uint8, all_touched=False).astype(bool)
    aux = LS / "aux"
    aux.mkdir(parents=True, exist_ok=True)
    wc_path, tr_path = aux / f"{rec['tile']}_worldcover.npy", aux / f"{rec['tile']}_terrain.npz"
    wc = fetch_worldcover(rec["tile"], out_path=wc_path)
    if not wc_path.exists():
        np.save(wc_path, wc)
    tr = fetch_terrain(rec["tile"], out_path=tr_path)
    if not tr_path.exists():
        np.savez_compressed(tr_path, **tr)
    slope = tr["slope"][:H, :W]
    wc = wc[:H, :W]
    land = ~np.isin(wc, (50, 70, 80)) & (wc > 0)
    far = distance_transform_edt(~ls) * 10 > 50
    steep = slope >= np.nanpercentile(slope[ls], 25) if ls.any() else np.zeros_like(ls)
    ref = land & far & steep
    return ls[::S, ::S], ref[::S, ::S]


def fetch_one(rec):
    out = _npz(rec)
    if out.exists():
        return rec["uid"], "cached"
    try:
        download_landmask(rec["tile"])
        ls, ref = _tile_masks(rec)
        rng = np.random.default_rng(zlib.crc32(rec["uid"].encode()))
        store = sample_tile_years(rec["tile"], rec["years"], {"dmg": ls, "ref": ref}, S, CAP, N_TRAIN, rng)
        if not store:
            return rec["uid"], "no years"
        PIX.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **store)
        return rec["uid"], ""
    except Exception as e:
        return rec["uid"], repr(e)


def phase_fetch():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch_one, SEL))
    bad = [(u, e) for u, e in res if e and e != "cached"]
    print(f"{len(res) - len(bad)}/{len(res)} units sampled; failed: {bad[:10]}", flush=True)


def phase_train():
    if CKPT.exists():
        print("cached"); return
    X = corpus([_npz(r) for r in SEL])
    print(f"landslide corpus {X.shape}", flush=True)
    train_and_save(X, CKPT)


def phase_screen():
    import torch
    sae = load_sae(path=CKPT)
    torch.set_num_threads(16)
    with open(JSONL, "w") as out:
        for r in SEL:
            if not _npz(r).exists():
                out.write(json.dumps(dict(ok=False, uid=r["uid"], tile=r["tile"], error="not sampled")) + "\n")
                continue
            z = np.load(_npz(r))
            dmg, ref, n = {}, {}, {}
            for y in YEARS:
                if f"{y}_dmg" not in z.files:
                    continue
                (a, na), (b, nb) = mean_activation(sae, z[f"{y}_dmg"], 300), mean_activation(sae, z[f"{y}_ref"], 2000)
                if a is None or b is None:
                    continue
                dmg[str(y)], ref[str(y)], n[str(y)] = a.tolist(), b.tolist(), [na, nb]
            ok = {str(r["year"]), str(r["year"] + 1)} <= set(dmg)
            out.write(json.dumps(dict(ok=ok, uid=r["uid"], event=r["event"], tile=r["tile"], year=r["year"], ha=r["ha"],
                                      screen=r["screen"], n=n, dmg=dmg, ref=ref,
                                      **({} if ok else {"error": "below the pixel floors or a gap at the window"}))) + "\n")
    verdicts()


def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    scr, held = [r for r in rows if r["screen"]], [r for r in rows if not r["screen"]]
    fam = anchored_family(aggregate_contrasts(row_units(scr)), window=WINDOW)
    v = peak_outcome(row_contrasts(held), fam, WINDOW) if fam else {}
    summ = outcome_summary(v, {r["uid"]: r["event"] for r in held})
    print(f"{len(scr)} screening tiles, {len(held)} held out; family {len(fam)} features {fam}")
    for k, s in summ.items():
        print(f"  {k}: dated to {{Y, Y+1}} {s['rate']:.3f} (n={s['n']}; Y share among dated {s['y0_share']:.2f})")
    json.dump(dict(family=fam, n_screen=len(scr), n_held=len(v), rate=summ["all"]["rate"], summary=summ,
                   outcomes={u: dict(peak=pk, ok=ok) for u, (pk, ok) in v.items()}),
              open(RESULTS_DIR / "landslides.json", "w"), indent=1, default=float)
    print("saved results/landslides.json")


if __name__ == "__main__":
    {"select": phase_select, "fetch": phase_fetch, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
