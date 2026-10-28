"""SE US wildfire and prescribed fire: screen MTBS prescribed fires and same-size wildfires in Tessera and AlphaEarth.

Each unit is one MTBS event, picked from MTBS alone: 1,000-5,000 acres, 2018-2023, centered at longitude -100 to -80
and latitude 24 to 40. It must have burned only once: its perimeter touches no other MTBS perimeter of 2017-2025 in
that band. Events are taken largest first, one per tile and at most 100 per fire type. Disturbed pixels are inside the
perimeter. Undisturbed pixels are the rest of the tile, outside every nearby MTBS perimeter of any year. Groups are at
stride 2 and need at least 300 and 2,000 pixels. Each model has its own SE dictionary, trained on 60 random selected
tiles. The wildfire family is screened on the 2018 wildfires and the prescribed family on the 2018-2019 prescribed
fires, with the anchored bar and the expected window Y to Y+1. Later events are held out. The AlphaEarth run screens
only the wildfire family, on the same events and masks. select_nocap and screen_extra add the events left out by the
cap of 100 per type, as more held-out units.

Usage: python experiments/se_us_fire.py select|train|screen|verdicts
       python experiments/se_us_fire.py aef_train|aef_screen|aef_verdicts
       python experiments/se_us_fire.py select_nocap|screen_extra
Writes results/se_us_fire_selection.json and se_us_fire_geoms.geojson (and _nocap versions), se_us_fire.jsonl/.json,
se_us_fire_aef.jsonl/.json and se_us_fire_extra.jsonl. Caches the dictionaries in data/checkpoints/se_us_fire/.
"""
import json
import os
import sys

import numpy as np
import torch
from lib.alphaearth import compute_scale, load_aef
from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.tiles import load_tile, tile_grid, sample_pixels
from lib.sae import load_sae, group_sums, train_and_save
from lib.settings import BY_KEY
from lib.screen import aggregate_contrasts, anchored_family

YEARS = list(range(2017, 2026))
S = 2
M = 1024
CKPT = CKPT_DIR / "se_us_fire" / "sae_se.pt"
JSONL = RESULTS_DIR / "se_us_fire.jsonl"
_SEL_PATH = RESULTS_DIR / "se_us_fire_selection.json"
# Event perimeters. screen_extra points this at se_us_fire_geoms_nocap.geojson instead.
GEOMS = RESULTS_DIR / "se_us_fire_geoms.geojson"
GEOMS_NOCAP = RESULTS_DIR / "se_us_fire_geoms_nocap.geojson"     # the uncapped selection, for screen_extra
# Empty until `select` has run, so that phase still works on a fresh checkout.
SEL = json.load(open(_SEL_PATH)) if _SEL_PATH.exists() else []

def phase_train():
    if CKPT.exists():
        print("cached"); return
    rng = np.random.default_rng(0)
    tiles = sorted({r["tile"] for r in SEL})
    pick = rng.choice(tiles, size=min(60, len(tiles)), replace=False)
    X = np.concatenate([sample_pixels([(t, y) for y in YEARS], n_per=6000, seed=0)
                        for t in pick])
    rng.shuffle(X)
    print(f"SE corpus {X.shape}", flush=True)
    train_and_save(X, CKPT, epochs=40, seed=0, log=True)
    print("trained", flush=True)

def event_masks(rec, others, geoms=GEOMS):
    """Return the disturbed and undisturbed masks at stride S. The undisturbed mask leaves out every perimeter in
    `others`."""
    import geopandas as gpd
    from rasterio import features
    g = gpd.read_file(geoms)
    g = g[g.event_id == rec["event_id"]]
    t, crs, H, W = tile_grid(rec["tile"])
    geom = g.to_crs(crs).geometry.iloc[0]
    burned = features.rasterize([(geom, 1)], out_shape=(H, W), transform=t,
                                fill=0, dtype=np.uint8).astype(bool)
    ref = ~burned
    oth = others[others.event_id != rec["event_id"]]
    if len(oth):
        o = features.rasterize(((gg, 1) for gg in oth.to_crs(crs).geometry),
                               out_shape=(H, W), transform=t, fill=0,
                               dtype=np.uint8).astype(bool)
        ref &= ~o
    return burned[::S, ::S], ref[::S, ::S]

def run_event(rec, years=YEARS, geoms=GEOMS):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import geopandas as gpd
        lon, lat = rec["lon"], rec["lat"]
        others = gpd.read_file(
            GT_DIR / "mtbs" / "mtbs_perims_DD.shp",
            bbox=(lon - 0.08, lat - 0.08, lon + 0.08, lat + 0.08))
        b, u = event_masks(rec, others, geoms)
        if b.sum() < 300 or u.sum() < 2000:
            return dict(ok=False, event_id=rec["event_id"], error="mask too small")
        sae = load_sae(path=CKPT)
        traj_b, traj_u = {}, {}
        for y in years:
            try:
                x = load_tile(rec["tile"], y, stride=S)
            except Exception:
                continue
            s, n = group_sums(sae, x, {"burn": b, "unburn": u})
            traj_b[y] = (s["burn"] / n["burn"]).tolist()
            traj_u[y] = (s["unburn"] / n["unburn"]).tolist()
            del x
        if not {rec["year"], rec["year"] + 1} <= set(traj_b):
            return dict(ok=False, event_id=rec["event_id"], error="gap at window")
        return dict(ok=True, **{k: rec[k] for k in
                                ("event_id", "fire_type", "tile", "year", "ig_date",
                                 "acres", "lon", "lat")},
                    burn={str(y): v for y, v in traj_b.items()},
                    ref={str(y): v for y, v in traj_u.items()})
    except Exception as e:
        return dict(ok=False, event_id=rec["event_id"], error=repr(e))

def phase_screen():
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                done.add(json.loads(line)["event_id"])
            except Exception:
                pass
    todo = [r for r in SEL if r["event_id"] not in done]
    print(f"{len(todo)} events to screen", flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for i, r in enumerate(ex.map(run_event, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    verdicts()

def blind_family(rows, years):
    rs = [r for r in rows if r["year"] in years]
    if len(rs) < 5:
        return [], len(rs)
    # Years are counted from the event. m=M checks the width of the stored tables.
    units = [{y - r["year"]: (r["burn"][str(y)], r["ref"][str(y)])
              for y in YEARS if str(y) in r["burn"]} for r in rs]
    med = aggregate_contrasts(units, min_units=BY_KEY["se_us_fire"].min_units, m=M)
    return anchored_family(med), len(rs)

def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    wf = [r for r in rows if r["fire_type"] == "wildfire"]
    rx = [r for r in rows if r["fire_type"] == "prescribed"]
    print(f"\n{len(wf)} wildfire + {len(rx)} prescribed events screened")
    fam, nscr = blind_family(wf, {2018})
    print(f"wildfire family (2018 events, n={nscr}): {len(fam)} features")
    if not fam:
        print("!! no family - stopping")
        return
    famx, nx = blind_family(rx, {2018, 2019})
    print(f"prescribed family (2018-2019 events, n={nx}): {len(famx)} features")
    json.dump(dict(family=fam, family_rx=famx), open(RESULTS_DIR / "se_us_fire.json", "w"))
    print("\nsaved results/se_us_fire.json")


# ---- selection

def phase_select(cap=100, tag=""):
    """Pick prescribed fires and wildfires from MTBS alone, matched on size and region, that burned only once.

    Keeps one event per tile, largest first, and at most `cap` per type (None means no cap). Writes
    se_us_fire_selection{tag}.json and se_us_fire_geoms{tag}.geojson.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import geopandas as gpd
    import pandas as pd
    BAND = dict(lon=(-100, -80), lat=(24, 40))
    ACRES = (1000, 5000)
    YEARS = (2018, 2023)
    m = gpd.read_file(GT_DIR / "mtbs" / "mtbs_perims_DD.shp")
    for c in ("burnbndac", "burnbndlat", "burnbndlon"):
        m[c] = pd.to_numeric(m[c], errors="coerce")

    m["yr"] = pd.to_datetime(m.ig_date, errors="coerce").dt.year
    m = m[m.yr.between(2017, 2025) & m.burnbndlon.between(*BAND["lon"])
          & m.burnbndlat.between(*BAND["lat"])].reset_index(drop=True)

    print(f"{len(m):,} MTBS perimeters in the band 2017-2025")
    cand = m[m.yr.between(*YEARS) & m.burnbndac.between(*ACRES)
             & m.incid_type.isin(["Prescribed Fire", "Wildfire"])]

    print(f"candidates in {ACRES} ac, {YEARS}: "
          f"{(cand.incid_type == 'Prescribed Fire').sum()} prescribed, "
          f"{(cand.incid_type == 'Wildfire').sum()} wildfire")

    sidx = m.sindex
    iso = []
    for i, row in cand.iterrows():
        hits = [j for j in sidx.query(row.geometry, predicate="intersects") if j != i]
        if not any(m.geometry.iloc[j].intersects(row.geometry) for j in hits):
            iso.append(i)

    cand = cand.loc[iso]
    print(f"single-burn survivors: "
          f"{(cand.incid_type == 'Prescribed Fire').sum()} prescribed, "
          f"{(cand.incid_type == 'Wildfire').sum()} wildfire")

    from lib.tiles import tile_for_point
    out, geoms = [], []
    for incid, name in [("Prescribed Fire", "prescribed"), ("Wildfire", "wildfire")]:
        sub = cand[cand.incid_type == incid].sort_values("burnbndac", ascending=False)
        seen = set()
        for _, r in sub.iterrows():
            t = tile_for_point(r.burnbndlon, r.burnbndlat)
            if t in seen:
                continue
            seen.add(t)
            out.append(dict(event_id=r.event_id, name=str(r.incid_name), fire_type=name,
                            tile=t, year=int(r.yr), ig_date=str(r.ig_date)[:10],
                            acres=float(r.burnbndac), lon=float(r.burnbndlon),
                            lat=float(r.burnbndlat)))
            geoms.append(r.geometry)
            if cap and len([o for o in out if o["fire_type"] == name]) >= cap:
                break

    sel = gpd.GeoDataFrame(out, geometry=geoms, crs=m.crs)
    sel.to_file(RESULTS_DIR / f"se_us_fire_geoms{tag}.geojson", driver="GeoJSON")
    json.dump(out, open(RESULTS_DIR / f"se_us_fire_selection{tag}.json", "w"))
    df = pd.DataFrame(out)
    print(f"\nselected {len(df)} events on {df.tile.nunique()} tiles")
    print(df.groupby("fire_type").agg(n=("tile", "size"), med_ac=("acres", "median"),
                                mo=("ig_date", lambda s: s.str[5:7].mode()[0])))

    print("by year:", df.groupby(["fire_type", "year"]).size().unstack(fill_value=0).to_dict("index"))


# ---- AlphaEarth run: the same events, masks and protocol, with an SE AlphaEarth dictionary

AEF_M = 512

AEF_CKPT = CKPT_DIR / "se_us_fire" / "sae_se_aef.pt"

AEF_JSONL = RESULTS_DIR / "se_us_fire_aef.jsonl"

def phase_aef_train():
    if AEF_CKPT.exists():
        print("cached"); return
    rng = np.random.default_rng(0)
    scale = compute_scale()
    print(f"SCALE x{scale:.1f}", flush=True)
    tiles = sorted({r["tile"] for r in SEL})
    pick = rng.choice(tiles, size=min(60, len(tiles)), replace=False)
    Xs = []
    for t in pick:
        for y in YEARS[::2]:
            try:
                x = np.nan_to_num(load_aef(t, y, stride=8)).reshape(-1, 64)
            except Exception:
                continue
            idx = rng.choice(len(x), size=min(9000, len(x)), replace=False)
            Xs.append(x[idx])
    X = (np.concatenate(Xs) * scale).astype(np.float32)
    rng.shuffle(X)
    print(f"SE US AlphaEarth corpus {X.shape} (scaled)", flush=True)
    train_and_save(X, AEF_CKPT, epochs=40, seed=0, log=True, d=64, m=AEF_M, k=8)
    print("trained", flush=True)

def aef_run_event(args):
    rec, scale = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import geopandas as gpd
        lon, lat = rec["lon"], rec["lat"]
        others = gpd.read_file(
            GT_DIR / "mtbs" / "mtbs_perims_DD.shp",
            bbox=(lon - 0.08, lat - 0.08, lon + 0.08, lat + 0.08))
        b, u = event_masks(rec, others)          # same masks as the Tessera run
        if b.sum() < 300 or u.sum() < 2000:
            return dict(ok=False, event_id=rec["event_id"], error="mask too small")
        sae = load_sae(path=AEF_CKPT, d=64, m=AEF_M, k=8)
        traj_b, traj_u = {}, {}
        for y in YEARS:
            try:
                x = np.nan_to_num(load_aef(rec["tile"], y, stride=S)) * scale
            except Exception:
                continue
            s, n = group_sums(sae, x, {"burn": b, "unburn": u})
            traj_b[y] = (s["burn"] / n["burn"]).tolist()
            traj_u[y] = (s["unburn"] / n["unburn"]).tolist()
            del x
        if not {rec["year"], rec["year"] + 1} <= set(traj_b):
            return dict(ok=False, event_id=rec["event_id"], error="gap at window")
        return dict(ok=True, **{k: rec[k] for k in
                                ("event_id", "fire_type", "tile", "year", "ig_date",
                                 "acres", "lon", "lat")},
                    burn={str(y): v for y, v in traj_b.items()},
                    ref={str(y): v for y, v in traj_u.items()})
    except Exception as e:
        return dict(ok=False, event_id=rec["event_id"], error=repr(e))

def phase_aef_screen():
    from concurrent.futures import ProcessPoolExecutor
    scale = compute_scale()
    print(f"SCALE x{scale:.1f}", flush=True)
    done = set()
    if AEF_JSONL.exists():
        for line in open(AEF_JSONL):
            try:
                done.add(json.loads(line)["event_id"])
            except Exception:
                pass
    todo = [r for r in SEL if r["event_id"] not in done]
    print(f"{len(todo)} events to screen", flush=True)
    out = open(AEF_JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for i, r in enumerate(ex.map(aef_run_event, [(t, scale) for t in todo]), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    aef_verdicts()

def aef_blind_family(rows, years):
    rs = [r for r in rows if r["year"] in years]
    if len(rs) < 5:
        return [], len(rs)
    # Years are counted from the event, with the same unit floor as the Tessera screen. m=AEF_M checks the width of the
    # stored tables.
    units = [{y - r["year"]: (r["burn"][str(y)], r["ref"][str(y)])
              for y in YEARS if str(y) in r["burn"]} for r in rs]
    med = aggregate_contrasts(units, min_units=BY_KEY["se_us_fire"].min_units, m=AEF_M)
    return anchored_family(med), len(rs)

def aef_verdicts():
    rows = [r for r in map(json.loads, open(AEF_JSONL)) if r.get("ok")]
    wf = [r for r in rows if r["fire_type"] == "wildfire"]
    rx = [r for r in rows if r["fire_type"] == "prescribed"]
    print(f"\n{len(wf)} wildfire + {len(rx)} prescribed events screened (AlphaEarth)")
    fam, nscr = aef_blind_family(wf, {2018})
    print(f"AlphaEarth wildfire family (2018 events, n={nscr}): {len(fam)} features")
    # An empty screen is a result, so it is written too.
    json.dump(dict(family=fam, n_screen_events=int(nscr)), open(RESULTS_DIR / "se_us_fire_aef.json", "w"))
    print("saved results/se_us_fire_aef.json")

# ---- uncapped extension. select_nocap repeats the selection without the cap. screen_extra encodes the events the cap
# left out, over 2017-2025, into se_us_fire_extra.jsonl. They are held-out units for the families screened above.
# Tile-years not already cached are downloaded and deleted after use, so at most `n_workers` events are on disk at once.
EXTRA_JSONL = RESULTS_DIR / "se_us_fire_extra.jsonl"

def phase_select_nocap():
    phase_select(cap=None, tag="_nocap")

def run_event_extra(rec):
    import shutil
    from lib.tiles import download_landmask, download_tile, EMB_DIR, REPR_DIR
    fetched = []
    try:
        for y in YEARS:
            d = EMB_DIR / REPR_DIR / str(y) / rec["tile"]
            if (d / f"{rec['tile']}.npy").exists():
                continue
            fetched.append(d)              # listed before the download, so a partial download is removed too
            try:
                download_tile(rec["tile"], y)
            except Exception:
                # run_event skips a missing year, but fails the event if that year is in the expected window
                continue
        download_landmask(rec["tile"])
        return run_event(rec, geoms=GEOMS_NOCAP)
    except Exception as e:
        return dict(ok=False, event_id=rec["event_id"], error=repr(e))
    finally:
        for d in fetched:
            shutil.rmtree(d, ignore_errors=True)

def phase_screen_extra(n_workers=6):
    from concurrent.futures import ProcessPoolExecutor
    nocap = json.load(open(RESULTS_DIR / "se_us_fire_selection_nocap.json"))
    main = {r["event_id"] for r in SEL}
    done = set()
    if EXTRA_JSONL.exists():
        # only successes count as done, so failures are retried
        done = {r["event_id"] for r in map(json.loads, open(EXTRA_JSONL))
                if r.get("ok")}
    # Take one event per tile per run. Two workers downloading and deleting the same tile folders would clash.
    # A skipped event is picked up by the next run.
    todo, seen = [], set()
    for r in nocap:
        if r["event_id"] not in main | done and r["tile"] not in seen:
            todo.append(r); seen.add(r["tile"])
    print(f"{len(todo)} extra events to screen", flush=True)
    with open(EXTRA_JSONL, "a") as out, ProcessPoolExecutor(max_workers=workers(n_workers)) as ex:
        for i, r in enumerate(ex.map(run_event_extra, todo), 1):
            out.write(json.dumps(r) + "\n"); out.flush()
            if i % 10 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)

if __name__ == "__main__":
    {"select": phase_select, "train": phase_train,
     "screen": phase_screen, "verdicts": verdicts, "aef_train": phase_aef_train, "aef_screen": phase_aef_screen, "aef_verdicts": aef_verdicts, "select_nocap": phase_select_nocap,
     "screen_extra": phase_screen_extra}[sys.argv[1]]()
