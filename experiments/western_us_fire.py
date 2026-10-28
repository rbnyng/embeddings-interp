"""Western US fire: test the Dixie burn family, unchanged, on large fires in Tessera and AlphaEarth.

Each unit is one MTBS wildfire or complex of at least 20,000 acres, from 2018-2023, centered at longitude -125 to -102
and latitude 31 to 49. It must cover at least 15% of the tile it overlaps most, and that tile must have Tessera for
Y-1 through Y+2. Disturbed pixels are inside the MTBS perimeter. Undisturbed pixels are the rest of the tile. Groups
are at stride 2 and a fire needs at least 500 burned pixels. Fires with fewer than 500 undisturbed pixels are scored on
the burned mean alone (mode=raw). The expected window is the year after the fire (Y+1). A fire is dated in the window
when the family's mean contrast over 2017-2025 peaks at Y+1. The AlphaEarth run scores the AlphaEarth Dixie burn
family on the same fires and masks. perfeat stores every feature's contrast without raw mode, on fires with at least
300 burned and 300 unburned pixels.

Usage: python experiments/western_us_fire.py inventory|download|screen|verdicts|perfeat|aef_screen|aef_verdicts
Writes results/mtbs_inventory.json, western_us_fire.jsonl, western_us_fire_perfeat.jsonl, western_us_fire_aef.jsonl and
western_us_fire_download_errors.json.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import torch
from lib.alphaearth import compute_scale, load_aef
from lib.config import CKPT_DIR, BURN_FAMILY, GT_DIR, RESULTS_DIR, workers
from lib.tiles import (download_tile, download_landmask, load_tile,
                                       tile_grid)
from lib.sae import load_sae, group_sums

YEARS = list(range(2017, 2026))
S = 2
_INV_PATH = RESULTS_DIR / "mtbs_inventory.json"
INV = json.load(open(_INV_PATH)) if _INV_PATH.exists() else {"fires": []}
FIRES = [r for r in INV["fires"] if r["usable"]]
JSONL = RESULTS_DIR / "western_us_fire.jsonl"

# ---- download
def phase_download():
    tiles = sorted({r["tile"] for r in FIRES})
    need = [(t, y) for t in tiles for y in YEARS]
    print(f"{len(tiles)} tiles x {len(YEARS)} years = {len(need)} tile-years "
          f"(~{len(need) * 0.117:.0f} GB)", flush=True)

    def one_mask(t):
        try:
            download_landmask(t)
            return None
        except Exception as e:
            return f"{t} landmask: {e!r}"

    def one(job):
        t, y = job
        try:
            download_tile(t, y)
            return None
        except Exception as e:
            return f"{t} {y}: {e!r}"

    errs = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(one_mask, tiles):
            if r:
                errs.append(r)
        for i, r in enumerate(ex.map(one, need), 1):
            if r:
                errs.append(r)
            if i % 100 == 0:
                print(f"  [{i}/{len(need)}]  ({len(errs)} errors)", flush=True)
    print(f"download done, {len(errs)} errors", flush=True)
    for e in errs[:20]:
        print("  ERR", e, flush=True)
    json.dump(errs, open(RESULTS_DIR / "western_us_fire_download_errors.json", "w"))

# ---- screen and verdicts
def fire_mask(rec):
    import geopandas as gpd
    from rasterio import features
    g = gpd.read_file(GT_DIR / "mtbs" / "mtbs_perims_DD.shp",
                      where=f"event_id = '{rec['event_id']}'")
    if len(g) == 0:
        raise RuntimeError(f"event {rec['event_id']} not found")
    t, crs, H, W = tile_grid(rec["tile"])
    geom = g.to_crs(crs).geometry.iloc[0]
    burned = features.rasterize([(geom, 1)], out_shape=(H, W), transform=t,
                                fill=0, dtype=np.uint8).astype(bool)
    return burned[::S, ::S], ~burned[::S, ::S]

def screen_fire(rec):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae = load_sae()
        b, u = fire_mask(rec)
        fam = list(BURN_FAMILY)
        from lib.config import EMB_DIR
        from lib.tiles import REPR_DIR
        years = [y for y in YEARS
                 if (EMB_DIR / REPR_DIR / str(y) / rec["tile"] /
                     f"{rec['tile']}.npy").exists()]
        if not {rec["year"], rec["year"] + 1, rec["year"] + 2} <= set(years):
            return dict(ok=False, event_id=rec["event_id"], error="gap at fire window")
        traj_fam, con_all = {}, {}
        for y in years:
            x = load_tile(rec["tile"], y, stride=S)
            H, W, _ = x.shape
            bb, uu = b[:H, :W], u[:H, :W]
            if bb.sum() < 500:
                return dict(ok=False, event_id=rec["event_id"], error="mask too small")
            # A fire that fills its tile has no undisturbed pixels. Score the burned mean alone and tag the fire
            # mode=raw.
            # This is safe because the family's undisturbed mean is about 0 across all fires.
            raw_mode = uu.sum() < 500
            s, n = group_sums(sae, x, {"burn": bb, "unburn": uu})
            con = s["burn"] / n["burn"] - (s["unburn"] / n["unburn"] if not raw_mode else 0.0)
            con_all[y] = np.round(con, 5).tolist()     # keep every feature so another family can be scored without re-encoding
            traj_fam[y] = float(con[fam].mean())
            del x
        fy = rec["year"]
        peak_fam = max(traj_fam, key=traj_fam.get)
        return dict(ok=True, mode=("raw" if raw_mode else "contrast"),
                    event_id=rec["event_id"], name=rec["name"], year=fy,
                    ig_date=rec["ig_date"], acres=rec["acres"], lat=rec["lat"],
                    lon=rec["lon"], tile=rec["tile"],
                    burn_frac=rec["burn_frac_of_tile"],
                    dated=bool(peak_fam == fy + 1),
                    peak=int(peak_fam),
                    fam={str(y): v for y, v in traj_fam.items()},
                    con={str(y): v for y, v in con_all.items()})
    except Exception as e:
        return dict(ok=False, event_id=rec["event_id"], error=repr(e))

def phase_screen():
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                done.add(json.loads(line)["event_id"])
            except Exception:
                pass
    todo = [r for r in FIRES if r["event_id"] not in done]
    print(f"{len(FIRES)} fires, {len(done)} already screened, {len(todo)} to go",
          flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(100)) as ex:
        for i, r in enumerate(ex.map(screen_fire, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if r.get("ok"):
                print(f"[{i:3d}/{len(todo)}] {r['name'][:24]:24s} {r['year']} "
                      f"fam@{r['peak']} {'OK' if r['dated'] else 'MISS'}",
                      flush=True)
            else:
                print(f"[{i:3d}/{len(todo)}] {r['event_id']} FAILED {r['error'][:60]}",
                      flush=True)
    out.close()
    verdicts()

def verdicts():
    rows = [json.loads(l) for l in open(JSONL)]
    ok = [r for r in rows if r.get("ok")]
    n_raw = sum(r.get("mode") == "raw" for r in ok)
    print(f"\n{len(ok)}/{len(rows)} fires screened cleanly ({n_raw} in raw mode)")
    if not ok:
        return
    rate = float(np.mean([r["dated"] for r in ok]))
    print(f"family validation rate: {rate:.3f}")


# ---- inventory

def phase_inventory():
    import geopandas as gpd
    import requests
    from lib.tiles import embedding_url
    YEARS = list(range(2017, 2026))
    MIN_ACRES = 20_000
    WEST = (-125.0, -102.0, 31.0, 49.0)
    FIRE_YEARS = range(2018, 2024)
    print("loading MTBS perimeters ...", flush=True)
    g = gpd.read_file(GT_DIR / "mtbs" / "mtbs_perims_DD.shp")
    g["ig_date"] = gpd.pd.to_datetime(g["ig_date"])
    g["year"] = g["ig_date"].dt.year
    for c in ("burnbndlon", "burnbndlat", "burnbndac"):
        g[c] = gpd.pd.to_numeric(g[c], errors="coerce")

    sel = (g["incid_type"].isin(["Wildfire", "Complex"])
           & g["year"].isin(FIRE_YEARS)
           & (g["burnbndac"] >= MIN_ACRES)
           & g["burnbndlon"].between(WEST[0], WEST[1])
           & g["burnbndlat"].between(WEST[2], WEST[3]))

    f = g[sel].copy()
    print(f"candidate fires (wildfire/complex, >= {MIN_ACRES} ac, west, {FIRE_YEARS.start}-"
          f"{FIRE_YEARS.stop-1}): {len(f)}", flush=True)

    from lib.tiles import snap

    def tiles_touched(geom):
        lonmin, latmin, lonmax, latmax = geom.bounds
        out = []
        for lo in np.arange(snap(lonmin), snap(lonmax) + 0.1, 0.1):
            for la in np.arange(snap(latmin), snap(latmax) + 0.1, 0.1):
                out.append((round(lo, 2), round(la, 2)))
        return out

    from shapely.geometry import box
    records = []
    for _, r in f.iterrows():
        best, best_frac = None, 0.0
        for lo, la in tiles_touched(r.geometry):
            b = box(lo - 0.05, la - 0.05, lo + 0.05, la + 0.05)
            try:
                frac = r.geometry.intersection(b).area / b.area
            except Exception:
                continue
            if frac > best_frac:
                best, best_frac = (lo, la), frac
        if best is None or best_frac < 0.15:
            continue
        records.append(dict(event_id=r["event_id"], name=r["incid_name"],
                            year=int(r["year"]), acres=float(r["burnbndac"]),
                            ig_date=str(r["ig_date"].date()),
                            tile=f"grid_{best[0]}_{best[1]}",
                            burn_frac_of_tile=round(best_frac, 3),
                            lon=float(r["burnbndlon"]), lat=float(r["burnbndlat"])))

    print(f"fires burning >=15% of a tile: {len(records)}", flush=True)
    tiles = sorted({r["tile"] for r in records})
    print(f"distinct dominant tiles: {len(tiles)} -> {len(tiles) * len(YEARS)} HEADs",
          flush=True)

    def head(job):
        tile, year = job
        try:
            rr = requests.head(embedding_url(tile, year), timeout=30,
                               allow_redirects=True)
            return tile, year, rr.status_code == 200
        except Exception:
            return tile, year, False

    cov = {t: set() for t in tiles}
    with ThreadPoolExecutor(max_workers=24) as ex:
        for tile, year, hit in ex.map(head, [(t, y) for t in tiles for y in YEARS]):
            if hit:
                cov[tile].add(year)

    for r in records:
        ys = cov[r["tile"]]
        r["years_available"] = sorted(ys)
        r["usable"] = bool({r["year"] - 1, r["year"], r["year"] + 1, r["year"] + 2} <= ys)

    usable = [r for r in records if r["usable"]]
    print(f"\nfires with pre-1 / fire / post-1 / post-2 coverage: {len(usable)} of {len(records)}")
    by_year = {}
    for r in usable:
        by_year.setdefault(r["year"], []).append(r)

    print(f"{'year':>6} {'fires':>6}   largest")
    for y in sorted(by_year):
        top = max(by_year[y], key=lambda r: r["acres"])
        print(f"{y:>6} {len(by_year[y]):>6}   {top['name']} ({top['acres']:,.0f} ac, "
              f"{top['burn_frac_of_tile']:.0%} of {top['tile']})")

    lat_bands = {"south (31-37N)": (31, 37), "central (37-42N)": (37, 42),
                 "north (42-49N)": (42, 49)}

    print("\nby latitude band (ecoregion proxy):")
    for name, (lo, hi) in lat_bands.items():
        n = sum(lo <= r["lat"] < hi for r in usable)
        print(f"  {name}: {n}")

    full = [r for r in usable if len(r["years_available"]) == len(YEARS)]
    print(f"\nfires whose tile has ALL years 2017-2025: {len(full)}")
    json.dump(dict(min_acres=MIN_ACRES, n_candidates=len(f), n_tile_dominant=len(records),
                   n_usable=len(usable), n_full_coverage=len(full), fires=records),
              open(RESULTS_DIR / "mtbs_inventory.json", "w"))

    print("\nsaved results/mtbs_inventory.json")


def _perfeat_one(rec):
    """Burned minus unburned contrast of every feature, in every year, for one western US fire."""
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae = load_sae()
        b, u = fire_mask(rec)
        con = {}
        for y in YEARS:
            try:
                x = load_tile(rec["tile"], y, stride=S)
            except Exception:
                continue
            H, W, _ = x.shape
            bb, uu = b[:H, :W], u[:H, :W]
            if bb.sum() < 300 or uu.sum() < 300:
                return dict(ok=False, event_id=rec["event_id"], error="mask")
            s, n = group_sums(sae, x, {"burn": bb, "unburn": uu})
            con[str(y)] = (s["burn"] / n["burn"] - s["unburn"] / n["unburn"]).tolist()
            del x
        if not {str(rec["year"]), str(rec["year"] + 1)} <= set(con):
            return dict(ok=False, event_id=rec["event_id"], error="gap")
        return dict(ok=True, event_id=rec["event_id"], year=rec["year"], con=con)
    except Exception as e:
        return dict(ok=False, event_id=rec["event_id"], error=repr(e))


def phase_perfeat():
    OUT = RESULTS_DIR / "western_us_fire_perfeat.jsonl"
    done = set()
    if OUT.exists():
        for line in open(OUT):
            try:
                done.add(json.loads(line)["event_id"])
            except Exception:
                pass
    todo = [r for r in FIRES if r["event_id"] not in done]
    print(f"{len(todo)} fires to re-screen", flush=True)
    f = open(OUT, "a")
    with ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for i, r in enumerate(ex.map(_perfeat_one, todo), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 20 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    f.close()
    print("per-feature screen done", flush=True)



from lib.measure import aef_dixie_family

AEF_JSONL = RESULTS_DIR / "western_us_fire_aef.jsonl"

def aef_screen_tile(args):
    """Score every fire on one tile on AlphaEarth, then remove the tile's cached years (about 63 MB each).

    One worker per tile, so no worker deletes files another is still reading."""
    from lib.alphaearth import AEF_DIR
    recs, scale = args
    try:
        return [_aef_screen_fire((rec, scale)) for rec in recs]
    finally:
        for f in AEF_DIR.glob(f"aef_{recs[0]['tile']}_*.npy"):
            f.unlink(missing_ok=True)

def _aef_screen_fire(args):
    rec, scale = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae = load_sae(path=CKPT_DIR / "sae_aef_dixie.pt", d=64, m=512, k=8)
        b, u = fire_mask(rec)
        fam = [int(f) for f in aef_dixie_family()]
        traj, con_all = {}, {}
        for y in YEARS:
            try:
                x = np.nan_to_num(load_aef(rec["tile"], y, stride=S)) * scale
            except Exception:
                continue
            H, W, _ = x.shape
            bb, uu = b[:H, :W], u[:H, :W]
            if bb.sum() < 500:
                return dict(ok=False, event_id=rec["event_id"], error="no burn px")
            raw_mode = uu.sum() < 500
            s, n = group_sums(sae, x, {"burn": bb} if raw_mode else {"burn": bb, "unburn": uu})
            con = s["burn"] / n["burn"] - (s["unburn"] / n["unburn"] if not raw_mode else 0.0)
            con_all[y] = np.round(con, 5).tolist()
            traj[y] = float(con[fam].mean())
            del x
        fy = rec["year"]
        need = {fy, fy + 1, fy + 2}
        if not need <= set(traj):
            return dict(ok=False, event_id=rec["event_id"], error="gap at fire window")
        peak = max(traj, key=traj.get)
        return dict(ok=True, mode=("raw" if raw_mode else "contrast"),
                    event_id=rec["event_id"], name=rec["name"], year=fy,
                    ig_date=rec["ig_date"], lat=rec["lat"],
                    dated=bool(peak == fy + 1), peak=int(peak),
                    traj={str(y): v for y, v in traj.items()},
                    con={str(y): v for y, v in con_all.items()})
    except Exception as e:
        return dict(ok=False, event_id=rec["event_id"], error=repr(e))

def phase_aef_screen():
    scale = compute_scale()
    print(f"SCALE x{scale:.1f}; AlphaEarth family {aef_dixie_family()}", flush=True)
    done = set()
    if AEF_JSONL.exists():
        for line in open(AEF_JSONL):
            try:
                done.add(json.loads(line)["event_id"])
            except Exception:
                pass
    todo = [r for r in FIRES if r["event_id"] not in done]
    by_tile = {}
    for r in todo:
        by_tile.setdefault(r["tile"], []).append(r)
    print(f"{len(todo)} fires to screen on {len(by_tile)} tiles", flush=True)
    out = open(AEF_JSONL, "a")
    n = 0
    with ProcessPoolExecutor(max_workers=workers(80)) as ex:
        for rows in ex.map(aef_screen_tile, [(recs, scale) for recs in by_tile.values()]):
            for r in rows:
                out.write(json.dumps(r) + "\n")
                n += 1
            out.flush()
            print(f"  [{n}/{len(todo)}]", flush=True)
    out.close()
    aef_verdicts()

def aef_verdicts():
    aef = {r["event_id"]: r for r in map(json.loads, open(AEF_JSONL)) if r.get("ok")}
    tes = {r["event_id"]: r for r in
           map(json.loads, open(RESULTS_DIR / "western_us_fire.jsonl")) if r.get("ok")}
    both = sorted(set(aef) & set(tes))
    a_con = [e for e in both if aef[e]["mode"] == "contrast"]
    print(f"\n{len(aef)} AlphaEarth-screened, {len(both)} matched with Tessera, "
          f"{len(a_con)} contrast-mode")
    if not a_con:
        return
    rate = float(np.mean([aef[e]["dated"] for e in a_con]))
    print(f"AlphaEarth family validation rate (contrast-mode fires): {rate:.3f}")

if __name__ == "__main__":
    {"inventory": phase_inventory, "download": phase_download, "screen": phase_screen, "perfeat": phase_perfeat,
     "verdicts": verdicts, "aef_screen": phase_aef_screen, "aef_verdicts": aef_verdicts}[sys.argv[1]]()
