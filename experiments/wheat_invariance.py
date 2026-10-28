"""Test whether a winter-wheat family found in the US southern Plains also picks out winter wheat in Europe.

The dictionary is trained only on the crop-atlas winter-wheat tiles (Kansas and Oklahoma). Pixel groups are at stride 2:
  US  USDA CDL winter wheat (code 24) against other cropland (codes 1-60 except 24)
  EU  JRC EU crop map common wheat (211, which also holds some spring wheat) against other arable land (212-290)
The family is screened on the US tiles with the crop atlas's anchored bar on the wheat minus other-cropland gap in
2018-2021. It is scored on the US tiles in the held-out years 2022-2024, and on the European tiles in the crop-map
years 2018 and 2022. The European tiles are the top three by 2018 common-wheat share in each of northern France,
central Germany, western Poland and Hungary.
  select   rank European candidate tiles on a 0.3 degree grid, keeping two spares per region
  fetch    download Tessera for the top three tiles per region, using the spares when a download fails
  train    train the dictionary on the US tiles
  screen   mean activations on wheat and other cropland per tile-year (groups of 400 pixels or more), then the family
  sample   up to 2,000 wheat and 2,000 other-cropland pixels per scored tile-year (200 or more), then runs analyze
  analyze  how often the family fires, the AUC of the family sum for wheat against other cropland (by side, map year
           and European region), the location R^2 among family-active wheat pixels and raw group-center distances

Usage: python experiments/wheat_invariance.py select|fetch|train|screen|sample|analyze
Writes results/wheat_selection.json, wheat_invariance_tables.jsonl, wheat_invariance_family.json, wheat_invariance.npz
and wheat_invariance.json, and caches the dictionary at data/checkpoints/wheat_invariance/sae_us_wheat.pt.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

from lib.config import CKPT_DIR, RESULTS_DIR, workers
from lib.sae import encode_array, load_sae, train_and_save
from lib.tiles import download_landmask, download_tile, load_tile, sample_pixels, tile_grid
import experiments.crop_atlas as ca
from lib.stats import group_r2

S = 2
US_WHEAT, EU_WHEAT = (24,), (211,)
EU_OTHER = (212, 290)                         # other arable classes, both ends included
EU_MAP = {2018: "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EUCROPMAP/2018/EUCROPMAP_2018.tif",
          2022: "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EUCROPMAP/2022/EU_CropMap_22_v1_stratum_EU27-HR.tif"}
EU_YEARS = (2018, 2022)
# Candidate boxes as (lon0, lon1, lat0, lat1), and a spacing that keeps the scan to about 200 windows
EU_BOXES = {"N France": (0.5, 4.5, 48.0, 50.5), "C Germany": (9.0, 13.0, 51.0, 53.0),
            "W Poland": (16.0, 20.0, 50.5, 53.0), "Hungary": (18.5, 21.5, 46.0, 48.0)}
SPACING, PER_REGION = 0.3, 3
US_TILES = sorted({r["tile"] for r in ca.SEL if r["system"] == "winter_wheat"})
SEL = RESULTS_DIR / "wheat_selection.json"
CKPT = CKPT_DIR / "wheat_invariance" / "sae_us_wheat.pt"
TABLES = RESULTS_DIR / "wheat_invariance_tables.jsonl"
FAMILY = RESULTS_DIR / "wheat_invariance_family.json"
NPZ = RESULTS_DIR / "wheat_invariance.npz"
OUT = RESULTS_DIR / "wheat_invariance.json"
N_PER, N_LOC = 2000, 150

# ---- European labels ------------------------------------------------------------
def eu_label(tile, year, grid=None):
    """Return JRC crop-map classes on the tile's stride-S grid, reading only the tile's window over HTTP."""
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    dst_t, dst_crs, H, W = grid or tile_grid(tile)
    lon, lat = (float(v) for v in tile.split("_")[1:])
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4"):
        with rasterio.open("/vsicurl/" + EU_MAP[year]) as src:
            wb = transform_bounds("EPSG:4326", src.crs, lon - 0.06, lat - 0.06, lon + 0.06, lat + 0.06)
            win = src.window(*wb)
            a = src.read(1, window=win)
            wt = src.window_transform(win)
            crs = src.crs
    d = np.zeros((H, W), np.float32)
    reproject(a.astype(np.float32), d, src_transform=wt, src_crs=crs, dst_transform=dst_t,
              dst_crs=dst_crs, resampling=Resampling.nearest)
    return d[::S, ::S].astype(np.int32)

def groups(lab, side):
    if side == "US":
        wheat = np.isin(lab, US_WHEAT)
        other = (lab >= 1) & (lab <= ca.CROPLAND_MAX) & ~wheat
    else:
        wheat = np.isin(lab, EU_WHEAT)
        other = (lab >= EU_OTHER[0]) & (lab <= EU_OTHER[1]) & ~wheat
    return wheat, other

# ---- select ----------------------------------------------------------------------
def _score(args):
    region, lon, lat = args
    tile = f"grid_{lon:.2f}_{lat:.2f}"
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open("/vsicurl/" + EU_MAP[2018]) as src:
                wb = transform_bounds("EPSG:4326", src.crs, lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
                a = src.read(1, window=src.window(*wb))
        return dict(region=region, tile=tile, wheat=float(np.isin(a, EU_WHEAT).mean()),
                    arable=float(((a >= 211) & (a <= 290)).mean()))
    except Exception as e:
        return dict(region=region, tile=tile, wheat=-1.0, error=repr(e))

def phase_select():
    jobs = []
    for region, (x0, x1, y0, y1) in EU_BOXES.items():
        for lon in np.arange(x0, x1, SPACING):
            for lat in np.arange(y0, y1, SPACING):
                jobs.append((region, round(np.floor(lon * 10) / 10 + 0.05, 2),
                             round(np.floor(lat * 10) / 10 + 0.05, 2)))
    print(f"scoring {len(jobs)} candidate cells", flush=True)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(_score, jobs))
    sel = []
    for region in EU_BOXES:
        rs = sorted((r for r in res if r["region"] == region and r["wheat"] >= 0),
                    key=lambda r: -r["wheat"])
        sel += rs[:PER_REGION + 2]                    # two spares per region, for failed fetches
        print(region, [(r["tile"], round(r["wheat"], 2)) for r in rs[:PER_REGION]])
    json.dump(sel, open(SEL, "w"), indent=1)

def eu_tiles():
    return [s["tile"] for s in json.load(open(SEL)) if s.get("fetched")]

def eu_region():
    return {s["tile"]: s["region"] for s in json.load(open(SEL))}

# ---- fetch -------------------------------------------------------------------------
def _fetch_one(tile):
    try:
        download_landmask(tile)
        for y in EU_YEARS:
            download_tile(tile, y)
        return True, ""
    except Exception as e:
        return False, repr(e)

def phase_fetch():
    sel = json.load(open(SEL))
    for region in EU_BOXES:
        got = 0
        for s in (s for s in sel if s["region"] == region):
            if got >= PER_REGION:
                break
            ok, err = _fetch_one(s["tile"])
            s["fetched"] = ok
            got += ok
            if not ok:
                print("  missing", s["tile"], err[:80])
        print(region, got, "tiles", flush=True)
    json.dump(sel, open(SEL, "w"), indent=1)

# ---- train / screen ------------------------------------------------------------------
def phase_train():
    if CKPT.exists():
        print("cached"); return
    X = np.concatenate([sample_pixels([(t, y) for y in ca.YEARS], n_per=5000, seed=0)
                        for t in US_TILES])
    np.random.default_rng(0).shuffle(X)
    print(f"US wheat corpus {X.shape} from {len(US_TILES)} tiles", flush=True)
    train_and_save(X, CKPT, epochs=40, seed=0, log=True)

def labels(tile, side):
    """Return the classes on the stride-S grid, keyed by year, for each year that side is read in."""
    if side == "US":
        return ca.cdl_stack(tile)
    grid = tile_grid(tile)
    return {y: eu_label(tile, y, grid) for y in EU_YEARS}

def run_tables(args):
    tile, side = args
    torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae, lab, T = load_sae(path=CKPT), labels(tile, side), {}
        for y, L in lab.items():
            x = load_tile(tile, y, stride=S)
            H, W, _ = x.shape
            wheat, other = groups(L[:H, :W], side)
            f = encode_array(sae, x)
            T[str(y)] = {k: f[m].mean(0).tolist() for k, m in
                         (("wheat", wheat), ("_other_crop", other)) if m.sum() >= 400}
        return dict(ok=True, tile=tile, side=side, tables=T)
    except Exception as e:
        return dict(ok=False, tile=tile, side=side, error=repr(e))

def phase_screen():
    jobs = [(t, "US") for t in US_TILES] + [(t, "EU") for t in eu_tiles()]
    with open(TABLES, "w") as f, ProcessPoolExecutor(max_workers=workers(12)) as ex:
        for r in ex.map(run_tables, jobs):
            f.write(json.dumps(r) + "\n")
            print(r["side"], r["tile"], r["ok"], r.get("error", "")[:100], flush=True)
    rows = [r for r in map(json.loads, open(TABLES)) if r["ok"] and r["side"] == "US"]
    fam = ca.blind_family(rows, "wheat")
    json.dump(dict(family=fam, screened_on=[r["tile"] for r in rows], years=ca.SCREEN_YEARS),
              open(FAMILY, "w"), indent=1)
    print(f"wheat family, screened on {len(rows)} US tiles: {len(fam)} features {fam}")

# ---- sample / analyze --------------------------------------------------------------------
def run_sample(args):
    tile, side = args
    torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    fam = json.load(open(FAMILY))["family"]
    sae, rng, out = load_sae(path=CKPT), np.random.default_rng(0), []
    lab = labels(tile, side)
    years = ca.HELD if side == "US" else EU_YEARS
    for y in years:
        x = load_tile(tile, y, stride=S)
        H, W, _ = x.shape
        X = x.reshape(-1, x.shape[-1])
        for cls, m in zip(("wheat", "other"), groups(lab[y][:H, :W], side)):
            idx = np.flatnonzero(m)
            if len(idx) < 200:
                continue
            idx = rng.choice(idx, min(N_PER, len(idx)), replace=False)
            out.append(dict(side=side, tile=tile, year=y, cls=cls, x=X[idx],
                            f=encode_array(sae, X[idx])[:, fam]))
    return out

def phase_sample():
    rows = []
    with ProcessPoolExecutor(max_workers=workers(12)) as ex:
        for r in ex.map(run_sample, [(t, "US") for t in US_TILES] + [(t, "EU") for t in eu_tiles()]):
            rows += r
    rep = lambda k: np.concatenate([[r[k]] * len(r["x"]) for r in rows])
    np.savez_compressed(NPZ, x=np.concatenate([r["x"] for r in rows]).astype(np.float32),
                        f=np.concatenate([r["f"] for r in rows]).astype(np.float32),
                        side=rep("side"), tile=rep("tile"), year=rep("year"), cls=rep("cls"))
    print(f"saved {sum(len(r['x']) for r in rows)} pixels")
    analyze()

def _auc(s, pos, neg):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[s[pos], s[neg]]))

def analyze():
    z = np.load(NPZ)
    x, f, side, tile, year, cls = z["x"], z["f"], z["side"], z["tile"], z["year"], z["cls"]
    s = f.sum(1); on = s > 0
    reg = eu_region()
    rng = np.random.default_rng(0)
    W, O = cls == "wheat", cls == "other"
    out = dict(family_size=int(f.shape[1]),
               n={f"{g}_{c}": int(((side == g) & (cls == c)).sum()) for g in ("US", "EU") for c in ("wheat", "other")},
               n_tiles={g: int(len(np.unique(tile[side == g]))) for g in ("US", "EU")})
    print(f"family: {f.shape[1]} features;  pixels: {out['n']};  tiles {out['n_tiles']}")
    out["family_fires"], out["auc_wheat_vs_other"] = {}, {}
    for g in ("US", "EU"):
        out["family_fires"][f"{g}_wheat"] = float(on[(side == g) & W].mean())
        out["family_fires"][f"{g}_other"] = float(on[(side == g) & O].mean())
        out["auc_wheat_vs_other"][g] = _auc(s, (side == g) & W, (side == g) & O)
    for y in EU_YEARS:
        e = (side == "EU") & (year == y)
        out["auc_wheat_vs_other"][f"EU_{y}"] = _auc(s, e & W, e & O)
    eu_reg = np.array([reg.get(t, "") for t in tile])
    for r_ in EU_BOXES:
        e = (side == "EU") & (eu_reg == r_)
        if (e & W).sum() > 100 and (e & O).sum() > 100:
            out["auc_wheat_vs_other"][f"EU {r_}"] = _auc(s, e & W, e & O)
    print("family fires on:", "  ".join(f"{k} {v:.3f}" for k, v in out["family_fires"].items()))
    print("AUC wheat vs other cropland:", "  ".join(f"{k} {v:.3f}" for k, v in out["auc_wheat_vs_other"].items()))

    m = W & on
    idx = np.concatenate([rng.choice(ii, N_LOC, replace=False) for t in np.unique(tile[m])
                          if len(ii := np.flatnonzero(m & (tile == t))) >= N_LOC])
    out["location_r2_raw"] = dict(r2=group_r2(x[idx], tile[idx]), n_loc=int(len(np.unique(tile[idx]))),
                                  n_eu=int(len(np.unique(tile[idx][side[idx] == "EU"]))))
    out["side_r2_raw"] = group_r2(x[idx], side[idx])
    print(f"family-active wheat px, {N_LOC}/tile, {out['location_r2_raw']['n_loc']} tiles "
          f"({out['location_r2_raw']['n_eu']} EU): location R^2 raw {out['location_r2_raw']['r2']:.3f}; "
          f"side (US vs EU) R^2 {out['side_r2_raw']:.3f}")

    tv = lambda X: float(((X - X.mean(0)) ** 2).sum(1).mean())
    G = {f"{g}_{c}": x[(side == g) & (cls == c)] for g in ("US", "EU") for c in ("wheat", "other")}
    out["raw_centroid_dist"] = {}
    print(f"raw centroid distances  {'abs':>6s} {'norm':>6s}")
    for a, b in (("US_wheat", "EU_wheat"), ("EU_wheat", "EU_other"), ("US_wheat", "US_other"),
                 ("US_other", "EU_other")):
        dd = float(np.linalg.norm(G[a].mean(0) - G[b].mean(0)))
        nd = dd / np.sqrt((tv(G[a]) + tv(G[b])) / 2)
        out["raw_centroid_dist"][f"{a}__{b}"] = dict(abs=dd, norm=float(nd))
        print(f"  {a + ' - ' + b:22s} {dd:6.2f} {nd:6.2f}")
    json.dump(out, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    {"select": phase_select, "fetch": phase_fetch, "train": phase_train,
     "screen": phase_screen, "sample": phase_sample, "analyze": analyze}[sys.argv[1]]()
