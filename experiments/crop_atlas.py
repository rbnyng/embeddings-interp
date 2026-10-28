"""Crop atlas: find crop-class families in one Tessera farmland dictionary and classify held-out pixels with them.

Tiles are picked from USDA CDL class density alone (see phase_select). The systems are corn/soy, winter wheat (KS/OK),
rice (Arkansas and Sacramento Valley), cotton (Texas High Plains), Central Valley perennials and fallow. The dictionary
is trained on 5,000 pixels per tile-year, 2017-2025. A class family is screened with the anchored bar on the contrast
between the class and other cropland (CDL codes 1-60), 2018-2021. Groups need at least 400 pixels at stride 2. In
2022-2024 each pixel goes to the family with the highest mean activation, with all families competing. crop_atlas.json
holds the families and these held-out results:
  c2     corn and soy accuracy, and the mean of the two
  c3_*   winter wheat, rice and cotton accuracy
  c4     rate of year-to-year prediction changes on stable perennial pixels and on rotating corn/soy pixels
  tiles  each tile's accuracy over the classes of its own systems (a tile passes at TILE_PASS = 0.70 or above)
`separation` compares raw Tessera and AlphaEarth on the corn/soy tiles: the distance between the corn and soy centers
in held-out years, divided by the spread within each class.

Usage: python experiments/crop_atlas.py select|train|screen|verdicts|separation
Writes results/crop_selection.json, crop_atlas.jsonl/.json, crop_aef.json and data/checkpoints/crop_atlas/sae_ag.pt.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from lib.alphaearth import compute_scale, load_aef
from lib.config import CKPT_DIR, GT_DIR, RESULTS_DIR, workers
from lib.tiles import load_tile, tile_grid, sample_pixels
from lib.sae import encode_array, group_sums, load_sae, train_and_save
from lib.screen import anchored_bar_family

CDL_DIR = GT_DIR / "cdl"
YEARS = list(range(2017, 2026))
CDL_YEARS = list(range(2018, 2025))
SCREEN_YEARS = [2018, 2019, 2020, 2021]
HELD = [2022, 2023, 2024]
S = 2
CKPT = CKPT_DIR / "crop_atlas" / "sae_ag.pt"
JSONL = RESULTS_DIR / "crop_atlas.jsonl"
_SEL_PATH = RESULTS_DIR / "crop_selection.json"
# Empty until `select` has run, so that phase still works on a fresh checkout.
SEL = json.load(open(_SEL_PATH)) if _SEL_PATH.exists() else []
TILE_SYS = defaultdict(set)
for r in SEL:
    TILE_SYS[r["tile"]].add(r["system"])
TILES = sorted(TILE_SYS)
# A tile passes when its held-out accuracy is at least TILE_PASS. Accuracy is pooled over pixels of the classes of the
# tile's own systems, with all families competing.
TILE_PASS = 0.70
SYS_CLASSES = dict(cornsoy=("corn", "soy"), winter_wheat=("winter_wheat",), rice=("rice",),
                   cotton=("cotton",), perennial=("perennial",), fallow=("fallow",))

CLASSES = dict(corn=(1,), soy=(5,), winter_wheat=(24,), rice=(3,), cotton=(2,),
               fallow=(61,), perennial=(69, 75, 76, 204))
CROPLAND_MAX = 60          # CDL codes 1 to 60 are crops. Code 61 (fallow) is kept separate.

def phase_train():
    if CKPT.exists():
        print("cached"); return
    rng = np.random.default_rng(0)
    X = np.concatenate([sample_pixels([(t, y) for y in YEARS], n_per=5000, seed=0)
                        for t in TILES])
    rng.shuffle(X)
    print(f"ag corpus {X.shape}", flush=True)
    train_and_save(X, CKPT, epochs=40, seed=0, log=True)
    print("trained", flush=True)

def cdl_stack(tile, years=None):
    """Return CDL class codes for each year (CDL_YEARS unless given), on the tile grid at stride S."""
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    lon, lat = (float(v) for v in tile.split("_")[1:])
    bb = (lon - 0.055, lat - 0.055, lon + 0.055, lat + 0.055)
    dst_t, dst_crs, H, W = tile_grid(tile)
    out = {}
    for y in (CDL_YEARS if years is None else years):
        with rasterio.open(CDL_DIR / f"{y}_30m_cdls.tif") as src:
            wb = transform_bounds("EPSG:4326", src.crs, *bb)
            win = src.window(*wb)
            a = src.read(1, window=win)
            wt = src.window_transform(win)
        d = np.zeros((H, W))
        reproject(a.astype(np.float32), d, src_transform=wt, src_crs=src.crs,
                  dst_transform=dst_t, dst_crs=dst_crs,
                  resampling=Resampling.nearest)
        out[y] = d[::S, ::S].astype(np.int16)
    return out

def run_tile(tile):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        cdl = cdl_stack(tile)
        sae = load_sae(path=CKPT)
        # For each year, the mean activation of every feature over each class present, and over the rest of the
        # cropland.
        tables = {}
        for y in CDL_YEARS:
            try:
                x = load_tile(tile, y, stride=S)
            except Exception:
                continue
            H, W, _ = x.shape
            lab = cdl[y][:H, :W]
            groups = {}
            for name, codes in CLASSES.items():
                mm = np.isin(lab, codes)
                if mm.sum() >= 400:
                    groups[name] = mm
            crop = (lab >= 1) & (lab <= CROPLAND_MAX)
            oth = crop.copy()
            for g in groups.values():
                oth &= ~g
            if oth.sum() >= 400:
                groups["_other_crop"] = oth
            s, n = group_sums(sae, x, groups)
            tables[y] = {k: (s[k] / n[k]).tolist() for k in groups}
            del x
        return dict(ok=True, tile=tile, systems=sorted(TILE_SYS[tile]),
                    tables={str(y): tables[y] for y in tables})
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

def phase_screen():
    from concurrent.futures import ProcessPoolExecutor
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
    with ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for i, r in enumerate(ex.map(run_tile, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    verdicts()

def blind_family(rows, cls):
    """Screen one class's family with the anchored bar. The contrast with other cropland is a median over tiles, then a
    mean over screening years."""
    per = defaultdict(list)
    for r in rows:
        for y in SCREEN_YEARS:
            T = r["tables"].get(str(y), {})
            if cls in T and "_other_crop" in T:
                per[y].append(np.array(T[cls]) - np.array(T["_other_crop"]))
    if not per:
        return []
    A = np.array([np.median(v, axis=0) for v in per.values()])
    return anchored_bar_family(A.mean(0), act=np.abs(A).max(0) > 0)

def score_tile(args):
    """Classify each held-out pixel of one tile, with all families competing.

    Returns [correct, n] per class. Also returns [changed, n] per class: prediction changes from one year to the next,
    on pixels that are that class in both years.
    """
    tile, fams = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        cdl = cdl_stack(tile)
        sae = load_sae(path=CKPT)
        names = sorted(fams)
        res = {n: [0, 0] for n in names}
        flips = {n: [0, 0] for n in names}
        prev = None
        for y in HELD:
            try:
                x = load_tile(tile, y, stride=S)
            except Exception:
                prev = None
                continue
            H, W, _ = x.shape
            lab = cdl[y][:H, :W]
            pred = np.full((H, W), -1, dtype=np.int8)
            for r0 in range(0, H, 192):
                r1 = min(r0 + 192, H)
                f = encode_array(sae, x[r0:r1])
                sc = np.stack([f[:, :, fams[n]].mean(-1) for n in names], -1)
                pred[r0:r1] = np.argmax(sc, -1)
            for ci, n in enumerate(names):
                m = np.isin(lab, CLASSES[n])
                if m.sum() < 200:
                    continue
                res[n][0] += int((pred[m] == ci).sum())
                res[n][1] += int(m.sum())
            if prev is not None:
                pH = min(prev.shape[0], H)
                for ci, n in enumerate(names):
                    m = np.isin(lab[:pH], CLASSES[n]) \
                        & np.isin(cdl[y - 1][:pH, :W], CLASSES[n])
                    if m.sum() < 200:
                        continue
                    flips[n][0] += int((pred[:pH][m] != prev[:pH][m]).sum())
                    flips[n][1] += int(m.sum())
            prev = pred
            del x
        return tile, res, flips
    except Exception as e:
        print(f"  {tile}: scoring failed, {e!r}", flush=True)
        return tile, {}, {}

def rotation_flip(args):
    """Count prediction changes (changed, n) on rotating pixels: corn in one held-out year and soy the next, or the
    reverse."""
    tile, fams = args
    torch.set_num_threads(1)
    try:
        cdl = cdl_stack(tile)
        sae = load_sae(path=CKPT)
        names = sorted(fams)
        changed = tot = 0
        prev = None
        for y in HELD:
            try:
                x = load_tile(tile, y, stride=S)
            except Exception:
                prev = None
                continue
            H, W, _ = x.shape
            pred = np.full((H, W), -1, dtype=np.int8)
            for r0 in range(0, H, 192):
                r1 = min(r0 + 192, H)
                f = encode_array(sae, x[r0:r1])
                sc = np.stack([f[:, :, fams[n]].mean(-1) for n in names], -1)
                pred[r0:r1] = np.argmax(sc, -1)
            if prev is not None:
                pH = min(prev.shape[0], H)
                lab0 = cdl[y - 1][:pH, :W]
                lab1 = cdl[y][:pH, :W]
                rot = ((np.isin(lab0, CLASSES["corn"]) & np.isin(lab1, CLASSES["soy"]))
                       | (np.isin(lab0, CLASSES["soy"]) & np.isin(lab1, CLASSES["corn"])))
                if rot.sum() >= 200:
                    changed += int((pred[:pH][rot] != prev[:pH][rot]).sum())
                    tot += int(rot.sum())
            prev = pred
            del x
        return changed, tot
    except Exception as e:
        print(f"  {tile}: rotation count failed, {e!r}", flush=True)
        return 0, 0

def verdicts():
    from concurrent.futures import ProcessPoolExecutor
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    print(f"\n{len(rows)} tiles screened")
    fams = {}
    for cls in ("corn", "soy", "winter_wheat", "rice", "cotton", "perennial",
                "fallow"):
        fam = blind_family(rows, cls)
        print(f"{cls:<13} family {len(fam):>3} features")
        if fam:
            fams[cls] = fam

    sys_tiles = defaultdict(set)
    for r in rows:
        for s_ in r["systems"]:
            sys_tiles[s_].add(r["tile"])

    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # by system, then class: [correct, n]
    stab = defaultdict(lambda: [0, 0])                       # by class: flips on stable pixels
    tile_sys = {r["tile"]: r["systems"] for r in rows}
    per_tile = {}
    with ProcessPoolExecutor(max_workers=workers(32)) as ex:
        for tile, res, flips in ex.map(score_tile,
                                       [(t, fams) for t in tile_sys]):
            own = [res[c] for s_ in tile_sys[tile] for c in SYS_CLASSES[s_] if c in res]
            c_, n_ = sum(x[0] for x in own), sum(x[1] for x in own)
            per_tile[tile] = dict(systems=sorted(tile_sys[tile]), n=int(n_),
                                  acc=(c_ / n_ if n_ else None),
                                  ok=(bool(c_ / n_ >= TILE_PASS) if n_ else None))
            for cls, (c, n) in res.items():
                for s_ in tile_sys[tile]:
                    agg[s_][cls][0] += c
                    agg[s_][cls][1] += n
            for cls, (c, n) in flips.items():
                stab[cls][0] += c
                stab[cls][1] += n

    def rate(sys_, cls):
        c, n = agg[sys_][cls]
        return c / n if n else float("nan"), n

    out = {}
    a_c, n_c = rate("cornsoy", "corn")
    a_s, n_s = rate("cornsoy", "soy")
    c2 = float(np.mean([a_c, a_s]))
    print(f"\nheld-out accuracy, all families competing: corn {a_c:.3f} (n={n_c:,}) "
          f"soy {a_s:.3f} (n={n_s:,}) mean {c2:.3f}")
    out["c2"] = dict(corn=a_c, soy=a_s, mean=c2)
    for cls in ("winter_wheat", "rice", "cotton"):
        if cls not in fams:
            print(f"{cls}: no family")
            continue
        a, n = rate(cls, cls)
        print(f"{cls} held-out accuracy: {a:.3f} (n={n:,})")
        out[f"c3_{cls}"] = a

    with ProcessPoolExecutor(max_workers=workers(32)) as ex:
        rots = list(ex.map(rotation_flip,
                           [(t, fams) for t in sys_tiles["cornsoy"]]))
    rc = sum(r[0] for r in rots)
    rn = sum(r[1] for r in rots)
    rot_rate = rc / rn if rn else float("nan")
    pf, pn = stab.get("perennial", (0, 0))
    per_rate = pf / pn if pn else float("nan")
    print(f"prediction change rate: perennial (stable pixels) {per_rate:.3f} (n={pn:,}), "
          f"corn/soy (rotating pixels) {rot_rate:.3f} (n={rn:,})")
    out["c4"] = dict(perennial=per_rate, rotation=rot_rate)
    out["families"] = fams
    out["tiles"] = per_tile
    scored = [v["ok"] for v in per_tile.values() if v["ok"] is not None]
    print(f"per-tile outcomes: {sum(scored)}/{len(scored)} tiles pass "
          f"(held-out accuracy >= {TILE_PASS})")
    json.dump(out, open(RESULTS_DIR / "crop_atlas.json", "w"), default=float)
    print("\nsaved results/crop_atlas.json")


# ---- selection

def phase_select():
    """Pick tiles for each crop system from CDL class density alone and write results/crop_selection.json.

    Scans each system's box on the 0.1-degree tile grid. Density is the class share of the tile, averaged over 2018 and
    2020 (both screening years). Each system keeps its 21 densest tiles above a floor, and each rice region 10.
      cornsoy       1 corn and 5 soy, the smaller share at least 0.15
      winter_wheat  24, at least 0.20
      rice          3, at least 0.20 (Arkansas and Sacramento Valley)
      cotton        2, at least 0.20
      perennial     69 grapes, 75 almonds, 76 walnuts, 204 pistachios, at least 0.25
      fallow        61, at least 0.15
    """
    import rasterio
    from rasterio.warp import transform_bounds
    CDL = GT_DIR / "cdl"
    YEARS = (2018, 2020)
    CAP = 21
    SYSTEMS = {
        "cornsoy": dict(box=(-95.5, 40.0, -88.0, 43.5), codes=(1, 5), floor=0.15,
                        both=True),
        "winter_wheat": dict(box=(-101.5, 36.0, -96.5, 39.5), codes=(24,), floor=0.20),
        "rice": dict(box=(-91.8, 33.5, -89.8, 36.3), codes=(3,), floor=0.20),
        "rice_ca": dict(box=(-122.3, 38.6, -121.3, 39.8), codes=(3,), floor=0.20),
        "cotton": dict(box=(-102.5, 32.5, -100.5, 34.5), codes=(2,), floor=0.20),
        "perennial": dict(box=(-121.0, 35.0, -118.8, 37.8), codes=(69, 75, 76, 204),
                          floor=0.25),
        "fallow": dict(box=(-103.0, 37.0, -100.5, 39.5), codes=(61,), floor=0.15),
    }
    from lib.tiles import snap, tile_for_point
    srcs = {y: rasterio.open(CDL / f"{y}_30m_cdls.tif") for y in YEARS}
    sel = []
    for name, cfg in SYSTEMS.items():
        x0, y0, x1, y1 = cfg["box"]
        scores = []
        for lo in np.arange(x0, x1, 0.1):
            for la in np.arange(y0, y1, 0.1):
                bb = (snap(lo) - 0.05, snap(la) - 0.05, snap(lo) + 0.05, snap(la) + 0.05)
                fr = []
                for y, src in srcs.items():
                    try:
                        wb = transform_bounds("EPSG:4326", src.crs, *bb)
                        win = src.window(*wb)
                        if win.width < 4 or win.height < 4:
                            continue
                        a = src.read(1, window=win)
                    except Exception:
                        continue
                    tot = a.size
                    if cfg.get("both"):
                        fr.append(min(float((a == c).sum() / tot) for c in cfg["codes"]))
                    else:
                        fr.append(float(np.isin(a, cfg["codes"]).sum() / tot))
                if len(fr) == len(YEARS):
                    f = float(np.mean(fr))
                    if f >= cfg["floor"]:
                        scores.append((f, tile_for_point(lo, la)))
        scores.sort(reverse=True)
        sysname = "rice" if name == "rice_ca" else name
        kept = scores[: (CAP // 2 if name.startswith("rice") else CAP)]
        for f, t in kept:
            sel.append(dict(tile=t, system=sysname, subregion=name, frac=round(f, 4)))
        print(f"{name:<13} {len(scores):>4} candidates -> kept {len(kept)} "
              f"(density {kept[-1][0]:.2f}-{kept[0][0]:.2f})" if kept
              else f"{name:<13} {len(scores):>4} candidates -> kept 0 !!")

    tiles = sorted({s["tile"] for s in sel})
    json.dump(sel, open(RESULTS_DIR / "crop_selection.json", "w"))
    print(f"\nselected {len(sel)} rows, {len(tiles)} distinct tiles")
    from collections import Counter
    print(Counter(s["system"] for s in sel))


# ---- Tessera against AlphaEarth, without a dictionary

def raw_separation(args):
    """No dictionary: for each model, the corn-soy centroid distance in held-out years divided by its within-class
    spread."""
    tile, scale = args
    torch.set_num_threads(1)
    try:
        cdl = cdl_stack(tile)
        out = {}
        for model in ("Tessera", "AlphaEarth"):
            vals = []
            for y in HELD:
                try:
                    x = (load_tile(tile, y, stride=S) if model == "Tessera"
                         else np.nan_to_num(load_aef(tile, y, stride=S)) * scale)
                except Exception:
                    continue
                x = np.nan_to_num(np.asarray(x, dtype=np.float32))
                H, W = x.shape[:2]
                lab = cdl[y][:H, :W]
                c = np.isin(lab, CLASSES["corn"])
                s = np.isin(lab, CLASSES["soy"])
                if c.sum() < 200 or s.sum() < 200:
                    continue
                xc, xs = x[c], x[s]
                d = np.linalg.norm(xc.mean(0) - xs.mean(0))
                spread = 0.5 * (xc.std(0).mean() + xs.std(0).mean()) * np.sqrt(x.shape[-1])
                vals.append(float(d / max(spread, 1e-9)))
            if vals:
                out[model] = float(np.mean(vals))
        return out
    except Exception as e:
        print(f"  {tile}: separation failed, {e!r}", flush=True)
        return {}

def phase_separation():
    """Raw corn vs soy separation of both models on the corn/soy tiles, stored as k4 in results/crop_aef.json."""
    from concurrent.futures import ProcessPoolExecutor
    scale = compute_scale()
    cornsoy = sorted(t for t, systems in TILE_SYS.items() if "cornsoy" in systems)
    with ProcessPoolExecutor(max_workers=workers(32)) as ex:
        seps = [s for s in ex.map(raw_separation, [(t, scale) for t in cornsoy]) if s]
    rt = float(np.median([s["Tessera"] for s in seps if "Tessera" in s]))
    ra = float(np.median([s["AlphaEarth"] for s in seps if "AlphaEarth" in s]))
    print(f"raw corn vs soy centroid separation (no dictionary, spread-normalized) on {len(seps)} tiles: "
          f"Tessera {rt:.3f}  AlphaEarth {ra:.3f}  ratio {ra/rt if rt else float('nan'):.2f}")
    json.dump(dict(k4=dict(tessera=rt, aef=ra)), open(RESULTS_DIR / "crop_aef.json", "w"), default=float)
    print("\nsaved results/crop_aef.json")

if __name__ == "__main__":
    {"select": phase_select, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts, "separation": phase_separation}[sys.argv[1]]()
