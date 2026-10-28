"""Africa and East Asia forest loss: find one family per region and test it on held-out tile-years. Detection-dated.

Reference product: Hansen Global Forest Change (GFC-2023-v1.11). Forest is tree cover of at least 60% in 2000. Tiles
come from the 2.5 degree blocks where Tessera has 2017 and 2022. They are ranked by 2018-2023 forest loss, with at most
4 per block and 80 per region. Units are tile-years, Y = 2018-2023, with at least 25 ha of loss and Tessera for Y-1, Y
and Y+1. The expected window is Y or Y+1. Pixel groups use every second pixel, up to 5,000 per tile-year:
  disturbed    forest lost in Y, at least 300 pixels
  undisturbed  forest on the same tile that was never lost, at least 2,000 pixels
Each region gets one dictionary, trained on 5,000 pixels from each tile-year. The family is screened on the region's
2018 units and tested on its 2019-2023 units.

Usage: python experiments/forest_loss.py [africa|eastasia] select|fetch|train|screen|verdicts   (default: africa)
Writes results/forest_loss_<region>_selection.json, forest_loss_<region>.jsonl and forest_loss_<region>.json, and the
region's dictionary checkpoint.
"""
import json
import math
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

from lib.config import CKPT_DIR, DATA_DIR, GT_DIR, RESULTS_DIR, HANSEN_VERSION
from lib.groundtruth import crop_to_grid
from lib.remote import download
from lib.tiles import available_years, block_of, coverage_blocks, granules, tile_for_point
from lib.sae import load_sae, train_and_save
from lib.sampling import corpus, mean_activation, sample_tile_years
from lib.screen import aggregate_contrasts, anchored_family, peak_outcome, row_contrasts, row_units, outcome_summary

YEARS = list(range(2017, 2026))
LOSS_YEARS = list(range(2018, 2024))
S, WINDOW = 2, (0, 1)
CAP, N_TRAIN = 5000, 5000
MIN_HA, PER_BLOCK, N_TILES, SCREEN_YEAR, FOREST = 25.0, 4, 80, 2018, 60
HANSEN = "https://storage.googleapis.com/earthenginepartners-hansen/{v}/Hansen_{v}_{layer}_{granule}.tif"
REGION = sys.argv[1] if len(sys.argv) > 2 else "africa"         # the phase is always the last argument
NAME = {"africa": "Africa forest loss", "eastasia": "East Asia forest loss"}[REGION]
GL = GT_DIR / "forest_loss" / REGION
PIX = DATA_DIR / "intermediate" / "forest_loss" / REGION
CKPT = CKPT_DIR / "forest_loss" / f"sae_{REGION}.pt"
SEL_PATH = RESULTS_DIR / f"forest_loss_{REGION}_selection.json"
JSONL = RESULTS_DIR / f"forest_loss_{REGION}.jsonl"
SEL = json.load(open(SEL_PATH)) if SEL_PATH.exists() else []


def _in_region(lon, lat):
    if REGION == "africa":           # Africa and Madagascar, not Arabia or the Levant
        return -20 <= lon <= 52 and -36 <= lat <= 33 and not (lon > 32 and lat > 12)
    return 97 <= lon <= 134 and 18 <= lat <= 54


def _granule(left, top):
    return f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}"


def _npz(rec):
    return PIX / f"{rec['tile']}.npz"


def phase_select():
    """Measure each tile's forest loss by year from the Hansen granules, pick tiles and keep their 30 m loss crops."""
    import rasterio
    blocks = [b for b in coverage_blocks(GT_DIR / "tessera_coverage" / "blocks_2017_2022.json")
              if _in_region(b[0] + 1.25, b[1] + 1.25)]
    covered = set(blocks)
    (GL / "tiles").mkdir(parents=True, exist_ok=True)
    stats = {}
    with tempfile.TemporaryDirectory() as tmp:
        for left, top in granules(blocks):
            arr = {}
            for layer in ("lossyear", "treecover2000"):
                f = Path(tmp) / f"{layer}_{_granule(left, top)}.tif"
                try:
                    download(HANSEN.format(v=HANSEN_VERSION, layer=layer, granule=_granule(left, top)), f)
                except Exception:
                    break
                with rasterio.open(f) as s:
                    arr[layer] = s.read(1)
                f.unlink()
            if len(arr) < 2:
                continue
            forest = arr["treecover2000"] >= FOREST
            L = np.where(forest, arr["lossyear"], 255).astype(np.uint8)        # 255 means not forest in 2000
            del arr, forest
            n = 0
            for i in range(100):
                for j in range(100):
                    lon, lat = left + 0.1 * j + 0.05, top - 0.1 * i - 0.05
                    if block_of(lon, lat) not in covered:
                        continue
                    c = L[400 * i:400 * (i + 1), 400 * j:400 * (j + 1)]
                    px_ha = (0.00025 * 111320) ** 2 * math.cos(math.radians(lat)) / 1e4
                    ha = {Y: float((c == Y - 2000).sum() * px_ha) for Y in LOSS_YEARS}
                    if max(ha.values()) < MIN_HA:
                        continue
                    t = tile_for_point(lon, lat)
                    stats[t] = ha
                    i0, j0 = max(400 * i - 40, 0), max(400 * j - 40, 0)        # a 0.01 degree margin on each side
                    np.savez_compressed(GL / "tiles" / f"{t}.npz", loss=L[i0:400 * (i + 1) + 40, j0:400 * (j + 1) + 40],
                                        transform=np.array([0.00025, 0, left + j0 * 0.00025, 0, -0.00025, top - i0 * 0.00025]))
                    n += 1
            print(f"  granule {_granule(left, top)}: {n} candidate tiles", flush=True)
    ranked = sorted(stats, key=lambda t: -sum(stats[t].values()))
    per, keep = {}, []
    for t in ranked:
        b = block_of(*(float(v) for v in t.replace("grid_", "").split("_")))
        if per.get(b, 0) < PER_BLOCK:
            keep.append(t); per[b] = per.get(b, 0) + 1
    have = available_years(keep[:2 * N_TILES], YEARS)
    out = []
    for t in keep[:2 * N_TILES]:
        years = [Y for Y in LOSS_YEARS if stats[t][Y] >= MIN_HA and {Y - 1, Y, Y + 1} <= have[t]]
        if years:                                   # the tile's loss years that become units
            out.append(dict(tile=t, loss_years=years, ha={str(Y): round(stats[t][Y], 1) for Y in years},
                            years=sorted(have[t])))
        if len(out) == N_TILES:
            break
    json.dump(out, open(SEL_PATH, "w"), indent=1)
    keepset = {r["tile"] for r in out}
    for f in (GL / "tiles").glob("*.npz"):
        if f.stem not in keepset:
            f.unlink()
    units = sum(len(r["loss_years"]) for r in out)
    print(f"{NAME}: {len(stats)} candidate tiles in {len(blocks)} covered blocks; kept {len(out)} tiles, {units} units "
          f"({sum(SCREEN_YEAR in r['loss_years'] for r in out)} screening)")


def _masks(rec):
    """Return stride-S masks on the tile grid: lost forest for each loss year, and never-lost forest under 'ref'."""
    z = np.load(GL / "tiles" / f"{rec['tile']}.npz")
    d = crop_to_grid(z["loss"], z["transform"], rec["tile"], 254)[::S, ::S]
    out = {Y: d == Y - 2000 for Y in rec["loss_years"]}
    out["ref"] = d == 0
    return out


def fetch_one(rec):
    out = _npz(rec)
    if out.exists():
        return rec["tile"], "cached"
    try:
        rng = np.random.default_rng(zlib.crc32(rec["tile"].encode()))
        store = sample_tile_years(rec["tile"], rec["years"], _masks(rec), S, CAP, N_TRAIN, rng)
        if not store:
            return rec["tile"], "no years"
        PIX.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **store)
        return rec["tile"], ""
    except Exception as e:
        return rec["tile"], repr(e)


def phase_fetch():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch_one, SEL))
    bad = [(t, e) for t, e in res if e and e != "cached"]
    print(f"{len(res) - len(bad)}/{len(res)} tiles sampled; failed: {bad[:8]}", flush=True)


def phase_train():
    if CKPT.exists():
        print("cached"); return
    X = corpus([_npz(r) for r in SEL])
    print(f"{NAME} corpus {X.shape}", flush=True)
    train_and_save(X, CKPT)


def phase_screen():
    import torch
    sae = load_sae(path=CKPT)
    torch.set_num_threads(16)
    with open(JSONL, "w") as out:
        for r in SEL:
            if not _npz(r).exists():
                continue
            z = np.load(_npz(r))
            ref = {}
            for y in YEARS:
                if f"{y}_ref" in z.files:
                    b, _ = mean_activation(sae, z[f"{y}_ref"], 2000)
                    if b is not None:
                        ref[str(y)] = b
            for Y in r["loss_years"]:
                dmg, n = {}, {}
                for y in YEARS:
                    if f"{y}_{Y}" in z.files and str(y) in ref:
                        a, na = mean_activation(sae, z[f"{y}_{Y}"], 300)
                        if a is not None:
                            dmg[str(y)], n[str(y)] = a.tolist(), na
                ok = {str(Y), str(Y + 1)} <= set(dmg)
                out.write(json.dumps(dict(ok=ok, uid=f"{r['tile']}:{Y}", tile=r["tile"], year=Y,
                                          screen=Y == SCREEN_YEAR, n=n, dmg=dmg,
                                          ref={y: ref[y].tolist() for y in dmg})) + "\n")
    verdicts()


def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    scr, held = [r for r in rows if r["screen"]], [r for r in rows if not r["screen"]]
    fam = anchored_family(aggregate_contrasts(row_units(scr)), window=WINDOW)
    v = peak_outcome(row_contrasts(held), fam, WINDOW) if fam else {}
    yr = {r["uid"]: r["year"] for r in held}
    summ = outcome_summary(v, yr)
    print(f"{NAME}: {len(scr)} screening units ({SCREEN_YEAR}), {len(held)} held out; family {len(fam)} features {fam}")
    for k, s in summ.items():
        print(f"  {k}: dated to {{Y, Y+1}} {s['rate']:.3f} (n={s['n']}; Y share among dated {s['y0_share']:.2f})")
    json.dump(dict(family=fam, n_screen=len(scr), n_held=len(v), rate=summ["all"]["rate"], summary=summ,
                   outcomes={u: dict(peak=pk, ok=ok, tile=u.split(":")[0], year=yr[u]) for u, (pk, ok) in v.items()}),
              open(RESULTS_DIR / f"forest_loss_{REGION}.json", "w"), indent=1, default=float)
    print(f"saved results/forest_loss_{REGION}.json")


if __name__ == "__main__":
    {"select": phase_select, "fetch": phase_fetch, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[-1]]()
