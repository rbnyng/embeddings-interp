"""Build the new permanent water labels from the JRC Global Surface Water yearly history v1.4, in Earth Engine.

Runs in its own environment with earthengine-api, and needs an Earth Engine project in $EE_PROJECT. A pixel has onset
year Y (2018-2021) when it is land in each of the three years before Y, water (seasonal or permanent) in Y and every
year after up to 2021, and permanent water in 2021.
  candidates  0.1 degree tiles in the blocks where Tessera has 2017 and 2022, with at least 25 ha of new permanent
              water (JRC transitions) that was water in at most 30% of 1984-2021 observations
  onsets      hectares of new water by onset year for each candidate tile, largest tile first. It can be resumed.
  tiles       the onset year (0 = none) and a mask of never water in 2015-2021, at 30 m, for each tile in
              results/new_water_selection.json

Usage: python -I experiments/new_water_labels.py candidates|onsets|tiles
Writes candidates.json, onsets_partial.json, survey.npz and tiles/<tile>.npz in data/groundtruth/new_water/.
"""
import io
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                     # run with -I, so add lib to the path by hand
from lib.config import EE_PROJECT, GT_DIR, RESULTS_DIR   # noqa: E402
from lib.remote import download                          # noqa: E402
from lib.tiles import block_of, coverage_blocks, granules, tile_for_point   # noqa: E402

OUT = GT_DIR / "new_water"
BLOCKS = GT_DIR / "tessera_coverage" / "blocks_2017_2022.json"
JRC = "https://storage.googleapis.com/global-surface-water/downloads2021/{layer}/{layer}_{tag}"
ONSETS = (2018, 2019, 2020, 2021)
EE_WORKERS = 2                                    # the most requests the project can run at once
_EE = {}


def _ee():
    """Return Earth Engine and the yearly water classes for 2015-2021, set up on first use.

    Classes: 0 no data or never water, 1 land, 2 seasonal water, 3 permanent water.
    """
    if not _EE:
        import ee
        if not EE_PROJECT:
            raise RuntimeError("set EE_PROJECT to an Earth Engine project")
        ee.Initialize(project=EE_PROJECT)
        hist = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
        _EE.update(ee=ee, cls={y: hist.filter(ee.Filter.eq("year", y)).first().select("waterClass").unmask(0)
                               for y in range(2015, 2022)})
    return _EE["ee"], _EE["cls"]


def onset(Y):
    ee, cls = _ee()
    land = cls[Y - 3].eq(1).And(cls[Y - 2].eq(1)).And(cls[Y - 1].eq(1))
    water = ee.Image(1)
    for y in range(Y, 2022):
        water = water.And(cls[y].gte(2))
    return land.And(water).And(cls[2021].eq(3))


def _npy(img, region, transform):
    for _ in range(3):
        try:
            r = requests.get(img.getDownloadURL(dict(region=region, format="NPY", crs="EPSG:4326", crs_transform=transform)),
                             timeout=600)
            if r.ok:
                return np.load(io.BytesIO(r.content), allow_pickle=False)
        except Exception:
            pass
    return None


def candidates():
    import rasterio
    blocks = set(coverage_blocks(BLOCKS))
    cand = {}
    with tempfile.TemporaryDirectory() as tmp:
        for left, top in granules(blocks):
            tag = f"{abs(left)}{'E' if left >= 0 else 'W'}_{abs(top)}{'N' if top >= 0 else 'S'}v1_4_2021.tif"
            arr = {}
            for layer in ("transitions", "occurrence"):
                f = Path(tmp) / f"{layer}_{tag}"
                try:
                    download(JRC.format(layer=layer, tag=tag), f)
                except Exception:
                    break
                with rasterio.open(f) as s:
                    arr[layer] = s.read(1)
                f.unlink()
            if len(arr) < 2:
                continue
            recent = (arr["transitions"] == 2) & (arr["occurrence"] <= 30)
            del arr
            n = recent.reshape(100, 400, 100, 400).sum((1, 3))             # recent new permanent water per tile
            for i, j in zip(*np.nonzero(n)):
                lon, lat = left + 0.1 * j + 0.05, top - 0.1 * i - 0.05
                ha = n[i, j] * (0.00025 * 111320) ** 2 * math.cos(math.radians(lat)) / 1e4
                if ha >= 25 and block_of(lon, lat) in blocks:
                    cand[tile_for_point(lon, lat)] = round(float(ha), 1)
            print(f"  {tag}: {len(cand)} candidate tiles so far", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(cand, open(OUT / "candidates.json", "w"))
    print(f"{len(cand)} tiles with >= 25 ha of recent new permanent water")


def onsets():
    ee, _ = _ee()
    cand = json.load(open(OUT / "candidates.json"))
    img = ee.Image.cat([onset(Y).multiply(ee.Image.pixelArea()).divide(1e4).rename(f"y{Y}") for Y in ONSETS])
    part = OUT / "onsets_partial.json"
    done = json.load(open(part)) if part.exists() else {}

    def one(tile):
        lon, lat = (float(v) for v in tile.replace("grid_", "").split("_"))
        g = ee.Geometry.Rectangle([lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05], "EPSG:4326", False)
        for k in range(6):
            try:
                return tile, img.reduceRegion(ee.Reducer.sum(), g, scale=30, maxPixels=1e9).getInfo()
            except Exception as e:
                if "Too Many Requests" not in str(e) and k >= 2:
                    return tile, None
                time.sleep(5 * (k + 1))
        return tile, None
    todo = sorted([t for t in cand if t not in done], key=lambda t: -cand[t])
    with ThreadPoolExecutor(EE_WORKERS) as ex:
        for i, (tile, v) in enumerate(ex.map(one, todo), 1):
            if v is not None:
                done[tile] = v
            if i % 100 == 0:
                _write(done)
                print(f"  [{i}/{len(todo)}] {time.strftime('%H:%M')}", flush=True)
    _write(done)
    print(f"{len(done)}/{len(cand)} candidate tiles dated")


def _write(done):
    """Write onsets_partial.json and survey.npz from the tiles dated so far."""
    json.dump(done, open(OUT / "onsets_partial.json", "w"))
    rows = [(Y, t, v[f"y{Y}"]) for t, v in done.items() for Y in ONSETS if v.get(f"y{Y}")]
    np.savez(OUT / "survey.npz", year=np.array([r[0] for r in rows]), tile=np.array([r[1] for r in rows]),
             ha=np.array([r[2] for r in rows]))
    for Y in ONSETS:
        v = np.array([r[2] for r in rows if r[0] == Y])
        print(f"    onset {Y}: tiles >= 25 ha {(v >= 25).sum()}, >= 50 ha {(v >= 50).sum()}", flush=True)


def tiles():
    ee, cls = _ee()
    sel = json.load(open(RESULTS_DIR / "new_water_selection.json"))
    img = ee.Image.cat([ee.Image(0).where(onset(Y), Y).rename(f"o{Y}") for Y in ONSETS]).reduce(ee.Reducer.max()).toInt16() \
        .rename("onset").addBands(ee.ImageCollection([cls[y].gte(2) for y in range(2015, 2022)]).max().Not()
                                  .rename("never").toInt16())        # pixels masked in the history count as never water
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)

    def one(r):
        f = OUT / "tiles" / f"{r['tile']}.npz"
        if f.exists():
            return r["tile"], "cached"
        lon, lat = (float(v) for v in r["tile"].replace("grid_", "").split("_"))
        x0, y1 = lon - 0.06, lat + 0.06
        T = [0.00025, 0, x0, 0, -0.00025, y1]
        a = _npy(img, ee.Geometry.Rectangle([x0, lat - 0.06, lon + 0.06, y1], "EPSG:4326", False), T)
        if a is None:
            return r["tile"], "failed"
        np.savez_compressed(f, onset=a["onset"], never=a["never"].astype(bool), transform=np.array(T))
        return r["tile"], ""
    with ThreadPoolExecutor(EE_WORKERS) as ex:
        res = list(ex.map(one, sel))
    print(f"{sum(e in ('', 'cached') for _, e in res)}/{len(res)} tiles labelled; failed {[t for t, e in res if e == 'failed'][:10]}")


if __name__ == "__main__":
    {"candidates": candidates, "onsets": onsets, "tiles": tiles}[sys.argv[1]]()
