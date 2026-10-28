"""Build the SE Australia fire inventory and download its AlphaEarth tiles.

Reference product: Geoscience Australia (GA) Historic Bushfire Boundaries. The inventory is every Black Summer
wildfire of at least 2,000 ha, ignited July 2019 to March 2020, on the 0.1 degree tile it covers most, where it covers
at least 15% of that tile. fetch downloads AlphaEarth for 2017-2023 on every fire's tile.

Usage: python experiments/se_australia_inventory.py inventory|fetch
Writes results/se_australia_inventory.json, se_australia_geoms.geojson and se_australia_fetch_errors.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lib.alphaearth import fetch_aef
from lib.config import RESULTS_DIR
from lib.tiles import download_landmask, tile_grid

YEARS = list(range(2017, 2024))
S = 2

_INV_PATH = RESULTS_DIR / "se_australia_inventory.json"
# Empty until the inventory phase has run, so that phase also works on a fresh checkout.
FIRES = json.load(open(_INV_PATH)) if _INV_PATH.exists() else []

GEOMS = None

def fire_mask(rec):
    global GEOMS
    from shapely.geometry import shape
    from rasterio import features as rfeat
    if GEOMS is None:
        GEOMS = {f["properties"]["uid"]: f["geometry"]
                 for f in json.load(open(RESULTS_DIR / "se_australia_geoms.geojson"))["features"]}
    g = shape(GEOMS[rec["uid"]])
    if not g.is_valid:
        g = g.buffer(0)
    t, crs, H, W = tile_grid(rec["tile"])
    import geopandas as gpd
    gs = gpd.GeoSeries([g], crs="EPSG:4326").to_crs(crs)
    m = rfeat.rasterize([(gs.iloc[0], 1)], out_shape=(H, W), transform=t,
                        fill=0, dtype=np.uint8).astype(bool)
    return m[::S, ::S]

def fetch_all():
    tiles = sorted({r["tile"] for r in FIRES})
    jobs = [(t, y) for t in tiles for y in YEARS]
    print(f"{len(tiles)} tiles x {len(YEARS)} years = {len(jobs)} AlphaEarth tile-years "
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
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            if r:
                errs.append(r)
            if i % 100 == 0:
                print(f"  [{i}/{len(jobs)}] ({len(errs)} errors)", flush=True)
    print(f"AlphaEarth fetch done ({len(errs)} errors)", flush=True)
    json.dump(errs, open(RESULTS_DIR / "se_australia_fetch_errors.json", "w"))


def build_inventory():
    """Download GA fire outlines and keep the fires that cover enough of their tile. Uses only the reference data."""
    import requests
    from shapely.geometry import shape, box as sbox
    U = ("https://services-ap1.arcgis.com/ypkPEy1AmwPKGNNv/arcgis/rest/services/"
         "Bushfire_Boundaries_Historic_Dec_view/FeatureServer/0/query")

    def harvest(where, min_ha):
        feats, offset = [], 0
        while True:
            r = requests.get(U, params=dict(
                where=f"({where}) AND area_ha >= {min_ha}",
                outFields="OBJECTID,fire_id,fire_name,ignition_date,extinguish_date,fire_type,"
                          "ignition_cause,area_ha,state,capt_method",
                outSR=4326, f="geojson", resultOffset=offset,
                resultRecordCount=2000), timeout=180)
            r.raise_for_status()
            chunk = r.json().get("features", [])
            feats += chunk
            if len(chunk) < 2000:
                return feats
            offset += 2000
            print(f"  ... {len(feats)}", flush=True)

    from lib.tiles import snap

    def dominant_tile(geom):
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
        return best, bf

    BS = "ignition_date >= DATE '2019-07-01' AND ignition_date <= DATE '2020-03-31'"
    print("harvesting Black Summer wildfires (>= 2000 ha) ...", flush=True)
    wild = harvest(f"{BS} AND fire_type = 'Bushfire'", 2000)
    print(f"  {len(wild)} wildfires", flush=True)
    records, geoms = [], []
    for f in wild:
        try:
            g = shape(f["geometry"])
            if not g.is_valid:
                g = g.buffer(0)
            tile, frac = dominant_tile(g)
        except Exception:
            continue
        p = f["properties"]
        if tile is None or frac < 0.15:
            continue
        import datetime as dt
        ig = p.get("ignition_date")
        ig = str(dt.datetime.fromtimestamp(ig / 1000, tz=dt.timezone.utc).date()) if isinstance(ig, (int, float)) else str(ig)[:10]
        uid = int(p["OBJECTID"])
        records.append(dict(uid=uid, fire_id=p.get("fire_id"),
                            name=p.get("fire_name"), ig_date=ig,
                            area_ha=float(p.get("area_ha") or 0),
                            state=p.get("state"),
                            tile=f"grid_{tile[0]}_{tile[1]}",
                            burn_frac=round(frac, 3)))
        geoms.append(dict(type="Feature", properties=dict(uid=uid),
                          geometry=f["geometry"]))

    months = {}
    for r in records:
        m = r["ig_date"][:7]
        months[m] = months.get(m, 0) + 1

    print(f"\ntile-dominant wildfires: {len(records)} on "
          f"{len({r['tile'] for r in records})} tiles")

    print("by ignition month:", dict(sorted(months.items())))
    json.dump(records, open(RESULTS_DIR / "se_australia_inventory.json", "w"))
    json.dump(dict(type="FeatureCollection", features=geoms),
              open(RESULTS_DIR / "se_australia_geoms.geojson", "w"))

    print("saved results/se_australia_inventory.json")

if __name__ == "__main__":
    {"inventory": build_inventory, "fetch": fetch_all}[sys.argv[1]]()
