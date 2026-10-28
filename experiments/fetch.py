"""Download the embedding tiles for one corpus (Tessera or AlphaEarth, 2017-2025) and the reference products.

Usage: python experiments/fetch.py tiles se_us_fire          # Tessera tiles for one corpus
       python experiments/fetch.py tiles crop_atlas --aef    # the same for AlphaEarth
       python experiments/fetch.py tiles --list              # the corpora
       python experiments/fetch.py reference                 # every product
       python experiments/fetch.py reference mtbs efda       # named products
       python experiments/fetch.py reference --list          # the products
A corpus other than fires takes its tiles from its selection file in results/. efda gets only the countries in
results/europe_selection.json, or every country with --all-countries or when that file is missing.
Writes the products under data/groundtruth/, and the tile-years that failed to results/<corpus>_fetch_errors.json
(<corpus>_aef_fetch_errors.json with --aef).
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from lib.alphaearth import fetch_aef
from lib.config import ARC_GRANULES, FIRES, GT_DIR, RESULTS_DIR
from lib.remote import download, extract_members
from lib.tiles import download_landmask, download_tile

YEARS = list(range(2017, 2026))

# Each corpus: its selection file (or a function that lists its tiles), and whether it needs the land mask.
CORPORA = {
    "se_us_fire":   ("se_us_fire_selection.json", False),   # SE US wildfire and prescribed fire
    "crop_atlas":   ("crop_selection.json", True),          # crop atlas
    "windthrow":    ("windthrow_selection.json", True),     # windthrow
    "europe_atlas": ("europe_selection.json", True),        # Europe atlas
    "amazon_arc":   ("amazon_arc_selection.json", False),   # Amazon arc clearing
    "fires":    (lambda: [s["tile"] for s in FIRES.values()], True),   # the five western US fires
}

def corpus_tiles(name):
    src = CORPORA[name][0]
    if callable(src):
        return sorted(set(src()))
    sel = json.load(open(RESULTS_DIR / src))
    return sorted({r["tile"] for r in sel} if isinstance(sel, list) else set(sel))


def cmd_tiles(a):
    tiles = corpus_tiles(a.corpus)
    needs_mask = CORPORA[a.corpus][1]
    jobs = [(t, y) for t in tiles for y in YEARS]
    model = "AlphaEarth" if a.aef else "Tessera"
    print(f"{a.corpus}: {len(tiles)} tiles x {len(YEARS)} years = {len(jobs)} "
          f"{model} tile-years", flush=True)

    if needs_mask and not a.aef:
        for t in tiles:
            try:
                download_landmask(t)
            except Exception as e:
                print(f"  landmask {t}: {e!r}", flush=True)

    get = fetch_aef if a.aef else download_tile

    def one(job):
        t, y = job
        try:
            get(t, y)
            return None
        except Exception as e:
            return f"{t} {y}: {e!r}"

    errs = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            if r:
                errs.append(r)
            if i % 100 == 0:
                print(f"  [{i}/{len(jobs)}] ({len(errs)} errors)", flush=True)

    suffix = "_aef_fetch_errors.json" if a.aef else "_fetch_errors.json"
    out = RESULTS_DIR / f"{a.corpus}{suffix}"
    json.dump(errs, open(out, "w"))
    print(f"done: {len(jobs) - len(errs)}/{len(jobs)} fetched, {len(errs)} errors "
          f"-> {out}")
    for e in errs[:8]:
        print(" ", e)


# ------------------------------------------------------------- reference --

ZENODO_EFDA = "13333034"        # European Forest Disturbance Atlas
FIGSHARE_FORWIND = "9555008"    # FORWIND v2


def efda_countries():
    """Return the countries in results/europe_selection.json, or None if it doesn't exist yet (efda() then gets all)."""
    sel = RESULTS_DIR / "europe_selection.json"
    if not sel.exists():
        return None
    return sorted({r["country"] for r in json.load(open(sel)) if "country" in r})


get = download                  # lib.remote.download: curl through a .part file, keeps files that already exist


def unzip(z, into):
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(into)
    print(f"    unzipped {z.name}")


def api(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def mtbs():
    """MTBS fire perimeters, with ignition dates and incident type (western and SE US)."""
    d = GT_DIR / "mtbs"
    if (d / "mtbs_perims_DD.shp").exists():
        print("    have mtbs_perims_DD.shp")
        return
    z = get("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
            "composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip",
            d / "mtbs_perimeter_data.zip")
    unzip(z, d)


def nbac():
    """NBAC, the Canadian National Burned Area Composite: boreal fire polygons and dates."""
    d = GT_DIR / "nbac"
    if list(d.glob("*.shp")):
        print("    have NBAC shapefile")
        return
    z = get("https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/"
            "NBAC_1972to2025_20260513_shp.zip", d / "nbac_all.zip")
    unzip(z, d)


def forwind():
    """FORWIND v2: European storm-damage polygons, with event dates."""
    d = GT_DIR / "forwind"
    if (d / "FORWIND_v2.shp").exists():
        print("    have FORWIND_v2.shp")
        return
    rec = api(f"https://api.figshare.com/v2/articles/{FIGSHARE_FORWIND}")
    for f in rec["files"]:
        if f["name"].startswith("FORWIND_v2."):
            get(f["download_url"], d / f["name"])


def efda(all_countries=False):
    """European Forest Disturbance Atlas: disturbance year, agent class and forest mask. One archive per country."""
    d = GT_DIR / "efda"
    rec = api(f"https://zenodo.org/api/records/{ZENODO_EFDA}")
    avail = {f["key"][:-4]: f for f in rec["files"] if f["key"].endswith(".zip")}
    used = efda_countries()
    want = sorted(avail) if (all_countries or used is None) else \
        [c for c in used if c in avail]
    missing = [c for c in want
               if not (d / f"annual_disturbances_1985_2023_{c}.tif").exists()]
    if not missing:
        print(f"    have all {len(want)} countries")
        return
    total = sum(avail[c]["size"] for c in missing) / 1e9
    print(f"    {len(missing)} countries to fetch, ~{total:.0f} GB", flush=True)
    for c in missing:
        z = get(avail[c]["links"]["self"], d / f"{c}.zip")
        unzip(z, d)
        z.unlink()          # only the extracted rasters are kept


def hansen():
    """Hansen Global Forest Change lossyear and treecover2000 granules for the Amazon arc (ARC_GRANULES)."""
    from lib.config import HANSEN_VERSION
    for g in ARC_GRANULES:
        for layer in ("lossyear", "treecover2000"):
            get(f"https://storage.googleapis.com/earthenginepartners-hansen/"
                f"{HANSEN_VERSION}/Hansen_{HANSEN_VERSION}_{layer}_{g}.tif",
                GT_DIR / "hansen_arc" / f"{layer}_{g}.tif")


def landslides():
    """Rainfall landslide inventories: RER2023 (Emilia-Romagna), RLBJ (Beijing) and RLZX (Zixing).

    Keeps only the polygon layers. RER2023 and RLBJ are RAR archives and need unrar. Only the inventory layer is read
    from the 8 GB RLZX zip.
    """
    root = GT_DIR / "landslides"
    unrar = shutil.which("unrar") is not None
    if not unrar:
        print("  unrar not found: skipping the RER2023 and RLBJ archives")
    if unrar and not (root / "rer2023" / "RER2023_V1.shp").exists():
        rar = get("https://zenodo.org/api/records/13742643/files/RER2023_V1.rar/content", root / "rer2023" / "RER2023_V1.rar")
        subprocess.run(["unrar", "x", "-o+", "-inul", str(rar), *[f"*.{e}" for e in ("shp", "shx", "dbf", "prj", "CPG")],
                        str(rar.parent) + "/"], check=True)
        rar.unlink()
    if unrar and not (root / "beijing2023" / "Polygon_RLBJ.shp").exists():
        rar = get("https://ndownloader.figshare.com/files/48909277", root / "beijing2023" / "RLBJ_v1.0_shapefiles.rar")
        subprocess.run(["unrar", "e", "-o+", "-inul", str(rar), "RLBJ_v1.0_shapefiles/Polygon_RLBJ.*", str(rar.parent) + "/"],
                       check=True)
        (rar.parent / "Polygon_RLBJ.shp.xml").unlink(missing_ok=True)
        rar.unlink()
    if not (root / "zixing2024" / "RLZX-LIM.shp").exists():
        extract_members("https://ndownloader.figshare.com/files/50973759",
                        lambda n: n.startswith("RLZX-LIM.") and not n.endswith(".xml"), root / "zixing2024")


def uspvdb():
    """US Large-Scale Solar Photovoltaic Database (USGS/LBNL): solar farm outlines as GeoJSON, in groundtruth/solar/."""
    if any((GT_DIR / "solar").glob("uspvdb_*.geojson")):
        return
    zp = get("https://energy.usgs.gov/uspvdb/assets/data/uspvdbGeoJSON.zip", GT_DIR / "solar" / "uspvdbGeoJSON.zip")
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            if m.endswith(".geojson") and not (zp.parent / Path(m).name).exists():
                with z.open(m) as src, open(zp.parent / Path(m).name, "wb") as out:
                    shutil.copyfileobj(src, out)
    zp.unlink()


def naturalearth():
    """Natural Earth 1:110m land polygons (public domain), for the world map of settings."""
    out = GT_DIR / "ne_110m_land.geojson"
    if not out.exists():
        get("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson", out)


PRODUCTS = {"mtbs": mtbs, "nbac": nbac, "forwind": forwind, "efda": efda,
            "hansen": hansen, "landslides": landslides, "uspvdb": uspvdb, "naturalearth": naturalearth}


def cmd_reference(a):
    if a.list:
        for k in sorted(PRODUCTS):
            print(f"  {k:<10} {PRODUCTS[k].__doc__.splitlines()[0]}")
        return
    unknown = [p for p in a.products if p not in PRODUCTS]
    if unknown:
        sys.exit(f"unknown product(s): {unknown}. Known: {sorted(PRODUCTS)}")
    if shutil.which("curl") is None:
        sys.exit("curl not found")
    for name in (a.products or sorted(PRODUCTS)):
        print(f"{name}:")
        fn = PRODUCTS[name]
        fn(all_countries=a.all_countries) if name == "efda" else fn()
    print(f"\nreference products under {GT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiles", help="embedding tiles for one corpus")
    t.add_argument("corpus", nargs="?", choices=sorted(CORPORA))
    t.add_argument("--aef", action="store_true",
                   help="fetch AlphaEarth instead of Tessera")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--list", action="store_true")
    t.set_defaults(fn=cmd_tiles)
    r = sub.add_parser("reference", help="reference maps: fire perimeters, disturbance atlases, landslides, solar farms, land outlines")
    r.add_argument("products", nargs="*", metavar="PRODUCT",
                   help=f"any of: {', '.join(sorted(PRODUCTS))} (default: all)")
    r.add_argument("--all-countries", action="store_true",
                   help="EFDA: fetch every country, not just those used")
    r.add_argument("--list", action="store_true")
    r.set_defaults(fn=cmd_reference)
    a = ap.parse_args()
    if a.cmd == "tiles":
        if a.list or not a.corpus:
            for k in sorted(CORPORA):
                src = CORPORA[k][0]
                print(f"  {k:<10} {'lib/config.py' if callable(src) else src}")
            return
    a.fn(a)


if __name__ == "__main__":
    main()
