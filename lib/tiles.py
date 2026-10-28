"""GeoTessera v1 embedding tiles: download, dequantize, tile grids, and which blocks Tessera covers.

A tile is 0.1 degree across at 10 m: an int8 array of 128 bands, with a float32 scale per pixel. Tiles are named
grid_{lon:.2f}_{lat:.2f} after their center, and centers sit at x.x5. We build URLs directly instead of using the
geotessera downloader, because it loads its whole tile registry into memory. The local cache uses the geotessera
directory layout (REPR_DIR, MASK_DIR).
"""
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests

from lib.config import EMB_DIR, GEOTESSERA_BASE_URL, GEOTESSERA_VERSION

REPR_DIR = "global_0.1_degree_representation"
MASK_DIR = "global_0.1_degree_tiff_all"

def snap(v: float) -> float:
    """Center coordinate of the 0.1-degree tile that contains v. Centers sit at x.x5.

    v is rounded to 6 decimals first. Otherwise float noise (26.3 stored as 26.29999999) could move it a tile down.
    """
    return round(math.floor(round(v, 6) / 0.1) * 0.1 + 0.05, 2)

def tile_for_point(lon: float, lat: float) -> str:
    """Name of the tile containing (lon, lat), such as grid_-81.05_26.35."""
    return f"grid_{snap(lon)}_{snap(lat)}"

def tile_center(tile: str):
    """(lon, lat) of a tile's center, parsed from its name."""
    lon, lat = (float(v) for v in tile.replace("grid_", "").split("_"))
    return lon, lat

def tile_bbox(tile: str, pad: float = 0.0):
    lon, lat = tile_center(tile)
    return [lon - 0.05 - pad, lat - 0.05 - pad, lon + 0.05 + pad, lat + 0.05 + pad]

def embedding_url(tile: str, year: int, suffix: str = "") -> str:
    """Remote URL of a tile-year's embedding (suffix "") or scales ("_scales")."""
    return (f"{GEOTESSERA_BASE_URL}/npy/{GEOTESSERA_VERSION}/"
            f"{year}/{tile}/{tile}{suffix}.npy")

def landmask_url(tile: str) -> str:
    return f"{GEOTESSERA_BASE_URL}/landmasks/{GEOTESSERA_VERSION}/{tile}.tiff"

def _fetch(url: str, dest: Path, deadline: float = 300, tries: int = 3):
    """Stream url to dest through a .part file. Does nothing if dest is already there.

    Some mirror connections stall but keep sending a trickle of bytes. A read timeout never catches that, so each
    attempt also has a hard deadline.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        t0 = time.monotonic()
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
                        if time.monotonic() - t0 > deadline:
                            raise TimeoutError(f"{url}: over {deadline}s")
            os.replace(tmp, dest)
            return dest
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == tries - 1:
                raise

def download_tile(tile: str, year: int) -> Path:
    """Fetch the embedding and scales npy files for one tile-year. Files already present are kept."""
    d = EMB_DIR / REPR_DIR / str(year) / tile
    for suf in ("", "_scales"):
        _fetch(embedding_url(tile, year, suf), d / f"{tile}{suf}.npy")
    return d

def download_landmask(tile: str) -> Path:
    return _fetch(landmask_url(tile), EMB_DIR / MASK_DIR / f"{tile}.tiff")

def load_tile(tile: str, year: int, stride: int = 1) -> np.ndarray:
    """Load a tile-year as dequantized float32 (H, W, 128). stride > 1 keeps every stride-th pixel."""
    d = EMB_DIR / REPR_DIR / str(year) / tile
    e = np.load(d / f"{tile}.npy", mmap_mode="r")
    s = np.load(d / f"{tile}_scales.npy")
    if stride > 1:
        e, s = e[::stride, ::stride], s[::stride, ::stride]
    return e.astype(np.float32) * s[..., None]

def tile_grid(tile: str):
    """(transform, crs, H, W) of the 10 m tile grid, read from the landmask tiff. The tiff is fetched if missing."""
    import rasterio
    p = EMB_DIR / MASK_DIR / f"{tile}.tiff"
    if not p.exists():
        download_landmask(tile)
    with rasterio.open(p) as src:
        return src.transform, src.crs, src.height, src.width

def sample_pixels(tile_years, n_per: int = 100_000, seed: int = 0) -> np.ndarray:
    """Dictionary training sample: n_per uniform pixels from each (tile, year) in tile_years, stacked together."""
    rng = np.random.default_rng(seed)
    out = []
    for tile, year in tile_years:
        d = EMB_DIR / REPR_DIR / str(year) / tile
        e = np.load(d / f"{tile}.npy", mmap_mode="r")
        s = np.load(d / f"{tile}_scales.npy")
        H, W, _ = e.shape
        idx = rng.choice(H * W, n_per, replace=False)
        r, c = idx // W, idx % W
        out.append(e[r, c].astype(np.float32) * s[r, c, None])
    return np.concatenate(out)

def available_years(tiles, years, workers: int = 24) -> dict:
    """Which years exist on the mirror for each tile, as {tile: set of years}. Asks for one byte of each scales file."""
    from concurrent.futures import ThreadPoolExecutor
    def probe(ty):
        try:
            with requests.get(embedding_url(*ty, "_scales"), stream=True, timeout=(15, 60),
                              headers={"Range": "bytes=0-0"}) as r:
                return ty, r.status_code in (200, 206)
        except Exception:
            return ty, False
    have = {t: set() for t in tiles}
    with ThreadPoolExecutor(workers) as ex:
        for (t, y), ok in ex.map(probe, [(t, y) for t in tiles for y in years]):
            if ok:
                have[t].add(y)
    return have

def _remove_tile_year(tile: str, year: int):
    d = EMB_DIR / REPR_DIR / str(year) / tile
    for p in [d / f"{tile}.npy", d / f"{tile}_scales.npy", *d.glob("*.part")]:
        p.unlink(missing_ok=True)
    try:
        d.rmdir()
    except OSError:
        pass

@contextmanager
def transient_tile(tile: str, year: int):
    """Download a tile-year if missing and delete it on exit, partial downloads too. One already on disk is kept."""
    had = (EMB_DIR / REPR_DIR / str(year) / tile / f"{tile}.npy").exists()
    try:
        yield download_tile(tile, year)
    finally:
        if not had:
            _remove_tile_year(tile, year)

def coverage_blocks(path: Path, years=(2017, 2022), step: float = 2.5) -> list:
    """Lower-left corners (lon, lat) of the step-degree blocks where Tessera has every year in `years`.

    Two tiles per block are checked. A block counts if either one has all the years. Cached at `path`.
    """
    import json
    if path.exists():
        return [tuple(b) for b in json.load(open(path))]
    blocks = [(float(x), float(y)) for x in np.arange(-180, 180, step) for y in np.arange(-57.5, 75, step)]
    probes = {b: [tile_for_point(b[0] + d, b[1] + d) for d in (0.26 * step, 0.74 * step)] for b in blocks}
    have = available_years(sorted({t for ts in probes.values() for t in ts}), years, workers=48)
    out = [b for b, ts in probes.items() if any(set(years) <= have[t] for t in ts)]
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out

def block_of(lon: float, lat: float, step: float = 2.5):
    """Lower-left corner of the step-degree block containing (lon, lat), as coverage_blocks lists them."""
    return (math.floor(lon / step) * step, math.floor(lat / step) * step)

def granules(blocks, size: int = 10, step: float = 2.5) -> list:
    """(left, top) corners of the size-degree source granules that any of the blocks overlap. Hansen and JRC name
    their 10-degree files by the upper-left corner."""
    out = set()
    for x, y in blocks:
        for cx, cy in ((x, y), (x + step - 0.1, y + step - 0.1)):
            out.add((int(math.floor(cx / size) * size), int(math.floor(cy / size) * size + size)))
    return sorted(out)
