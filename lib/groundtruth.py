"""Fetch reference rasters on the tile grid: Sentinel-2 dNBR, ESA WorldCover v200, Copernicus GLO-30 terrain,
USDA CDL, WUR RADD alerts and Sentinel-2 composites.

All come from the Planetary Computer STAC, except RADD, which comes from the GFW Data API.
"""
import functools
import warnings

import numpy as np

from lib.config import STAC_URL, DNBR_BURN_THRESHOLD
from lib.tiles import tile_bbox, tile_grid

CLEAR_SCL = (4, 5, 6, 7)


def _quiet(fn):
    """Silence library warnings (all-NaN medians, georeferencing notes) while a fetch runs, and only then."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)
    return wrapper


def _utm_epsg(tile):
    """EPSG code of the UTM zone at the tile's center."""
    lon0, lat0, lon1, lat1 = tile_bbox(tile)
    lon, lat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1

def _catalog():
    import pystac_client, planetary_computer
    return pystac_client.Client.open(STAC_URL,
                                     modifier=planetary_computer.sign_inplace)

def to_grid(arr, x, y, src_crs, tile, resampling="bilinear", nodata=np.nan):
    """Reproject an array onto the tile grid, given its pixel-center coordinates x and y."""
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    dst_t, dst_crs, H, W = tile_grid(tile)
    resx, resy = x[1] - x[0], abs(y[1] - y[0])
    t = from_origin(x[0] - resx / 2, y[0] + resy / 2, resx, resy)
    dst = np.full((H, W), nodata, dtype=np.float64)
    reproject(np.asarray(arr, dtype=np.float64), dst, src_transform=t,
              src_crs=src_crs, dst_transform=dst_t, dst_crs=dst_crs,
              resampling=getattr(Resampling, resampling),
              src_nodata=nodata, dst_nodata=nodata)
    return dst

def crop_to_grid(arr, transform, tile, nodata, src_crs="EPSG:4326"):
    """Reproject a label crop with an affine transform (6 numbers) onto the tile grid. Nearest neighbor, same dtype."""
    from affine import Affine
    from rasterio.warp import reproject, Resampling
    dst_t, dst_crs, H, W = tile_grid(tile)
    dst = np.full((H, W), nodata, dtype=arr.dtype)
    reproject(arr, dst, src_transform=Affine(*transform), src_crs=src_crs, dst_transform=dst_t, dst_crs=dst_crs,
              resampling=Resampling.nearest, src_nodata=nodata, dst_nodata=nodata)
    return dst

def _boa_offset(items):
    """The L2A offset of each scene in digital numbers, where reflectance = (DN - offset) / 10000. The offset is 1000
    from processing baseline 04.00 on (scenes from 2022-01-25), and 0 before."""
    return np.array([1000.0 if float(it.properties.get("s2:processing_baseline", "0")) >= 4 else 0.0
                     for it in items])

def _nbr_composite(cat, bbox, datetime, epsg, max_cloud=40):
    import stackstac
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                            datetime=datetime,
                            query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    da = stackstac.stack(items, assets=["B8A", "B12", "SCL"], bounds_latlon=bbox,
                         resolution=20, dtype="float64", fill_value=np.nan,
                         rescale=False, epsg=epsg, sortby_date=False).compute(scheduler="threads")
    arr = da.values
    off = _boa_offset(items)[:, None, None]          # scenes stay in search order, so offsets line up
    b8a, b12, scl = arr[:, 0] - off, arr[:, 1] - off, arr[:, 2]
    clear = np.isin(scl, CLEAR_SCL)
    b8a = np.where(clear, b8a, np.nan)
    b12 = np.where(clear, b12, np.nan)
    with np.errstate(invalid="ignore"):
        nbr = (b8a - b12) / (b8a + b12)
    return np.nanmedian(nbr, axis=0), da.x.values, da.y.values

@_quiet
def fetch_dnbr(tile, pre_window, post_window, out_path=None, epsg=None):
    """dNBR on the tile grid: median NBR before the fire minus after, over clear Sentinel-2 L2A pixels. Cached."""
    if out_path is not None and out_path.exists():
        return np.load(out_path)
    epsg = epsg or _utm_epsg(tile)
    cat = _catalog()
    bbox = tile_bbox(tile)
    pre, x, y = _nbr_composite(cat, bbox, pre_window, epsg)
    post, _, _ = _nbr_composite(cat, bbox, post_window, epsg)
    dnbr = to_grid(pre - post, x, y, f"EPSG:{epsg}", tile).astype(np.float32)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, dnbr)
    return dnbr

RADD_EPOCH = "2014-12-31"          # RADD date_conf counts days from this date

def _gfw_key():
    """Read the GFW Data API key from $GFW_API_KEY or ~/.config/gfw_api_key."""
    import os
    from pathlib import Path
    k = os.environ.get("GFW_API_KEY")
    if k:
        return k.strip()
    p = Path.home() / ".config" / "gfw_api_key"
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError("No GFW API key: set $GFW_API_KEY or ~/.config/gfw_api_key")

@_quiet
def fetch_radd(tile, granule, version="v20260802", out_path=None):
    """Read WUR RADD alerts (date_conf, 10 m) for the tile from one 10-degree granule, onto the tile grid.

    Needs a GFW key. Returns uint16. 0 means no alert. Otherwise the value is conf * 10000 + days since
    RADD_EPOCH, where conf 2 is low (unconfirmed) and 3 is high (confirmed). Granules are named by their top-left
    corner, like Hansen's.
    """
    import subprocess
    import rasterio
    from rasterio.warp import reproject, Resampling

    if out_path is not None and out_path.exists():
        return np.load(out_path)

    api = (f"https://data-api.globalforestwatch.org/dataset/wur_radd_alerts/"
           f"{version}/download/geotiff?grid=10/100000&tile_id={granule}"
           f"&pixel_meaning=date_conf")
    r = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{redirect_url}",
                        "-H", f"x-api-key: {_gfw_key()}", api],
                       capture_output=True, text=True)
    url = r.stdout.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"GFW API returned no presigned URL for {granule}")

    lonmin, latmin, lonmax, latmax = tile_bbox(tile, pad=0.005)
    dst_t, dst_crs, H, W = tile_grid(tile)
    with rasterio.open("/vsicurl/" + url) as src:
        win = src.window(lonmin, latmin, lonmax, latmax)
        arr = src.read(1, window=win)
        wt = src.window_transform(win)
    dst = np.zeros((H, W), dtype=np.float64)
    reproject(arr.astype(np.float64), dst, src_transform=wt, src_crs="EPSG:4326",
              dst_transform=dst_t, dst_crs=dst_crs, resampling=Resampling.nearest)
    dst = dst.astype(np.uint16)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, dst)
    return dst

def radd_decode(v):
    """Split RADD date_conf values into (confidence, date as numpy datetime64[D])."""
    conf = (v // 10000).astype(np.uint8)
    days = (v % 10000).astype(np.int32)
    dates = np.datetime64(RADD_EPOCH) + days.astype("timedelta64[D]")
    return conf, dates

def burn_masks(dnbr, threshold=DNBR_BURN_THRESHOLD):
    """Return (burned, unburned) boolean masks. Pixels with NaN dNBR are in neither."""
    valid = ~np.isnan(dnbr)
    return (dnbr > threshold) & valid, ~(dnbr > threshold) & valid

@_quiet
def fetch_worldcover(tile, out_path=None):
    """ESA WorldCover v200 classes (uint8) on the tile grid."""
    if out_path is not None and out_path.exists():
        return np.load(out_path)
    import stackstac
    cat = _catalog()
    bbox = tile_bbox(tile)
    items = list(cat.search(collections=["esa-worldcover"], bbox=bbox,
                            query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    da = stackstac.stack(items, assets=["map"], bounds_latlon=bbox,
                         resolution=0.0000898311, dtype="float64",
                         fill_value=np.nan, rescale=False,
                         epsg=4326).compute(scheduler="threads")
    wc = np.nanmax(da.values, axis=0)[0]
    grid = to_grid(wc, da.x.values, da.y.values, "EPSG:4326", tile,
                   resampling="nearest", nodata=0).astype(np.uint8)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, grid)
    return grid

@_quiet
def fetch_terrain(tile, out_path=None, smooth_sigma=3):
    """Turn the Copernicus GLO-30 DEM into dict(dem, slope, northness, eastness) on the tile grid.

    Aspect is the direction a slope faces, so northness is +1 on north-facing slopes.
    """
    if out_path is not None and out_path.exists():
        return dict(np.load(out_path))
    import stackstac
    from scipy.ndimage import gaussian_filter
    cat = _catalog()
    bbox = tile_bbox(tile, pad=0.01)
    items = list(cat.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    da = stackstac.stack(items, assets=["data"], bounds_latlon=bbox,
                         resolution=0.000277778, dtype="float64",
                         fill_value=np.nan, epsg=4326).compute(scheduler="threads")
    dem = np.nanmean(da.values, axis=0)[0]
    grid = to_grid(dem, da.x.values, da.y.values, "EPSG:4326", tile)
    z = gaussian_filter(np.nan_to_num(grid, nan=np.nanmean(grid)), smooth_sigma)
    gy, gx = np.gradient(z, 10.0)          # rows run from north to south
    slope = np.hypot(gx, gy)
    out = dict(dem=grid.astype(np.float32), slope=slope.astype(np.float32),
               northness=(gy / (slope + 1e-6)).astype(np.float32),
               eastness=(-gx / (slope + 1e-6)).astype(np.float32))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **out)
    return out

@_quiet
def fetch_cdl(tile, year):
    """USDA Cropland Data Layer classes (uint8) on the tile grid, nearest neighbor. None if the year has no CDL."""
    import stackstac
    cat = _catalog()
    bbox = tile_bbox(tile, pad=0.01)
    items = [it for it in cat.search(collections=["usda-cdl"], bbox=bbox,
                                     datetime=f"{year}-01-01/{year}-12-31").items()
             if it.properties.get("usda_cdl:type") == "cropland"]
    if not items:
        return None
    da = stackstac.stack(items[:1], assets=["cropland"], bounds_latlon=bbox,
                         resolution=30, dtype="float64",
                         fill_value=np.nan).compute(scheduler="threads")
    arr = np.nan_to_num(da.values[0, 0], nan=0)
    return to_grid(arr, da.x.values, da.y.values,
                   da.attrs.get("crs", "EPSG:5070"), tile,
                   resampling="nearest", nodata=0).astype(np.uint8)


@_quiet
def fetch_s2_rgb(tile, window, out_path=None, epsg=None, max_cloud=20,
                 bands=("B04", "B03", "B02"), clear_classes=CLEAR_SCL):
    """Median cloud-free Sentinel-2 L2A reflectance of three bands, (H, W, 3) on the tile grid. Cached to out_path.

    clear_classes are the SCL classes to keep. The default also drops shadow. The default bands give true color, and
    ("B12", "B8A", "B04") gives the SWIR false color for burn scars.
    """
    if out_path is not None and out_path.exists():
        return np.load(out_path)
    import stackstac
    epsg = epsg or _utm_epsg(tile)
    cat = _catalog()
    bbox = tile_bbox(tile)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=window,
                            query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    da = stackstac.stack(items, assets=[*bands, "SCL"], bounds_latlon=bbox,
                         resolution=10, dtype="float64", fill_value=np.nan,
                         rescale=False, epsg=epsg, sortby_date=False).compute(scheduler="threads")
    arr = da.values
    off = _boa_offset(items)
    clear = np.isin(arr[:, 3], clear_classes)
    rgb = [np.nanmedian(np.where(clear, (arr[:, i] - off[:, None, None]) / 10000, np.nan), axis=0)
           for i in range(3)]
    out = np.stack([to_grid(b, da.x.values, da.y.values, f"EPSG:{epsg}", tile)
                    for b in rgb], -1).astype(np.float32)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out)
    return out
