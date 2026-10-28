"""Load AlphaEarth Foundations embeddings (Satellite Embedding V1, annual) onto GeoTessera tile grids.

The files are read anonymously from Google's public GCS bucket. They are 64-band int8 COGs on UTM grids. Values are
dequantized as sign(v) * (|v|/127.5)^2, and -128 means nodata. License CC-BY 4.0, Google and Google DeepMind.
"""
import numpy as np

from lib.config import DATA_DIR

AEF_BASE = "https://storage.googleapis.com/alphaearth_foundations/"
AEF_DIR = DATA_DIR / "alphaearth"
INDEX_PATH = AEF_DIR / "aef_index.parquet"

def _index():
    import pyarrow.parquet as pq
    import requests
    if not INDEX_PATH.exists():
        AEF_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(AEF_BASE + "satellite_embedding/v1/annual/aef_index.parquet",
                         timeout=600)
        r.raise_for_status()
        INDEX_PATH.write_bytes(r.content)
    return pq.read_table(INDEX_PATH,
                         columns=["path", "year", "crs",
                                  "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"])

def dequantize(v):
    x = v.astype(np.float32)
    out = np.sign(x) * (np.abs(x) / 127.5) ** 2
    out[v == -128] = np.nan
    return out

def fetch_aef(tile, year, out_path=None):
    """Raw int8 (H, W, 64) AlphaEarth embedding of a tile-year on the tile's 10 m grid. Nearest neighbor, cached."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from lib.tiles import tile_bbox, tile_grid

    out_path = out_path or AEF_DIR / f"aef_{tile}_{year}.npy"
    if out_path.exists():
        return np.load(out_path)
    lonmin, latmin, lonmax, latmax = tile_bbox(tile, pad=0.003)
    t = _index()
    import pyarrow.compute as pc
    m = pc.and_(pc.equal(t["year"], year),
                pc.and_(pc.and_(pc.less(t["wgs84_west"], lonmax),
                                pc.greater(t["wgs84_east"], lonmin)),
                        pc.and_(pc.less(t["wgs84_south"], latmax),
                                pc.greater(t["wgs84_north"], latmin))))
    rows = t.filter(m).to_pylist()
    if not rows:
        raise FileNotFoundError(f"no AlphaEarth granules for {tile} {year}")
    dst_t, dst_crs, H, W = tile_grid(tile)
    dst = np.full((64, H, W), -128, dtype=np.int8)
    for row in rows:
        url = "/vsicurl/" + AEF_BASE + row["path"].replace("gs://alphaearth_foundations/", "")
        with rasterio.open(url) as src:
            from rasterio.warp import transform_bounds
            l, b, r_, tp = transform_bounds("EPSG:4326", src.crs,
                                            lonmin, latmin, lonmax, latmax)
            # Some AlphaEarth COGs are stored bottom-up, so build the window by hand.
            inv = ~src.transform
            cs, rs = zip(*(inv * xy for xy in ((l, b), (r_, tp), (l, tp), (r_, b))))
            win = rasterio.windows.Window(
                col_off=int(np.floor(min(cs))), row_off=int(np.floor(min(rs))),
                width=int(np.ceil(max(cs) - min(cs))),
                height=int(np.ceil(max(rs) - min(rs))))
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                continue
            arr = src.read(window=win)
            wt = src.window_transform(win)
            piece = np.full((64, H, W), -128, dtype=np.int8)
            reproject(arr, piece, src_transform=wt, src_crs=src.crs,
                      dst_transform=dst_t, dst_crs=dst_crs,
                      resampling=Resampling.nearest,
                      src_nodata=-128, dst_nodata=-128)
            fill = piece != -128
            dst[fill] = piece[fill]
    out = np.transpose(dst, (1, 2, 0))
    AEF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    return out

def load_aef(tile, year, stride=1):
    """Load a tile-year as dequantized float32 (H, W, 64)."""
    raw = fetch_aef(tile, year)
    if stride > 1:
        raw = raw[::stride, ::stride]
    return dequantize(raw)

def compute_scale():
    """One global factor that puts AlphaEarth embeddings on Tessera's scale.

    It is the standard deviation of Tessera pixels on the Dixie tile (2019-2020) over that of AlphaEarth pixels on the
    same tile (2019-2024). AlphaEarth embeddings have unit length, so their values are far smaller than Tessera's.
    Every AlphaEarth screen multiplies by this factor.
    """
    from lib.config import DIXIE, DIXIE_SAE_YEARS, FIRES
    from lib.tiles import sample_pixels
    tile = FIRES[DIXIE]["tile"]
    t = sample_pixels([(tile, y) for y in DIXIE_SAE_YEARS[:2]], n_per=50_000, seed=0)
    rng = np.random.default_rng(0)
    araw = []
    for y in DIXIE_SAE_YEARS:
        raw = fetch_aef(tile, y)
        H, W, _ = raw.shape
        idx = rng.choice(H * W, 20_000, replace=False)
        araw.append(dequantize(raw.reshape(-1, 64)[idx]))
    araw = np.nan_to_num(np.concatenate(araw))
    return float(t.std() / araw.std())
