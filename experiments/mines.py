"""Mining land: find a family that tells mining land from its surroundings, and test it on held-out mine sites.

Reference product: the global mining land-use dataset. It labels the pixels inside hand-drawn mine outlines as open pit,
waste dumping site, water body, facility, general disturbed land, bare soil or vegetation. Mining land is the first
five classes. Units are sites with at least 0.5 km2 of mining land, up to 30 per world region (seed 0) and one per
0.1 degree tile. Each is read in the Tessera year closest to its label year, within 2 years. A site keeps up to 3,000
full-resolution pixels of each mining class and of its surroundings, which lie more than 500 m from the site's
polygons. It needs at least 300 mining pixels and 50 surrounding pixels. The mine dictionary is trained on 5,000
pixels from each site. The family is screened on a random fifth of the sites (seed 0) and scored on the rest with
lib/detect.py. verdicts also screens on four random regions and scores the others.

Usage: python experiments/mines.py select|fetch|train|screen|verdicts
Writes results/mine_selection.json, mines.jsonl and mines.json, and the dictionary checkpoint.
"""
import io
import json
import sys
import zlib

import numpy as np

from lib.config import CKPT_DIR, DATA_DIR, GT_DIR, RESULTS_DIR
from lib import detect
from lib.remote import HTTPRange
from lib.tiles import available_years, download_landmask, load_tile, tile_for_point, tile_grid, transient_tile
from lib.sae import encode_array, load_sae, train_and_save
from lib.sampling import corpus, sample_groups

ZENODO = "https://zenodo.org/api/records/15726306/files/{}/content"
REGIONS = ["Africa", "Eastern_Asia", "Eastern_Europe", "Latin_America_and_the_Caribbean", "Northern_America", "Oceania",
           "Rest_of_Europe", "Rest_of_Asia", "South_Eastern_Asia"]
MINING = ("Open pit", "Waste dumping site", "Water body", "Facility", "General disturbed land")
SITES_PER_REGION, MIN_MINING_M2, CAP, N_TRAIN = 30, 5e5, 3000, 5000
M = 1024
MG = GT_DIR / "mines"
PIX = DATA_DIR / "intermediate" / "mines"
CKPT = CKPT_DIR / "mines" / "sae_mines.pt"
SEL_PATH = RESULTS_DIR / "mine_selection.json"
JSONL = RESULTS_DIR / "mines.jsonl"
SEL = json.load(open(SEL_PATH)) if SEL_PATH.exists() else []


def _member(region):
    """Copy the region's smallest GeoPackage out of the remote zip into MG/raw/, without downloading the whole zip."""
    import shutil
    import zipfile
    raw = MG / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with io.BufferedReader(HTTPRange(ZENODO.format(region + ".zip")), buffer_size=2**23) as f:
        z = zipfile.ZipFile(f)
        info = min(z.infolist(), key=lambda i: i.file_size)
        dest = raw / info.filename.split("/")[-1]
        if not dest.exists():
            with z.open(info) as src, open(dest.with_suffix(".part"), "wb") as out:
                shutil.copyfileobj(src, out, 2**22)
            dest.with_suffix(".part").rename(dest)
    return dest


def _available(tile, years):
    """Return the given years that the mirror has for this tile, kept in the same order."""
    have = available_years([tile], years)[tile]
    return [y for y in years if y in have]


def phase_select():
    import pyogrio
    rng = np.random.default_rng(0)
    sel = []
    for region in REGIONS:
        path = _member(region)
        layer = pyogrio.list_layers(path)[0][0]
        a = pyogrio.read_dataframe(path, layer=layer, read_geometry=False, columns=["index_n", "class_name", "area_m2", "year"])
        piv = a.pivot_table(index="index_n", columns="class_name", values="area_m2", aggfunc="sum").fillna(0)
        mining = piv[[c for c in MINING if c in piv]].sum(1)
        year = a.groupby("index_n").year.first()
        cand = list(mining[mining >= MIN_MINING_M2].index)
        rng.shuffle(cand)
        got, tiles = [], set()
        for k in range(0, len(cand), 40):
            if len(got) >= SITES_PER_REGION:
                break
            chunk = cand[k:k + 40]
            g = pyogrio.read_dataframe(path, layer=layer, where=f"index_n IN ({','.join(str(int(i)) for i in chunk)})")
            g = g.to_crs(4326)
            for sid in chunk:
                if len(got) >= SITES_PER_REGION:
                    break
                s = g[g.index_n == sid]
                m = s[s.class_name.isin(MINING)]
                if m.empty:
                    continue
                c = m.dissolve().geometry.iloc[0].representative_point()
                tile = tile_for_point(c.x, c.y)
                if tile in tiles:
                    continue
                y0 = int(year[sid])
                avail = _available(tile, sorted({y0, y0 - 1, y0 + 1, y0 - 2, y0 + 2} & set(range(2017, 2026)), key=lambda y: abs(y - y0)))
                if not avail:
                    continue
                tiles.add(tile)
                got.append(dict(site=int(sid), region=region, tile=tile, label_year=y0, year=avail[0],
                                lon=float(c.x), lat=float(c.y), mining_m2=float(mining[sid]),
                                class_m2={k_: float(piv.loc[sid].get(k_, 0)) for k_ in piv.columns}))
                s[["index_n", "class_name", "area_m2", "year", "geometry"]].to_file(MG / "sites.gpkg", layer=f"site_{int(sid)}",
                                                                                    driver="GPKG")
        print(f"{region}: {len(cand)} candidate sites, {len(got)} selected", flush=True)
        sel += got
        path.unlink()                                       # keep only the selected sites' polygons
    json.dump(sel, open(SEL_PATH, "w"), indent=1)
    print(f"{len(sel)} sites selected", flush=True)


def _masks(rec):
    """Return full-resolution masks for each mining class and the surroundings, from MG/masks/ or the site polygons."""
    f = MG / "masks" / f"site_{rec['site']}.npz"
    if f.exists():
        z = np.load(f)
        H, W = z["shape"]
        return {k: np.unpackbits(z[k.replace(" ", "_")])[:H * W].reshape(H, W).astype(bool)
                for k in list(MINING) + ["surroundings"]}
    import geopandas as gpd
    from rasterio import features
    from scipy.ndimage import distance_transform_edt
    t, crs, H, W = tile_grid(rec["tile"])
    g = gpd.read_file(MG / "sites.gpkg", layer=f"site_{rec['site']}").to_crs(crs)
    out = {}
    for c in MINING:
        geoms = list(g[g.class_name == c].geometry)
        out[c] = features.rasterize(((x, 1) for x in geoms), out_shape=(H, W), transform=t, fill=0,
                                    dtype=np.uint8).astype(bool) if geoms else np.zeros((H, W), bool)
    anyl = features.rasterize(((x, 1) for x in g.geometry), out_shape=(H, W), transform=t, fill=0,
                              dtype=np.uint8).astype(bool)
    out["surroundings"] = distance_transform_edt(~anyl) * 10 > 500
    return out


def fetch_one(rec):
    out = PIX / f"site_{rec['site']}.npz"
    if out.exists():
        return rec["site"], "cached"
    try:
        download_landmask(rec["tile"])
        masks = _masks(rec)
        (MG / "masks").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(MG / "masks" / f"site_{rec['site']}.npz", shape=np.array(next(iter(masks.values())).shape),
                            **{k.replace(" ", "_"): np.packbits(m) for k, m in masks.items()})
        with transient_tile(rec["tile"], rec["year"]):
            x = load_tile(rec["tile"], rec["year"])
        rng = np.random.default_rng(zlib.crc32(str(rec["site"]).encode()))
        groups, train, _ = sample_groups(x, masks, CAP, N_TRAIN, rng)
        PIX.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **groups, _train=train)
        return rec["site"], ""
    except Exception as e:
        return rec["site"], repr(e)


def phase_fetch():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch_one, SEL))
    bad = [(s, e) for s, e in res if e and e != "cached"]
    print(f"{len(res) - len(bad)}/{len(res)} sites sampled; failed: {bad[:10]}", flush=True)


def phase_train():
    if CKPT.exists():
        print("cached"); return
    X = corpus([PIX / f"site_{r['site']}.npz" for r in SEL])
    print(f"mine corpus {X.shape}", flush=True)
    train_and_save(X, CKPT)


def phase_screen():
    """Encode each site's stored pixels. Class means go to mines.jsonl and pixel activations to PIX."""
    import torch
    sae = load_sae(path=CKPT)
    torch.set_num_threads(16)
    sids = sorted(r["site"] for r in SEL)
    screen = set(np.random.default_rng(0).choice(sids, len(sids) // 5, replace=False).tolist())
    PIX.mkdir(parents=True, exist_ok=True)
    with open(JSONL, "w") as out:
        for r in SEL:
            f = PIX / f"site_{r['site']}.npz"
            if not f.exists():
                out.write(json.dumps(dict(ok=False, site=r["site"], error="not sampled")) + "\n")
                continue
            z = np.load(f)
            means, n = {}, {}
            for k in list(MINING) + ["surroundings"]:
                a = z[k].astype(np.float32)
                n[k] = len(a)
                if len(a) >= 50:
                    F = encode_array(sae, a)
                    means[k] = F.mean(0).tolist()
                    np.save(PIX / f"site_{r['site']}_{k.replace(' ', '_')}_act.npy", F.astype(np.float16))
            ok = "surroundings" in means and sum(n[c] for c in MINING) >= 300
            out.write(json.dumps(dict(ok=ok, site=r["site"], region=r["region"], tile=r["tile"], year=r["year"],
                                      screen=r["site"] in screen, n=n, means=means)) + "\n")
    verdicts()


def _mining_mean(r):
    """Mean activation over the site's mining classes, weighted by pixel count."""
    cs = [c for c in MINING if c in r["means"]]
    return np.sum([np.array(r["means"][c]) * r["n"][c] for c in cs], 0) / sum(r["n"][c] for c in cs)


def _pairs(rows):
    return [(_mining_mean(r), np.array(r["means"]["surroundings"])) for r in rows]


def _sites(rows):
    """For each site, return its record, its mining-pixel activations and its surrounding-pixel activations."""
    def load(r, k):
        return np.load(PIX / f"site_{r['site']}_{k.replace(' ', '_')}_act.npy").astype(np.float32)
    return [(dict(site=r["site"], region=r["region"]),
             np.concatenate([load(r, c) for c in MINING if c in r["means"]]), load(r, "surroundings")) for r in rows]


def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    scr, held = [r for r in rows if r["screen"]], [r for r in rows if not r["screen"]]
    pairs = _pairs(scr)
    fam, cut = detect.screen(pairs)
    if not fam:
        print(f"{len(scr)} screening sites: no family passes the bar")
        json.dump(dict(family=[], cutoff=None, n_screen=len(scr)), open(RESULTS_DIR / "mines.json", "w"), indent=1)
        return
    sites = _sites(held)
    v = detect.score(sites, fam, cut)
    res = dict(family=fam, cutoff=cut, n_screen=len(scr),
               **detect.summary(v, detect.null_rates(pairs, sites, len(fam), M, "Mine presence")), outcomes=v)
    print(f"{len(scr)} screening sites, {len(held)} held out; family {len(fam)} features, cut-off {cut:.4f}")
    print(f"held-out sites validated (balanced accuracy >= {detect.BAR}): {res['rate']:.3f}; median accuracy "
          f"{res['acc_median']:.3f}; median AUC {res['auc_median']:.3f}")
    print(f"random families of {len(fam)} features: median {res['null']['median']:.3f}, "
          f"95th percentile {res['null']['p95']:.3f}, p = {res['null']['p']:.3f}")
    # Cross-region check: screen on four regions and score the others.
    regs = sorted({r["region"] for r in rows})
    tr = set(np.random.default_rng(0).choice(regs, 4, replace=False).tolist())
    fx, cx = detect.screen(_pairs([r for r in rows if r["region"] in tr]))
    if fx:
        vx = detect.score(_sites([r for r in rows if r["region"] not in tr]), fx, cx)
        xr = float(np.mean([x["ok"] for x in vx]))
        print(f"cross-region (screened on {sorted(tr)}): {xr:.3f} of {len(vx)} sites (median accuracy "
              f"{np.median([x['acc'] for x in vx]):.3f}, median AUC {np.median([x['auc'] for x in vx]):.3f})")
        res["cross_region"] = dict(screen_regions=sorted(tr), family=fx, cutoff=cx, rate=xr, n=len(vx))
    else:
        print(f"cross-region (screened on {sorted(tr)}): no family")
        res["cross_region"] = dict(screen_regions=sorted(tr), family=[], cutoff=None, rate=None, n=0)
    json.dump(res, open(RESULTS_DIR / "mines.json", "w"), indent=1, default=float)
    print("saved results/mines.json")


if __name__ == "__main__":
    {"select": phase_select, "fetch": phase_fetch, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
