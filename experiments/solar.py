"""Solar farms: find and test a family that tells farms apart from the ground around them. Classified, not dated.

Reference product: USPVDB v4 farm outlines. We use ground-mounted farms online by 2023 and read Tessera for 2024. Each
unit is a 0.1 degree tile with at least 25 ha of those farms. Tiles are taken largest first, at most 15 per state and
300 in all. Pixel groups are at stride 2 and need at least 300 and 2,000 pixels (up to 3,000 are sampled per group):
  panel       inside a farm that counts, shrunk by one pixel to leave out edge pixels mixed with outside ground
  background  more than 100 m from any mapped farm of any year, and within 2 km of one
One dictionary is trained on these tiles. The family is found on a random fifth of the tiles (seed 0), using the crop
atlas's anchored bar on the median gap between panel and background. A pixel is called panel when the family's mean
activation is above a cut-off set on the screening tiles. A held-out tile validates when its balanced accuracy is at
least 0.70. Also reported: each tile's AUC, the pass rate of 200 random families of the same size, and a cross-region
check that screens on two random census regions and tests on the rest.

Usage: python experiments/solar.py select|fetch|train|screen|verdicts
Writes results/solar_farm_selection.json, solar.jsonl and solar.json, the sampled pixels in data/intermediate/solar/,
and caches the dictionary at data/checkpoints/solar/sae_solar.pt.
"""
import json
import sys
import zlib

import numpy as np

from lib.config import CKPT_DIR, DATA_DIR, GT_DIR, RESULTS_DIR
from lib import detect
from lib.tiles import available_years, load_tile, tile_bbox, tile_for_point, tile_grid, transient_tile
from lib.sae import encode_array, load_sae, train_and_save
from lib.sampling import corpus, sample_groups

S, M = 2, 1024
YEAR, BUILT = 2024, 2023
MIN_HA, PER_STATE, N_TILES = 25.0, 15, 300
CAP, N_TRAIN = 3000, 10000
PIX = DATA_DIR / "intermediate" / "solar"
CKPT = CKPT_DIR / "solar" / "sae_solar.pt"
SEL_PATH = RESULTS_DIR / "solar_farm_selection.json"
JSONL = RESULTS_DIR / "solar.jsonl"
SEL = json.load(open(SEL_PATH)) if SEL_PATH.exists() else []
REGION = {**dict.fromkeys("CT ME MA NH RI VT NJ NY PA".split(), "Northeast"),
          **dict.fromkeys("IL IN MI OH WI IA KS MN MO NE ND SD".split(), "Midwest"),
          **dict.fromkeys("DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX".split(), "South"),
          **dict.fromkeys("AZ CO ID MT NV NM UT WY AK CA HI OR WA".split(), "West")}


def _farms():
    """Load the USPVDB farms from the latest release on disk, with each farm's tile and whether it counts.

    A farm counts when it is ground-mounted and online by BUILT.
    """
    import geopandas as gpd
    g = gpd.read_file(max((GT_DIR / "solar").glob("uspvdb_*.geojson")))
    g = g[g.geometry.notna()]
    g["tile"] = [tile_for_point(x, y) for x, y in zip(g.xlong, g.ylat)]
    g["admit"] = (g.p_sys_type == "ground") & (g.p_year <= BUILT)
    return g


def phase_select():
    g = _farms()
    a = g[g.admit]
    ha = (a.to_crs(5070).area / 1e4).groupby(a.tile).sum().sort_values(ascending=False)
    state = a.groupby("tile").p_state.agg(lambda s: s.value_counts().index[0])
    big = [t for t in ha.index if ha[t] >= MIN_HA]
    have = {t for t, ys in available_years(big, [YEAR]).items() if YEAR in ys}
    per, keep = {}, []
    for t in big:
        if t in have and per.get(state[t], 0) < PER_STATE:
            keep.append(t); per[state[t]] = per.get(state[t], 0) + 1
        if len(keep) == N_TILES:
            break
    screen = set(np.random.default_rng(0).choice(sorted(keep), len(keep) // 5, replace=False).tolist())
    out = [dict(tile=t, state=state[t], region=REGION.get(state[t], "other"), ha=round(float(ha[t]), 1),
                screen=t in screen) for t in keep]
    json.dump(out, open(SEL_PATH, "w"), indent=1)
    print(f"{len(big)} tiles >= {MIN_HA} ha of farms online by {BUILT}, {len(have)} with Tessera {YEAR}; kept {len(keep)} "
          f"from {len(per)} states ({len(screen)} screening); by region "
          f"{ {r: sum(x['region'] == r for x in out) for r in sorted(set(REGION.values()))} }", flush=True)


def _masks(tile, farms):
    """Return the panel and background masks on the stride-S grid."""
    from rasterio.features import rasterize
    from scipy.ndimage import binary_erosion, distance_transform_edt
    t, crs, H, W = tile_grid(tile)
    b = tile_bbox(tile, 0.03)
    f = farms.cx[b[0]:b[2], b[1]:b[3]].to_crs(crs)
    ras = lambda gs: (rasterize([(x, 1) for x in gs], out_shape=(H, W), transform=t, fill=0, dtype="uint8").astype(bool)
                      if len(gs) else np.zeros((H, W), bool))[::S, ::S]
    farm, anyfarm = ras(list(f[f.admit].geometry)), ras(list(f.geometry))
    d = distance_transform_edt(~anyfarm) * 10.0 * S
    return binary_erosion(farm), (d > 100) & (d <= 2000)


def fetch_one(args):
    rec, farms = args
    out = PIX / f"{rec['tile']}.npz"
    if out.exists():
        return rec["tile"], "cached"
    try:
        panel, bg = _masks(rec["tile"], farms)
        with transient_tile(rec["tile"], YEAR):
            x = load_tile(rec["tile"], YEAR, stride=S)
        rng = np.random.default_rng(zlib.crc32(rec["tile"].encode()))
        groups, train, counts = sample_groups(x, {"panel": panel, "background": bg}, CAP, N_TRAIN, rng)
        PIX.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **groups, n=np.array([counts["panel"], counts["background"]]), _train=train)
        return rec["tile"], ""
    except Exception as e:
        return rec["tile"], repr(e)


def phase_fetch():
    from concurrent.futures import ThreadPoolExecutor
    farms = _farms()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch_one, [(r, farms) for r in SEL]))
    bad = [(t, e) for t, e in res if e and e != "cached"]
    print(f"{len(res) - len(bad)}/{len(res)} tiles sampled; failed: {bad[:10]}", flush=True)


def phase_train():
    if CKPT.exists():
        print("cached"); return
    X = corpus([PIX / f"{r['tile']}.npz" for r in SEL])
    print(f"solar corpus {X.shape}", flush=True)
    train_and_save(X, CKPT)


def phase_screen():
    """Encode each tile's stored pixels.

    Keeps the mean activation of each group and the activations themselves (sparse, k per pixel).
    """
    import scipy.sparse as sp
    import torch
    sae = load_sae(path=CKPT)
    torch.set_num_threads(16)
    with open(JSONL, "w") as out:
        for r in SEL:
            f = PIX / f"{r['tile']}.npz"
            if not f.exists():
                out.write(json.dumps(dict(ok=False, tile=r["tile"], error="not sampled")) + "\n")
                continue
            z = np.load(f)
            n = dict(zip(("panel", "background"), z["n"].tolist()))
            if n["panel"] < 300 or n["background"] < 2000:
                out.write(json.dumps(dict(ok=False, tile=r["tile"], n=n, error="below the pixel floors")) + "\n")
                continue
            means = {}
            for k in ("panel", "background"):
                F = encode_array(sae, z[k].astype(np.float32))
                means[k] = F.mean(0).tolist()
                sp.save_npz(PIX / f"{r['tile']}_{k}_act.npz", sp.csr_matrix(F.astype(np.float32)))
            out.write(json.dumps(dict(ok=True, **r, n=n, means=means)) + "\n")
    verdicts()


def _pairs(rows):
    return [(np.array(r["means"]["panel"]), np.array(r["means"]["background"])) for r in rows]


def _sites(rows):
    """Return each tile's record with the sparse activations of its panel pixels and its background pixels."""
    import scipy.sparse as sp
    return [(dict(tile=r["tile"], state=r["state"], region=r["region"]),
             *(sp.load_npz(PIX / f"{r['tile']}_{k}_act.npz") for k in ("panel", "background"))) for r in rows]


def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    scr, held = [r for r in rows if r["screen"]], [r for r in rows if not r["screen"]]
    pairs = _pairs(scr)
    fam, cut = detect.screen(pairs)
    if not fam:
        print(f"{len(scr)} screening tiles: no family passes the bar")
        json.dump(dict(family=[], cutoff=None, n_screen=len(scr)), open(RESULTS_DIR / "solar.json", "w"), indent=1)
        return
    sites = _sites(held)
    v = detect.score(sites, fam, cut)
    res = dict(family=fam, cutoff=cut, n_screen=len(scr),
               **detect.summary(v, detect.null_rates(pairs, sites, len(fam), M, "Solar farms")), outcomes=v)
    print(f"{len(scr)} screening tiles, {len(held)} held out; family {len(fam)} features, cut-off {cut:.4f}")
    print(f"held-out tiles validated (balanced accuracy >= {detect.BAR}): {res['rate']:.3f}; median accuracy "
          f"{res['acc_median']:.3f}; median AUC {res['auc_median']:.3f}")
    print(f"random families of {len(fam)} features: median {res['null']['median']:.3f}, "
          f"95th percentile {res['null']['p95']:.3f}, p = {res['null']['p']:.3f}")
    for reg in sorted({x["region"] for x in v}):
        vr = [x for x in v if x["region"] == reg]
        print(f"  {reg}: {np.mean([x['ok'] for x in vr]):.3f} of {len(vr)}")
    regs = sorted({r["region"] for r in rows})
    tr = set(np.random.default_rng(0).choice(regs, 2, replace=False).tolist())
    fx, cx = detect.screen(_pairs([r for r in rows if r["region"] in tr]))
    if fx:
        vx = detect.score(_sites([r for r in rows if r["region"] not in tr]), fx, cx)
        xr = float(np.mean([x["ok"] for x in vx]))
        print(f"cross-region (screened on {sorted(tr)}, family {len(fx)}): {xr:.3f} of {len(vx)} tiles (median accuracy "
              f"{np.median([x['acc'] for x in vx]):.3f}, median AUC {np.median([x['auc'] for x in vx]):.3f})")
        res["cross_region"] = dict(screen_regions=sorted(tr), family=fx, cutoff=cx, rate=xr, n=len(vx))
    else:
        print(f"cross-region (screened on {sorted(tr)}): no family")
        res["cross_region"] = dict(screen_regions=sorted(tr), family=[], cutoff=None, rate=None, n=0)
    json.dump(res, open(RESULTS_DIR / "solar.json", "w"), indent=1, default=float)
    print("saved results/solar.json")


if __name__ == "__main__":
    {"select": phase_select, "fetch": phase_fetch, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
