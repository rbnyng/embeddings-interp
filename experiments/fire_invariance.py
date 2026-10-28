"""Invariance test for fire: compare Californian and European burned pixels in raw embedding space and in the western
US fire family (BURN_FAMILY, in the Dixie dictionary).

Units are the five western US fires in FIRES and the EFDA fire tile-years of the tiles in
results/europe_fire_dixie.jsonl. Each unit gives up to 3,000 burned and 3,000 intact pixels at stride 2, from the fire
year Y or Y+1, whichever has the higher family mean on burned pixels.
  burned  California: dNBR above 0.27. Europe: the Europe atlas's fire_Y group.
  intact  California: dNBR below 0.1 (there is no forest mask). Europe: forest with no EFDA disturbance in 1985-2023.
analyze reports the share of burned and intact pixels on which the family fires, the share of raw variance among
family-active burned pixels explained by location (fire or tile, 100 and 300 pixels each), their raw variance against
intact pixels, and raw distances between pairs of the four group centers.

Usage: python experiments/fire_invariance.py sample|analyze   (sample also runs analyze)
Writes results/fire_invariance.npz and results/fire_invariance.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import FIRES, BURN_FAMILY, RESULTS_DIR, dnbr_path, workers
from lib.groundtruth import burn_masks
from lib.sae import load_sae, encode_array
from lib.stats import group_r2
from lib.tiles import load_tile
import experiments.europe_atlas as ea

S = 2
N_PER = 3000
N_LOC = (100, 300)
NPZ = RESULTS_DIR / "fire_invariance.npz"
OUT = RESULTS_DIR / "fire_invariance.json"
FAM = np.array(BURN_FAMILY)

def _sample(mask, n, rng):
    idx = np.flatnonzero(mask)
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)

def _fire_unit(sae, tile, Y, burn, intact, rng):
    """Sample burned and intact pixels in Y or Y+1, whichever has the higher family mean on burned pixels.

    Returns (year, burned raw, burned encoded, intact raw, intact encoded).
    """
    best = None
    for y in (Y, Y + 1):
        x = load_tile(tile, y, stride=S)
        H, W, _ = x.shape
        b, u = burn[:H, :W], intact[:H, :W]
        ib, iu = _sample(b, N_PER, rng), _sample(u, N_PER, rng)
        X = x.reshape(-1, x.shape[-1])
        xb, xu = X[ib], X[iu]
        fb, fu = encode_array(sae, xb), encode_array(sae, xu)
        score = fb[:, FAM].mean()
        if best is None or score > best[0]:
            best = (score, y, xb, fb, xu, fu)
    return best[1:]

def run_ca(fire):
    torch.set_num_threads(4)
    spec = FIRES[fire]
    rng = np.random.default_rng(0)
    sae = load_sae()
    d = np.load(dnbr_path(fire))[::S, ::S]
    b, _ = burn_masks(d)
    u = ~np.isnan(d) & (d < 0.1)        # clearly unburned. There is no forest mask in CA
    y, xb, fb, xu, fu = _fire_unit(sae, spec["tile"], spec["fire_year"], b, u, rng)
    return [dict(region="CA", unit=fire, year=y, xb=xb, fb=fb, xu=xu, fu=fu)]

def run_eu(tile):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        rng = np.random.default_rng(0)
        sae = load_sae()
        g = ea.tile_groups(tile)
        country = ",".join(sorted({m["country"] for m in ea.TILE_META[tile]}))
        out = []
        for k, m in g.items():
            if not k.startswith("fire_"):
                continue
            Y = int(k.split("_")[1])
            y, xb, fb, xu, fu = _fire_unit(sae, tile, Y, m[::S, ::S], g["intact"][::S, ::S], rng)
            out.append(dict(region="EU", unit=f"{tile}:{country}:{Y}", year=y,
                            xb=xb, fb=fb, xu=xu, fu=fu))
        return out
    except Exception as e:
        print(tile, repr(e), flush=True)
        return []

def phase_sample():
    tiles = [r["tile"] for r in map(json.loads, open(RESULTS_DIR / "europe_fire_dixie.jsonl"))
             if r.get("ok")]
    print(f"{len(FIRES)} CA fires, {len(tiles)} EU tiles", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers(len(FIRES))) as ex:
        for r in ex.map(run_ca, FIRES):
            rows += r
    with ProcessPoolExecutor(max_workers=workers(64)) as ex:
        for i, r in enumerate(ex.map(run_eu, tiles), 1):
            rows += r
            if i % 20 == 0:
                print(f"  [{i}/{len(tiles)}]", flush=True)
    cat = lambda k: np.concatenate([r[k] for r in rows])
    rep = lambda k: np.concatenate([[r[k]] * len(r["xb"]) for r in rows])
    repu = lambda k: np.concatenate([[r[k]] * len(r["xu"]) for r in rows])
    np.savez_compressed(NPZ, xb=cat("xb").astype(np.float32), fb=cat("fb").astype(np.float32),
                        xu=cat("xu").astype(np.float32), fu=cat("fu").astype(np.float32),
                        region_b=rep("region"), unit_b=rep("unit"),
                        region_u=repu("region"), unit_u=repu("unit"))
    print(f"saved {len(rows)} fire units", flush=True)
    analyze()

def analyze():
    z = np.load(NPZ)
    rng = np.random.default_rng(0)
    xb, fb, reg, xu, regu = z["xb"], z["fb"], z["region_b"], z["xu"], z["region_u"]
    loc = np.array([u.split(":")[0] for u in z["unit_b"]])
    on = fb[:, FAM].sum(1) > 0
    on_u = z["fu"][:, FAM].sum(1) > 0
    out = dict(n_burn={g: int((reg == g).sum()) for g in ("CA", "EU")},
               family_fires={f"{g}_{k}": float(m[r == g].mean()) for g in ("CA", "EU")
                             for k, m, r in (("burned", on, reg), ("intact", on_u, regu))})
    print(f"burned pixels: CA {out['n_burn']['CA']}  EU {out['n_burn']['EU']}")
    print("family fires on:", "  ".join(f"{k} {v:.3f}" for k, v in out["family_fires"].items()))

    out["location_r2_raw"] = {}
    for n in N_LOC:
        idx = np.concatenate([rng.choice(ii, n, replace=False) for t in np.unique(loc[on])
                              if len(ii := np.flatnonzero(on & (loc == t))) >= n])
        v = group_r2(xb[idx], loc[idx])
        out["location_r2_raw"][n] = dict(r2=v, n_loc=int(len(np.unique(loc[idx]))),
                                         n_ca=int(len(np.unique(loc[idx][reg[idx] == "CA"]))))
        print(f"location R^2, raw, family-active burned ({n}/location, "
              f"{out['location_r2_raw'][n]['n_loc']} locations): {v:.3f}")
        if n == N_LOC[0]:
            act_idx = idx

    lu = np.array([u.split(":")[0] for u in z["unit_u"]])
    iu = np.concatenate([rng.choice(ii, N_LOC[0], replace=False) for t in np.unique(lu)
                         if len(ii := np.flatnonzero(lu == t)) >= N_LOC[0]])
    tv = lambda X: float(((X - X.mean(0)) ** 2).sum(1).mean())
    # intact_all_locations stores a ratio: family-active burned variance over intact variance
    out["raw_spread"] = sp = dict(family_active_burned=tv(xb[act_idx]), intact_all=tv(xu[iu]),
                                  intact_all_locations=tv(xb[act_idx]) / tv(xu[iu]))
    print(f"raw total variance: family-active burned {sp['family_active_burned']:.1f}, "
          f"intact forest {sp['intact_all']:.1f}  (ratio {sp['intact_all_locations']:.2f})")

    # norm is the distance between centers divided by the pooled RMS spread. A value near 1 means the
    # centers are one typical pixel-to-center distance apart.
    G = dict(ca_burn=xb[reg == "CA"], eu_burn=xb[reg == "EU"],
             ca_intact=xu[regu == "CA"], eu_intact=xu[regu == "EU"])
    pairs = [("ca_burn", "eu_burn"), ("eu_burn", "eu_intact"),
             ("ca_intact", "eu_intact"), ("ca_burn", "ca_intact")]
    out["raw_centroid_dist"] = dist = {}
    print(f"raw centroid distances  {'abs':>6s} {'norm':>6s}")
    for a, b in pairs:
        dd = float(np.linalg.norm(G[a].mean(0) - G[b].mean(0)))
        nd = dd / np.sqrt((tv(G[a]) + tv(G[b])) / 2)
        dist[f"{a}__{b}"] = dict(abs=dd, norm=float(nd))
        print(f"  {a + ' - ' + b:22s} {dd:6.2f} {nd:6.2f}")
    json.dump(out, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    {"sample": phase_sample, "analyze": analyze}[sys.argv[1]]()
