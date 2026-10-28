"""Check that the burn family (the western US fire family) does not depend on the Dixie dictionary's training seed.

Five Dixie dictionaries are trained with the Dixie dictionary's recipe and seeds 0-4. The seed sets both the pixel
sample and the starting weights. Each family is screened like the burn family: anchored_family on the yearly
burned-minus-unburned contrast on Dixie at full resolution, with expected window Y+1. Each family is scored on the
held-out western US fires, with western_us_fire's fire masks (at least 300 burned and 300 unburned pixels), and
compared with 200 random same-size families from its own dictionary. The Dixie dictionary's burn family is scored the
same way. analyze also compares decoder directions with seed 0 and checks that seed 0 reproduces lib.config.BURN_FAMILY.

Usage: python experiments/dixie_seeds.py train|stats|perfeat|analyze
Writes per-seed files to data/checkpoints/dixie_seeds and results/dixie_seeds, and results/dixie_seeds.json.
"""
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import (CKPT_DIR, DIXIE, DIXIE_SAE_YEARS, FIRES, BURN_FAMILY, GT_DIR,
                        RESULTS_DIR)
from lib.config import workers
from lib.groundtruth import burn_masks
from lib.sae import encode_array, load_sae, group_sums, train_and_save
from lib.screen import anchored_family, peak_outcome
from lib.tiles import load_tile, sample_pixels

SEEDS = (0, 1, 2, 3, 4)
TILE = FIRES[DIXIE]["tile"]
FY = FIRES[DIXIE]["fire_year"]
OUT = RESULTS_DIR / "dixie_seeds"
CK = CKPT_DIR / "dixie_seeds"
M = 1024
DRAWS = 200


def ckpt(seed):
    return CK / f"sae_s{seed}.pt"


def load(seed):
    sae = load_sae(path=ckpt(seed))
    return sae


def _train(seed):
    if ckpt(seed).exists():
        return seed
    torch.set_num_threads(12)
    X = sample_pixels([(TILE, y) for y in DIXIE_SAE_YEARS], seed=seed)
    train_and_save(X, ckpt(seed), threads=None, epochs=40, seed=seed, log=False)
    return seed


def phase_train():
    with ProcessPoolExecutor(max_workers=workers(len(SEEDS))) as ex:
        for s in ex.map(_train, SEEDS):
            print("trained seed", s, flush=True)


def _stats(seed):
    """One seed's yearly burned-minus-unburned contrast on Dixie at full resolution, and its family."""
    torch.set_num_threads(8)
    sae = load(seed)
    dnbr = np.load(GT_DIR / f"dnbr_{DIXIE}.npy")
    mb_y, mu_y = {}, {}
    for y in DIXIE_SAE_YEARS:
        x = load_tile(TILE, y)
        H, W, _ = x.shape
        b, u = burn_masks(dnbr[:H, :W])
        mb = np.zeros(M); mu = np.zeros(M); nb = nu = 0
        for r0 in range(0, H, 64):
            r1 = min(r0 + 64, H)
            f = encode_array(sae, x[r0:r1])
            mb += f[b[r0:r1]].sum(0); nb += b[r0:r1].sum()
            mu += f[u[r0:r1]].sum(0); nu += u[r0:r1].sum()
        mb_y[y], mu_y[y] = mb / max(nb, 1), mu / max(nu, 1)
    traj = {y - FY: mb_y[y] - mu_y[y] for y in DIXIE_SAE_YEARS}
    fam = sorted(anchored_family(traj, window=(1,)))
    return seed, fam, {str(k): v.tolist() for k, v in traj.items()}


def phase_stats():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers(len(SEEDS))) as ex:
        for seed, fam, traj in ex.map(_stats, SEEDS):
            json.dump(dict(seed=seed, family=fam, traj=traj),
                      open(OUT / f"stats_s{seed}.json", "w"))
            print(f"seed {seed}: family {len(fam)}", flush=True)


def _perfeat(args):
    """For one fire and one seed, each feature's burned-minus-unburned contrast in each year 2017-2025."""
    rec, seed = args
    import experiments.western_us_fire as ma
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae = load(seed)
        b, u = ma.fire_mask(rec)
        con = {}
        for y in range(2017, 2026):
            try:
                x = load_tile(rec["tile"], y, stride=ma.S)
            except Exception:
                continue
            H, W, _ = x.shape
            bb, uu = b[:H, :W], u[:H, :W]
            if bb.sum() < 300 or uu.sum() < 300:
                return dict(ok=False, event_id=rec["event_id"], error="mask")
            s, n = group_sums(sae, x, {"burn": bb, "unburn": uu})
            con[str(y)] = np.round(s["burn"] / n["burn"] - s["unburn"] / n["unburn"], 5).tolist()
        if not {str(rec["year"]), str(rec["year"] + 1)} <= set(con):
            return dict(ok=False, event_id=rec["event_id"], error="gap")
        return dict(ok=True, event_id=rec["event_id"], year=rec["year"], con=con)
    except Exception as e:
        return dict(ok=False, event_id=rec["event_id"], error=repr(e))


def phase_perfeat():
    import experiments.western_us_fire as ma
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        path = OUT / f"perfeat_s{seed}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["event_id"] for l in open(path)}
        todo = [(r, seed) for r in ma.FIRES if r["event_id"] not in done]
        print(f"seed {seed}: {len(todo)} fires", flush=True)
        with open(path, "a") as f, ProcessPoolExecutor(max_workers=workers(96)) as ex:
            for r in ex.map(_perfeat, todo):
                f.write(json.dumps(r) + "\n")


def _rate(units, fam):
    v = peak_outcome(units, list(fam), (1,))
    return float(np.mean([ok for _, ok in v.values()]))


def phase_analyze():
    from lib import measure, settings as S
    _, setting = S.SETTINGS["Western US fire"]
    held = [u.uid for u in measure.held_out(measure.load(setting.unit_key), setting)]
    dec0 = load(0).W_d.data.numpy()
    st = {s: json.load(open(OUT / f"stats_s{s}.json")) for s in SEEDS}
    fam0 = st[0]["family"]
    out = dict(n_held=len(held), seeds={})
    for s in SEEDS:
        fam = st[s]["family"]
        rows = {r["event_id"]: r for r in map(json.loads, open(OUT / f"perfeat_s{s}.jsonl"))
                if r.get("ok")}
        miss = [e for e in held if e not in rows]
        units = {e: {int(y) - rows[e]["year"]: np.asarray(c) for y, c in rows[e]["con"].items()}
                 for e in held if e in rows}
        obs = _rate(units, fam)
        rg = np.random.default_rng(zlib.crc32(f"dixie_seed{s}".encode()))
        null = np.array([_rate(units, rg.choice(M, len(fam), replace=False))
                         for _ in range(DRAWS)])
        # match each feature to the nearest seed-0 family feature by decoder direction
        dec = load(s).W_d.data.numpy()
        a = dec[:, fam] / np.linalg.norm(dec[:, fam], axis=0)
        b = dec0[:, fam0] / np.linalg.norm(dec0[:, fam0], axis=0)
        best = (a.T @ b).max(1)
        fa, fb = a.mean(1), b.mean(1)
        out["seeds"][str(s)] = dict(
            family_size=len(fam), n_scored=len(units), missing=len(miss), rate=round(obs, 4),
            null_median=round(float(np.median(null)), 4),
            null_p95=round(float(np.percentile(null, 95)), 4),
            p=round(float((null >= obs).mean()), 4),
            match_cos_median=round(float(np.median(best)), 3),
            match_share_cos_0_9=round(float((best >= 0.9).mean()), 3),
            family_direction_cos=round(float(fa @ fb / np.linalg.norm(fa) / np.linalg.norm(fb)), 3))
        print(s, out["seeds"][str(s)], flush=True)
    # score the Dixie dictionary (load_sae()) and the burn family from their stored per-feature gaps
    pub_rows = {r["event_id"]: r for r in map(json.loads, open(RESULTS_DIR / "western_us_fire_perfeat.jsonl"))
                if r.get("ok")}
    units = {e: {int(y) - pub_rows[e]["year"]: np.asarray(c) for y, c in pub_rows[e]["con"].items()}
             for e in held if e in pub_rows}
    obs = _rate(units, BURN_FAMILY)
    rg = np.random.default_rng(2411673976)          # fixed seed
    null = np.array([_rate(units, rg.choice(M, len(BURN_FAMILY), replace=False))
                     for _ in range(DRAWS)])
    pubdec = load_sae().W_d.data.numpy()
    a = pubdec[:, BURN_FAMILY] / np.linalg.norm(pubdec[:, BURN_FAMILY], axis=0)
    b = dec0[:, fam0] / np.linalg.norm(dec0[:, fam0], axis=0)
    fa, fb = a.mean(1), b.mean(1)
    out["dixie"] = dict(
        family_size=len(BURN_FAMILY), n_scored=len(units), rate=round(obs, 4),
        null_median=round(float(np.median(null)), 4),
        null_p95=round(float(np.percentile(null, 95)), 4), p=round(float((null >= obs).mean()), 4),
        match_cos_median_vs_seed0=round(float(np.median((a.T @ b).max(1))), 3),
        family_direction_cos_vs_seed0=round(float(fa @ fb / np.linalg.norm(fa) / np.linalg.norm(fb)), 3))
    print("Dixie dictionary", out["dixie"], flush=True)
    out["seed0_reproduces_burn_family"] = sorted(fam0) == sorted(BURN_FAMILY)
    out["seed0_max_abs_weight_diff_vs_dixie"] = float(np.abs(pubdec - dec0).max())
    json.dump(out, open(RESULTS_DIR / "dixie_seeds.json", "w"), indent=1)
    print("seed 0 reproduces the burn family:", out["seed0_reproduces_burn_family"],
          "| max decoder weight difference:", out["seed0_max_abs_weight_diff_vs_dixie"])


if __name__ == "__main__":
    {"train": phase_train, "stats": phase_stats, "perfeat": phase_perfeat,
     "analyze": phase_analyze}[sys.argv[1]]()
