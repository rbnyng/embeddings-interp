"""Count verified burn features in AlphaEarth (64-d) Dixie dictionaries as dictionary size grows.

Sizes m are 256 to 4096, or 4 to 64 times the 64 dimensions (the same expansions as burn_capacity.py on Tessera),
with k = 8 and seeds 1-5. Each dictionary is trained on 100k pixels per year of the Dixie tile, 2019-2024, scaled to
Tessera's range. It is screened on Dixie at stride 4, on the burned-minus-unburned contrast, at two bars: a fixed bar
of 0.2 and an anchored bar of ANCHOR times the median absolute Y+1 contrast over active features. A feature passes
when its contrast is above the bar at Y+1, stays under half the bar in the two earliest pre-fire years, and its burned
activation at Y+2 is under half its Y+1 value. It is verified when it peaks at Y+1 on all five western US fires.

Usage: python experiments/alphaearth_capacity.py fetch|sweep
Writes the dictionaries to data/checkpoints/aef_sweep and results/alphaearth_capacity.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

from lib.config import (FIRES, DIXIE, DIXIE_SAE_YEARS, CKPT_DIR,
                                   RESULTS_DIR)
from lib.config import workers
from lib.alphaearth import compute_scale, fetch_aef, load_aef, dequantize
from lib.trajectories import fire_tables, year_after_peaks
from lib.sae import load_sae, train_and_save
from lib.screen import ANCHOR

MS = [256, 512, 1024, 2048, 4096]        # 4 to 64 times the 64 dimensions
K = 8
SEEDS = range(1, 6)
STRIDE = 4
FIXED_BAR = 0.2
OUT = CKPT_DIR / "aef_sweep"
TILE = FIRES[DIXIE]["tile"]

def phase_fetch():
    jobs = [(s["tile"], y) for n, s in FIRES.items() for y in s["years"]]
    print(f"{len(jobs)} AlphaEarth tile-years", flush=True)
    def one(j):
        t, y = j
        try:
            fetch_aef(t, y)
            return None
        except Exception as e:
            return f"{t} {y}: {e!r}"
    with ThreadPoolExecutor(max_workers=8) as ex:
        errs = [r for r in ex.map(one, jobs) if r]
    print(f"fetch done, {len(errs)} errors", flush=True)
    for e in errs[:10]:
        print(" ", e)

def sample_aef(scale, n_per=100_000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for y in DIXIE_SAE_YEARS:
        raw = fetch_aef(TILE, y)
        H, W, _ = raw.shape
        idx = rng.choice(H * W, n_per, replace=False)
        out.append(dequantize(raw.reshape(-1, 64)[idx]))
    return np.nan_to_num(np.concatenate(out)) * scale

def run_cell(job):
    m, seed, scale = job
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    try:
        p = OUT / f"sae_aef_m{m}_s{seed}.pt"
        if p.exists():
            sae = load_sae(path=p, d=64, m=m, k=K)
        else:
            X = sample_aef(scale, seed=seed)
            sae = train_and_save(X, p, threads=None, epochs=40, seed=seed, log=False, d=64, m=m, k=K)
            sae.eval()
            del X

        loader = lambda t, y, stride=1: np.nan_to_num(
            load_aef(t, y, stride=stride)) * scale
        tables = fire_tables(sae, stride=STRIDE, loader=loader)
        fy, years, mb, mu = tables[DIXIE]
        MB = np.array([mb[y] for y in years])
        MU = np.array([mu[y] for y in years])
        i1 = list(years).index(fy + 1)
        C1 = np.abs(MB[i1] - MU[i1])
        active = (MB.max(0) > 0) | (MU.max(0) > 0)
        med = float(np.median(C1[active]))

        def screen(bar):
            ys = list(years)
            C = MB - MU
            i2 = ys.index(fy + 2) if fy + 2 in ys else None
            pre = [i for i, y in enumerate(ys) if y <= fy - 1][:2]
            passes = (C[i1] > bar) & (np.abs(C[pre]).max(0) < bar / 2) \
                if pre else (C[i1] > bar)
            if i2 is not None:
                passes &= MB[i2] < 0.5 * MB[i1]
            fam = np.where(passes)[0]
            core = sum(int(year_after_peaks(tables, int(f))[0] == 5) for f in fam)
            return int(fam.size), int(core)

        f_fam, f_core = screen(FIXED_BAR)
        a_fam, a_core = screen(ANCHOR * med)
        return dict(ok=True, m=m, seed=seed, median_abs_C1=med,
                    n_active=int(active.sum()),
                    fixed=dict(family=f_fam, core=f_core),
                    anchored=dict(family=a_fam, core=a_core))
    except Exception as e:
        return dict(ok=False, m=m, seed=seed, error=repr(e))

def phase_sweep():
    scale = compute_scale()
    print(f"SCALE = x{scale:.1f}", flush=True)
    jobs = [(m, s, scale) for m in MS for s in SEEDS]
    jobs.sort(key=lambda j: -j[0])
    rows = []
    with ProcessPoolExecutor(max_workers=workers(25)) as ex:
        for i, r in enumerate(ex.map(run_cell, jobs), 1):
            rows.append(r)
            if r["ok"]:
                print(f"[{i:2d}/{len(jobs)}] m={r['m']:5d} s{r['seed']}  "
                      f"fixed core={r['fixed']['core']:3d}  "
                      f"anchored core={r['anchored']['core']:3d}  "
                      f"med|C1|={r['median_abs_C1']:.5f}", flush=True)
            else:
                print(f"[{i:2d}/{len(jobs)}] m={r['m']} s{r['seed']} FAILED "
                      f"{r['error'][:70]}", flush=True)

    good = [r for r in rows if r["ok"]]

    def med(m, tag, field):
        v = [r[tag][field] for r in good if r["m"] == m]
        return float(np.median(v)) if v else float("nan")

    print(f"\n{'m':>6} {'expansion':>10} | fixed fam/core | anchored fam/core | med|C1|")
    for m in MS:
        mc = float(np.median([r["median_abs_C1"] for r in good if r["m"] == m]))
        print(f"{m:>6} {m//64:>9}x | {med(m,'fixed','family'):5.0f} "
              f"{med(m,'fixed','core'):4.0f} | {med(m,'anchored','family'):8.0f} "
              f"{med(m,'anchored','core'):4.0f} | {mc:.5f}")

    fc = [med(m, "fixed", "core") for m in MS]
    ac = [med(m, "anchored", "core") for m in MS]
    meds = [float(np.median([r["median_abs_C1"] for r in good if r["m"] == m]))
            for m in MS]
    json.dump(dict(fixed_core=fc, anchored_core=ac, med_contrast=meds, cells=rows),
              open(RESULTS_DIR / "alphaearth_capacity.json", "w"))
    print("\nsaved results/alphaearth_capacity.json")

if __name__ == "__main__":
    {"fetch": phase_fetch, "sweep": phase_sweep}[sys.argv[1]]()
