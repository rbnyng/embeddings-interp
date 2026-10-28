"""Count verified burn features in Tessera Dixie dictionaries as dictionary size (capacity) grows.

The grid is m = 512, 1024, 2048, 4096 and 8192, k = 8, 16 and 32, and seeds 1 to n_seeds (default 5). Each dictionary
is trained on 100k pixels per year of the Dixie tile, 2019-2024, or loaded if already saved. It is screened on Dixie
at stride 4, on the burned-minus-unburned contrast, at two bars: a fixed bar of 0.2 and an anchored bar of ANCHOR
times the median absolute Y+1 contrast over active features (Y+1 is the year after the fire). A screened feature is
verified when it peaks at Y+1 on all five western US fires.

Usage: python experiments/burn_capacity.py screen [n_workers] [n_seeds]
Writes the dictionaries to data/checkpoints/dictsweep (m up to 4096) and m8192, and results/m8192.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import (FIRES, DIXIE, DIXIE_SAE_YEARS, CKPT_DIR,
                                   RESULTS_DIR)
from lib.config import workers
from lib.tiles import sample_pixels
from lib.trajectories import fire_tables, year_after_peaks
from lib.sae import load_sae, train_and_save
from lib.screen import ANCHOR

def screen(MB, MU, years, fire_year, bar):
    """Return the burn features at contrast bar `bar`.

    A feature qualifies when its contrast is above `bar` at Y+1, its absolute contrast stays under half of `bar` in the
    two earliest pre-fire years, and its burned-pixel activation at Y+2 is less than half its Y+1 value."""
    years = list(years)
    C = MB - MU
    i1 = years.index(fire_year + 1)
    i2 = years.index(fire_year + 2) if fire_year + 2 in years else None
    pre = [i for i, y in enumerate(years) if y <= fire_year - 1][:2]
    passes = (C[i1] > bar) & (np.abs(C[pre]).max(0) < bar / 2)
    if i2 is not None:
        passes &= MB[i2] < 0.5 * MB[i1]
    return np.where(passes)[0]

MS = [512, 1024, 2048, 4096, 8192]
KS = [8, 16, 32]
NEW_M = 8192
FIXED_BAR = 0.2
STRIDE = 4
TILE = FIRES[DIXIE]["tile"]
OLD = CKPT_DIR / "dictsweep"                 # the dictionaries up to m = 4096
NEW = CKPT_DIR / "m8192"
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
N_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5

def get_sae(m, k, seed):
    p = (OLD if m != NEW_M else NEW) / f"sae_m{m}_k{k}_s{seed}.pt"
    if p.exists():
        sae = load_sae(path=p, m=m, k=k)
        return sae
    X = sample_pixels([(TILE, y) for y in DIXIE_SAE_YEARS], seed=seed)
    sae = train_and_save(X, p, threads=None, epochs=40, seed=seed, log=False, m=m, k=k)
    sae.eval()
    del X
    return sae

def run_cell(job):
    m, k, seed = job
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae = get_sae(m, k, seed)
        # Smaller row_chunk at high m. A block of 128 rows by 8192 latents takes about 0.9 GB.
        rc = 128 if m <= 4096 else 48
        tables = fire_tables(sae, stride=STRIDE, row_chunk=rc)
        fy, years, mb, mu = tables[DIXIE]
        MB = np.array([mb[y] for y in years])
        MU = np.array([mu[y] for y in years])
        i1 = list(years).index(fy + 1)
        C1 = np.abs(MB[i1] - MU[i1])
        active = (MB.max(0) > 0) | (MU.max(0) > 0)
        med = float(np.median(C1[active]))
        cache = {}

        def evaluate(bar):
            fam = screen(MB, MU, years, fy, bar)
            n5 = 0
            for f in fam:
                f = int(f)
                if f not in cache:
                    cache[f] = year_after_peaks(tables, f)[0]
                n5 += int(cache[f] == 5)
            return dict(bar=float(bar), family=int(fam.size), core=int(n5))

        return dict(ok=True, m=m, k=k, seed=seed, n_active=int(active.sum()),
                    median_abs_C1=med,
                    fixed=evaluate(FIXED_BAR),
                    anchored=evaluate(ANCHOR * med))
    except Exception as e:
        return dict(ok=False, m=m, k=k, seed=seed, error=repr(e))


def phase_screen():
    """Screen every dictionary in the grid at both bars and write results/m8192.json."""
    jobs = [(m, k, s) for m in MS for k in KS for s in range(1, N_SEEDS + 1)]
    jobs.sort(key=lambda j: -j[0])                 # m = 8192 first, as those jobs are the slowest
    print(f"{len(jobs)} cells ({len(MS)} capacities x {len(KS)} k x {N_SEEDS} seeds); "
          f"{sum(1 for j in jobs if j[0] == NEW_M)} are new at m={NEW_M}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers(N_WORKERS)) as ex:
        for i, r in enumerate(ex.map(run_cell, jobs), 1):
            rows.append(r)
            if r["ok"]:
                print(f"[{i:3d}/{len(jobs)}] m={r['m']:5d} k={r['k']:3d} s{r['seed']}  "
                      f"fixed core={r['fixed']['core']:3d}  "
                      f"anc core={r['anchored']['core']:4d}  "
                      f"med|C1|={r['median_abs_C1']:.6f}", flush=True)
            else:
                print(f"[{i:3d}/{len(jobs)}] m={r['m']} k={r['k']} s{r['seed']} "
                      f"FAILED {r['error']}", flush=True)

    good = [r for r in rows if r["ok"]]

    def med(tag, m, k, field):
        v = [r[tag][field] for r in good if r["m"] == m and r["k"] == k]
        return float(np.median(v)) if v else float("nan")

    print(f"\n{'k':>3} | {'bar':>10} | verified core by m (512..8192)        ratios")
    summary = {}
    for k in KS:
        for tag in ("fixed", "anchored"):
            core = [med(tag, m, k, "core") for m in MS]
            ratios = [core[i + 1] / core[i] if core[i] else float("nan")
                      for i in range(len(MS) - 1)]
            summary[f"k{k}|{tag}"] = dict(core=core, ratios=ratios)
            print(f"{k:>3} | {tag:>10} | {[int(x) for x in core]}   "
                  f"{[round(r, 2) for r in ratios]}")

    json.dump(dict(summary=summary, cells=rows), open(RESULTS_DIR / "m8192.json", "w"))
    print("\nsaved results/m8192.json")

if __name__ == "__main__":
    {"screen": phase_screen}[sys.argv[1]]()
