"""Random-family nulls for every dated setting, and a 95% interval on each observed rate.

Each of 200 draws swaps the family for a random set of the same size from the whole dictionary. It then scores the
same held-out units with the same expected window, using lib.screen.peak_outcome. A second null draws only live
features outside the family. A feature is live if its contrast is nonzero in at least one held-out unit. The interval
on the observed rate is the tile bootstrap, or the Wilson interval when every unit or none is dated. Each setting has
its own fixed seeds (Setting.null_seeds).

Usage: python pipeline/nulls.py
Writes results/all_nulls.json."""
import json

import numpy as np

from lib import measure
from lib import settings as S
from lib.screen import peak_outcome
from lib.config import RESULTS_DIR
from lib.stats import rate, tile_bootstrap, wilson

DRAWS = 200

def summarise(name, obs, null, n, m):
    null = np.asarray(null)
    return dict(observed=round(float(obs), 4), n=int(n), m=int(m),
                null_median=round(float(np.median(null)), 4),
                null_lo=round(float(np.percentile(null, 2.5)), 4),
                null_hi=round(float(np.percentile(null, 97.5)), 4),
                null_p95=round(float(np.percentile(null, 95)), 4),   # one-sided, matches p
                draws=[round(float(x), 4) for x in null],
                p=round(float((null >= obs).mean()), 4))

def observed_interval(name):
    """95% interval on a setting's observed rate. Uses the tile bootstrap, or Wilson when all units or none are
    dated, since the bootstrap interval would then have zero width."""
    recs = [dict(tile=u.tile, ok=bool(ok)) for u, _, ok in measure.outcomes(name)]
    obs, lo, hi = tile_bootstrap(recs, rate)
    if obs in (0.0, 1.0):
        lo, hi = wilson(sum(r["ok"] for r in recs), len(recs))
        return lo, hi, "Wilson"
    return lo, hi, "tile bootstrap"

def compute(name, dictionary, setting):
    """Observed rate (from lib.measure.outcomes) and its null on the same units."""
    units = measure.held_out(measure.load(setting.unit_key), setting)
    size = len(measure.family(name))
    obs = float(np.mean([ok for *_, ok in measure.outcomes(name)]))
    rg = np.random.default_rng(setting.null_seeds[0])
    null = []
    for _ in range(DRAWS):
        fam = rg.choice(dictionary.m, size=size, replace=False)
        v = peak_outcome({u.uid: u.offsets() for u in units}, fam, setting.window)
        null.append(float(np.mean([ok for _, ok in v.values()])))
    r = summarise(name, obs, null, len(units), size)
    lo, hi, method = observed_interval(name)
    r.update(obs_lo=round(lo, 4), obs_hi=round(hi, 4), obs_interval=method)
    r["live"] = live_null(name, setting, units, size)
    return r

def live_null(name, setting, units, size):
    """The same null, but drawing only live features outside the family."""
    stack = np.concatenate([np.stack(list(u.offsets().values())) for u in units])
    live = np.abs(stack).max(0) > 0
    fam = set(measure.family(name))
    pool = np.array([f for f in np.where(live)[0] if f not in fam])
    rg = np.random.default_rng(setting.null_seeds[1])
    null = []
    for _ in range(DRAWS):
        v = peak_outcome({u.uid: u.offsets() for u in units},
                         rg.choice(pool, size=size, replace=False), setting.window)
        null.append(float(np.mean([ok for _, ok in v.values()])))
    return dict(null_median=round(float(np.median(null)), 4),
                dead_share=round(float(1 - live.mean()), 4))

def main():
    out, skipped = {}, {}
    for name, (dictionary, setting) in S.SETTINGS.items():
        if not setting.null:
            skipped[name] = "scored by held-out classification, not a peak offset"
            continue
        r = compute(name, dictionary, setting)
        out[name] = r
        print(f"  {name:<24} obs {r['observed']:.3f} [{r['obs_lo']:.3f}, {r['obs_hi']:.3f}]  null {r['null_median']:.3f} "
              f"[{r['null_lo']:.3f}, {r['null_hi']:.3f}]  n={r['n']}  "
              f"|fam|={r['m']}  P={r['p']:.3f}  live null {r['live']['null_median']:.3f} "
              f"(dead {r['live']['dead_share']:.1%})")

    out["_not_computable"] = skipped
    json.dump(out, open(RESULTS_DIR / "all_nulls.json", "w"), indent=1)
    print(f"\n{len(out) - 1} settings with a null, {len(skipped)} without: "
          f"{list(skipped)}")

if __name__ == "__main__":
    main()
