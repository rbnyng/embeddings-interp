"""Confidence intervals on validation rates: a bootstrap over tiles, and the Wilson score interval.

Units on the same tile are not independent, so the bootstrap resamples tiles rather than units (10,000 draws, seed 0,
95% percentile interval)."""
import numpy as np

DRAWS = 10_000
SEED = 0

def tile_bootstrap(records, stat, draws=DRAWS, seed=SEED, key="tile"):
    """Return (observed, lo, hi) from a 95% percentile bootstrap over clusters, the records that share `key`.

    Draws where `stat` returns NaN or None are dropped. That happens when a resample lacks a group it needs.
    """
    by = {}
    for r in records:
        by.setdefault(r[key], []).append(r)
    clusters = list(by)
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    obs = stat(records)
    reps = []
    for _ in range(draws):
        pick = rng.choice(len(clusters), size=len(clusters), replace=True)
        pooled = [r for i in pick for r in by[clusters[i]]]
        s = stat(pooled)
        if s is not None and not np.isnan(s):
            reps.append(s)
    return obs, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))

def wilson(k, n, z=1.96):
    """Wilson score interval (lo, hi) for k out of n. Unlike a normal approximation, it stays inside [0, 1]."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

def rate(records, field="ok"):
    """Mean of a boolean field, NaN when empty."""
    return float(np.mean([bool(r[field]) for r in records])) if records else float("nan")

def group_r2(V, lab):
    """Return the share of the total variance of rows V (summed over columns) explained by the means of groups lab."""
    _, g = np.unique(lab, return_inverse=True)
    mu = V.mean(0)
    n = np.bincount(g)
    means = np.stack([np.bincount(g, weights=V[:, j]) for j in range(V.shape[1])], 1) / n[:, None]
    return float((n * ((means - mu) ** 2).sum(1)).sum() / ((V - mu) ** 2).sum())
