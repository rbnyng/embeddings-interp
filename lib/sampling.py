"""Sample and delete: keep only the pixels a setting needs from each tile-year, not the whole tiles.

Some settings use hundreds of tiles over many years (landslides, new water, forest loss, mines, solar). For each
tile-year they store up to `cap` pixels of each mask and a uniform training sample. Then they delete the tile-years
they downloaded (lib.tiles.transient_tile). Samples are float16. The random draws are seeded per unit, so sampling
again gives the same pixels.
"""
import numpy as np

from lib.tiles import load_tile, transient_tile


def valid_pixels(x):
    """Pixels that have an embedding: finite and not all zero. A few source tile-years contain inf."""
    return np.isfinite(x).all(-1) & (np.abs(x).sum(-1) > 0)


def take(idx, n, rng):
    """Return idx, or a random n of its entries in sorted order."""
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def sample_groups(x, masks, cap, n_train, rng):
    """Sample one (H, W, D) array: up to `cap` valid pixels per mask, and `n_train` uniform valid pixels.

    Mask pixels stay in mask order. Returns ({mask: (n, D) float16}, (n_train, D) float16, {mask: valid pixel count}).
    """
    H, W, D = x.shape
    valid = valid_pixels(x)
    flat = x.reshape(-1, D)
    groups, counts = {}, {}
    for k, m in masks.items():
        idx = np.flatnonzero((m[:H, :W] & valid).ravel())
        counts[k] = len(idx)
        groups[k] = flat[take(idx, cap, rng)].astype(np.float16)
    train = flat[take(np.flatnonzero(valid.ravel()), n_train, rng)].astype(np.float16)
    return groups, train, counts


def sample_tile_years(tile, years, masks, stride, cap, n_train, rng):
    """Sample every tile-year that can be fetched, as {f"{year}_{mask}": samples, f"{year}_train": samples}."""
    store = {}
    for y in years:
        try:
            with transient_tile(tile, y):
                x = load_tile(tile, y, stride=stride)
        except Exception:
            continue
        groups, train, _ = sample_groups(x, masks, cap, n_train, rng)
        store.update({f"{y}_{k}": v for k, v in groups.items()})
        store[f"{y}_train"] = train
        del x
    return store


def finite(a):
    """The rows of a as float32, dropping any row with a non-finite value."""
    a = np.asarray(a, np.float32)
    return a[np.isfinite(a).all(1)]


def corpus(paths, seed=0):
    """Every stored training sample (keys ending in "_train") in the given .npz files, as finite rows, shuffled."""
    X = []
    for f in paths:
        if f.exists():
            z = np.load(f)
            X += [finite(z[k]) for k in z.files if k.endswith("_train")]
    X = np.concatenate(X)
    np.random.default_rng(seed).shuffle(X)
    return X


def mean_activation(sae, a, floor):
    """Mean dictionary activation over the finite rows of a, and their count. The mean is None under `floor` rows."""
    from lib.sae import encode_array
    a = finite(a)
    return (encode_array(sae, a).mean(0) if len(a) >= floor else None), len(a)
