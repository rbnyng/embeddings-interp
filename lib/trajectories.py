"""Yearly mean activations over pixel groups (burned and unburned, or any others), and how many fires a feature
peaks the year after."""
import numpy as np

from lib.config import FIRES, dnbr_path
from lib.tiles import load_tile
from lib.groundtruth import burn_masks
from lib.sae import group_means

def fire_dnbr(fire):
    return np.load(dnbr_path(fire))

def fire_tables(sae, fires=None, stride=2, row_chunk=128, loader=None):
    """Per fire: (fire_year, years, {year: (M,) mean over burned pixels}, {year: (M,) mean over unburned pixels})."""
    loader = loader or load_tile
    out = {}
    for name, spec in (fires or FIRES).items():
        dnbr = fire_dnbr(name)
        mb, mu = {}, {}
        for y in spec["years"]:
            x = loader(spec["tile"], y, stride=stride)
            H, W, _ = x.shape
            b, u = burn_masks(dnbr[::stride, ::stride][:H, :W])
            m = group_means(sae, x, {"burn": b, "unburn": u}, chunk=row_chunk)
            mb[y], mu[y] = m["burn"], m["unburn"]
        out[name] = (spec["fire_year"], spec["years"], mb, mu)
    return out

def year_after_peaks(tables, feat):
    """For one feature: how many fires it peaks the year after, and {fire: peak_year}."""
    ok, peaks = 0, {}
    for fire, (fy, years, mb, mu) in tables.items():
        cs = {y: mb[y][feat] - mu[y][feat] for y in years}
        peak = max(cs, key=cs.get)
        peaks[fire] = int(peak)
        ok += int(peak == fy + 1)
    return ok, peaks

def group_tables(sae, tile, years, groups, stride=2, row_chunk=192, loader=None):
    """Mean activation per year and group, as {year: {group: (M,)}}.

    groups maps names to (H, W) bool masks on the full-resolution tile grid.
    """
    loader = loader or load_tile
    gs = {k: v[::stride, ::stride] for k, v in groups.items()}
    return {y: group_means(sae, loader(tile, y, stride=stride), gs, chunk=row_chunk) for y in years}
