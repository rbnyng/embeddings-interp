"""Encode the five western US fire tiles with the Dixie dictionary and tabulate burned and unburned activation by year.

Downloads each fire's tile for its years in lib.config.FIRES, and computes its dNBR from Sentinel-2 L2A. For each fire
and year, it records every feature's mean activation over burned (dNBR above 0.27) and unburned pixels, at stride 1.

Usage: python experiments/fire_trajectories.py
Writes results/fire_trajectories.json, with the reconstruction R^2 on each fire's tile the year after the fire.
"""
import json

from lib.config import FIRES, RESULTS_DIR, dnbr_path
from lib.tiles import download_tile, download_landmask, load_tile
from lib.groundtruth import fetch_dnbr
from lib.sae import load_sae, val_r2
from lib.trajectories import fire_tables

for name, spec in FIRES.items():
    download_landmask(spec["tile"])
    for y in spec["years"]:
        download_tile(spec["tile"], y)
    fetch_dnbr(spec["tile"], spec["pre"], spec["post"], out_path=dnbr_path(name))
    print("data ready:", name, flush=True)

sae = load_sae()
tables = fire_tables(sae, stride=1)

out = {}
for name, (fy, years, mb, mu) in tables.items():
    out[name] = dict(fire_year=fy, years=years,
                     mean_burn={y: mb[y].tolist() for y in years},
                     mean_unburn={y: mu[y].tolist() for y in years})

for name, spec in FIRES.items():
    # reconstruction R^2 on the tile from the year after the fire, or the last year if that one is missing
    x1 = load_tile(spec["tile"], spec["fire_year"] + 1
                   if spec["fire_year"] + 1 in spec["years"] else spec["years"][-1],
                   stride=6)
    out[name]["r2_age1"] = val_r2(sae, x1.reshape(-1, 128)[:100000])
    print(f"{name}: reconstruction R^2 {out[name]['r2_age1']:.3f}", flush=True)

json.dump(out, open(RESULTS_DIR / "fire_trajectories.json", "w"))
print("saved results/fire_trajectories.json")
