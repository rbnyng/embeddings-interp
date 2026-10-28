"""Encode Europe-atlas fire tiles with the Dixie dictionary, for the specificity and invariance tests on European fire.

Tiles are the Europe-atlas tiles in results/europe_atlas.jsonl with at least one EFDA fire tile-year. Pixel groups are
the Europe atlas's fire_Y (forest EFDA burned in year Y) and intact forest. Each table is every feature's mean
activation over each pixel group, for each year 2017-2025, at stride 2.

Usage: python experiments/europe_fire_dixie.py
Writes results/europe_fire_dixie.jsonl.
"""
import json, os
from concurrent.futures import ProcessPoolExecutor
import torch
from lib.config import RESULTS_DIR, workers
from lib.sae import load_sae
from lib.trajectories import group_tables
import experiments.europe_atlas as ea

OUT = RESULTS_DIR / "europe_fire_dixie.jsonl"

def run(tile):
    torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    try:
        g = ea.tile_groups(tile)
        g = {k: v for k, v in g.items() if k == "intact" or k.startswith("fire_")}
        sae = load_sae()
        T = group_tables(sae, tile, ea.YEARS, g, stride=ea.S)
        return dict(ok=True, tile=tile, lat=float(tile.split("_")[2]),
                    countries=sorted({m["country"] for m in ea.TILE_META[tile]}),
                    tables={str(y): {k: T[y][k].tolist() for k in T[y]} for y in ea.YEARS})
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

if __name__ == "__main__":
    tiles = []
    for l in open(RESULTS_DIR / "europe_atlas.jsonl"):
        r = json.loads(l)
        if r.get("ok") and any(k.startswith("fire_") for k in r["tables"]["2018"]):
            tiles.append(r["tile"])
    print(len(tiles), "tiles", flush=True)
    with open(OUT, "w") as f, ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for i, r in enumerate(ex.map(run, tiles), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 10 == 0: print(f"[{i}/{len(tiles)}]", flush=True)
