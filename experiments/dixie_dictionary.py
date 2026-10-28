"""Train the Dixie dictionary on Tessera v1 embeddings of the Dixie Fire tile and tabulate its features.

The dictionary is a TopK SAE with 1,024 features and k = 16, trained with seed 0 on 100k pixels per year, 2019-2024.
The script downloads the tile and its reference products: dNBR (Sentinel-2 L2A), ESA WorldCover v200 and Copernicus
GLO-30 terrain. For each year it records every feature's activation rate over the tile and its mean over burned and
unburned pixels, at full resolution.

Usage: python experiments/dixie_dictionary.py
Writes lib.config.DIXIE_CKPT (if missing), results/feature_stats.json and the reference maps in data/groundtruth.
"""
import json
import numpy as np

from lib.config import FIRES, DIXIE, DIXIE_SAE_YEARS, GT_DIR, RESULTS_DIR, DIXIE_CKPT
from lib.tiles import load_tile, download_tile, download_landmask, sample_pixels
from lib.groundtruth import fetch_dnbr, fetch_worldcover, fetch_terrain, burn_masks
from lib.sae import load_sae, encode_array, train_and_save

TILE = FIRES[DIXIE]["tile"]

# ---- data ------------------------------------------------------------------
download_landmask(TILE)
for y in DIXIE_SAE_YEARS:
    download_tile(TILE, y)
dnbr = fetch_dnbr(TILE, FIRES[DIXIE]["pre"], FIRES[DIXIE]["post"],
                  out_path=GT_DIR / f"dnbr_{DIXIE}.npy")
fetch_worldcover(TILE, out_path=GT_DIR / "gt_worldcover_dixie.npy")
fetch_terrain(TILE, out_path=GT_DIR / "gt_terrain_dixie.npz")

# ---- dictionary ------------------------------------------------------------
if DIXIE_CKPT.exists():
    sae = load_sae()
else:
    X = sample_pixels([(TILE, y) for y in DIXIE_SAE_YEARS], seed=0)
    sae = train_and_save(X, DIXIE_CKPT, threads=None, epochs=40, seed=0)

# ---- stats for each year, a block of rows at a time (activations for a full tile take about 4 GB)
stats = {"act_freq": {}, "mean_burn": {}, "mean_unburn": {}}
CHUNK, M = 64, 1024
for y in DIXIE_SAE_YEARS:
    x = load_tile(TILE, y)
    H, W, _ = x.shape
    b, u = burn_masks(dnbr[:H, :W])
    freq = np.zeros(M); mb = np.zeros(M); mu = np.zeros(M)
    nb = nu = 0
    for r0 in range(0, H, CHUNK):
        r1 = min(r0 + CHUNK, H)
        f = encode_array(sae, x[r0:r1])
        freq += (f > 0).reshape(-1, M).sum(0)
        mb += f[b[r0:r1]].sum(0); nb += b[r0:r1].sum()
        mu += f[u[r0:r1]].sum(0); nu += u[r0:r1].sum()
    stats["act_freq"][y] = (freq / (H * W)).tolist()
    stats["mean_burn"][y] = (mb / max(nb, 1)).tolist()
    stats["mean_unburn"][y] = (mu / max(nu, 1)).tolist()
    del x
    print("year", y, "done", flush=True)

json.dump(stats, open(RESULTS_DIR / "feature_stats.json", "w"))
print("saved results/feature_stats.json")
