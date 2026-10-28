"""Train the Iowa dictionary on one Corn Belt tile and cache the tile's USDA CDL labels.

The dictionary is trained on 100,000 pixels of IOWA_TILE from each year 2018-2024. CDL is cached for 2018-2021.

Usage: python experiments/iowa_dictionary.py
Writes data/checkpoints/sae_iowa.pt and data/groundtruth/gt_cdl.npz.
"""
import numpy as np

from lib.config import IOWA_TILE, IOWA_YEARS, CDL_YEARS, GT_DIR, CKPT_DIR
from lib.tiles import download_tile, download_landmask, sample_pixels
from lib.groundtruth import fetch_cdl
from lib.sae import train_and_save

CKPT = CKPT_DIR / "sae_iowa.pt"

download_landmask(IOWA_TILE)
for y in IOWA_YEARS:
    download_tile(IOWA_TILE, y)
cdl_path = GT_DIR / "gt_cdl.npz"
if cdl_path.exists():
    cdl = dict(np.load(cdl_path))
else:
    cdl = {str(y): fetch_cdl(IOWA_TILE, y) for y in CDL_YEARS}
    cdl = {k: v for k, v in cdl.items() if v is not None}
    np.savez_compressed(cdl_path, **cdl)
print(f"CDL years {sorted(cdl)} in {cdl_path}")

if CKPT.exists():
    print(f"dictionary cached at {CKPT}")
else:
    X = sample_pixels([(IOWA_TILE, y) for y in IOWA_YEARS], seed=0)
    train_and_save(X, CKPT, threads=None, epochs=40, seed=0)
    print(f"dictionary trained on {X.shape[0]:,} pixels, saved to {CKPT}")
