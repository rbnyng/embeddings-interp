"""Encode the Dixie Fire tile (2022, stride 2) with the Dixie dictionary, for Figure 1.

Usage: python pipeline/prep_fig1_encode.py
Writes results/fig1_dixie_feats.npy, the (H, W, 1024) float16 activations.
"""

import numpy as np

from lib.config import RESULTS_DIR
from lib.tiles import load_tile
from lib.sae import encode_array, load_sae

S = 2                                    # stride. Must match figures/paper_fig1_decompose.py
TILE, YEAR = "grid_-121.25_40.05", 2022  # Dixie Fire footprint, first full year after the fire
ROWS = 192                               # rows encoded per chunk
OUT = RESULTS_DIR / "fig1_dixie_feats.npy"

x = load_tile(TILE, YEAR, stride=S)
h, w, _ = x.shape
sae = load_sae()
m = sae.W_d.shape[1]
print(f"{TILE} {YEAR} stride {S}: {h}x{w} px -> {m} features", flush=True)

F = np.empty((h, w, m), dtype=np.float16)
for r0 in range(0, h, ROWS):
    r1 = min(r0 + ROWS, h)
    F[r0:r1] = encode_array(sae, x[r0:r1]).astype(np.float16)
    print(f"  rows {r0}-{r1}", flush=True)

np.save(OUT, F)
print(f"wrote {OUT} {F.shape} {F.dtype} "
      f"({OUT.stat().st_size / 1e6:.0f} MB)")
