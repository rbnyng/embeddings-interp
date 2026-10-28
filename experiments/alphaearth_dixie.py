"""Train the AlphaEarth Dixie dictionary and tabulate its features on the five western US fires.

The dictionary has 512 features and k = 8, the same expansion and active share as the Tessera Dixie dictionary. It is
trained on 100k AlphaEarth pixels per year of the Dixie tile, 2019-2024, scaled to Tessera's range. For each fire and
year, the table holds every feature's mean activation over burned and unburned pixels (from dNBR), at stride 2.

Usage: python experiments/alphaearth_dixie.py
Writes results/alphaearth_stats.json and the checkpoint data/checkpoints/sae_aef_dixie.pt.
"""
import json
import numpy as np

from lib.config import FIRES, DIXIE, DIXIE_SAE_YEARS, CKPT_DIR, RESULTS_DIR
from lib.alphaearth import compute_scale, fetch_aef, load_aef, dequantize
from lib.trajectories import fire_tables
from lib.sae import load_sae, train_and_save

TILE = FIRES[DIXIE]["tile"]
CKPT = CKPT_DIR / "sae_aef_dixie.pt"
M, K = 512, 8


def _sample_aef_raw(n_per=100_000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for y in DIXIE_SAE_YEARS:
        raw = fetch_aef(TILE, y)
        H, W, _ = raw.shape
        idx = rng.choice(H * W, n_per, replace=False)
        out.append(dequantize(raw.reshape(-1, 64)[idx]))
    return np.nan_to_num(np.concatenate(out))

SCALE = compute_scale()      # converts AlphaEarth to Tessera units. See lib/alphaearth.py.
print(f"units: scale x{SCALE:.1f}", flush=True)

def sample_aef(n_per=100_000, seed=0):
    return _sample_aef_raw(n_per, seed) * SCALE

def loader(tile, year, stride=1):
    return np.nan_to_num(load_aef(tile, year, stride=stride)) * SCALE

if CKPT.exists():
    sae = load_sae(path=CKPT, d=64, m=M, k=K)
else:
    X = sample_aef()
    print("corpus:", X.shape, flush=True)
    sae = train_and_save(X, CKPT, threads=None, epochs=40, seed=0, d=64, m=M, k=K)
    sae.eval()

# ---- activation tables -----------------------------------------------------
tables = fire_tables(sae, stride=2, loader=loader)
for f, t in tables.items():
    print(f"{f}: fire year {t[0]}, years {t[1][0]}-{t[1][-1]}")

json.dump(dict(
    tables={f: dict(fire_year=t[0], years=t[1],
                    mean_burn={y: t[2][y].tolist() for y in t[1]},
                    mean_unburn={y: t[3][y].tolist() for y in t[1]})
            for f, t in tables.items()},
), open(RESULTS_DIR / "alphaearth_stats.json", "w"))
print("saved results/alphaearth_stats.json")
