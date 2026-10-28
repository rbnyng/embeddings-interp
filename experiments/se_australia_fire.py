"""SE Australia fire: screen each Black Summer fire in an AlphaEarth dictionary and find a fire family.

Each unit is one wildfire of the SE Australia inventory. The dictionary (512 features, 8
active per pixel) is trained on 60 random wildfire tiles. Disturbed pixels are inside the fire's GA outline and
undisturbed pixels are the rest of its tile, at stride 2. A fire needs at least 300 burned pixels and years 2019-2021.
Its row holds the burned minus unburned gap of all 512 features in each year 2017-2023.
The family is screened with the anchored bar on the wildfires ignited in September-October 2019, with expected window
2019-2020. All other wildfires are held out.

Usage: python experiments/se_australia_fire.py
Writes results/se_australia_fire.jsonl and se_australia_fire.json.
Caches the dictionary at data/checkpoints/aef_australia/sae_aef_au.pt.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import CKPT_DIR, RESULTS_DIR, workers
from lib.alphaearth import compute_scale, load_aef
from lib.screen import anchored_family
import se_australia_inventory
from lib.sae import group_sums, load_sae, train_and_save

YEARS = se_australia_inventory.YEARS
S = 2
CKPT = CKPT_DIR / "aef_australia" / "sae_aef_au.pt"
JSONL = RESULTS_DIR / "se_australia_fire.jsonl"

def get_sae(scale):
    if CKPT.exists():
        sae = load_sae(path=CKPT, d=64, m=512, k=8)
        return sae
    rng = np.random.default_rng(0)
    tiles = sorted({r["tile"] for r in se_australia_inventory.FIRES})
    pick = rng.choice(tiles, size=60, replace=False)
    out = []
    for t in pick:
        for y in YEARS:
            try:
                x = np.nan_to_num(load_aef(t, y, stride=1)) * scale
            except Exception:
                continue
            H, W, _ = x.shape
            idx = rng.choice(H * W, 1700, replace=False)
            out.append(x.reshape(-1, 64)[idx])
            del x
    X = np.concatenate(out).astype(np.float32)
    print(f"AU corpus {X.shape}", flush=True)
    sae = train_and_save(X, CKPT, epochs=40, seed=0, log=True, d=64, m=512, k=8)
    sae.eval()
    return sae

def screen_fire(args):
    rec, scale = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae = load_sae(path=CKPT, d=64, m=512, k=8)
        b = se_australia_inventory.fire_mask(rec)
        traj = {}
        for y in YEARS:
            try:
                x = np.nan_to_num(load_aef(rec["tile"], y, stride=S)) * scale
            except Exception:
                continue
            H, W, _ = x.shape
            bb = b[:H, :W]
            uu = ~bb
            if bb.sum() < 300:
                return dict(ok=False, uid=rec["uid"], error="few px")
            s, n = group_sums(sae, x, {"burn": bb, "unburn": uu})
            traj[y] = ((s["burn"] / n["burn"]) - (s["unburn"] / max(n["unburn"], 1))).tolist()
        if not {2019, 2020, 2021} <= set(traj):
            return dict(ok=False, uid=rec["uid"], error="gap")
        return dict(ok=True, uid=rec["uid"], name=rec["name"],
                    ig_date=rec["ig_date"], area_ha=rec["area_ha"],
                    traj={str(y): v for y, v in traj.items()})
    except Exception as e:
        return dict(ok=False, uid=rec.get("uid"), error=repr(e))

def run():
    scale = compute_scale()
    print(f"SCALE x{scale:.1f}", flush=True)
    get_sae(scale)
    print("dictionary ready", flush=True)
    done = set()
    if JSONL.exists():
        for line in open(JSONL):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [r for r in se_australia_inventory.FIRES if r["uid"] not in done]
    print(f"{len(todo)} fires to screen (full 512-d contrast stored)", flush=True)
    out = open(JSONL, "a")
    with ProcessPoolExecutor(max_workers=workers(60)) as ex:
        for i, r in enumerate(ex.map(screen_fire, [(t, scale) for t in todo]), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 30 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)
    out.close()
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    scr = [r for r in rows if "2019-09" <= r["ig_date"][:7] <= "2019-10"]
    held = [r for r in rows if r not in scr]
    A = {y: np.median([r["traj"][str(y)] for r in scr], axis=0) for y in YEARS}
    fam = anchored_family({y - 2019: A[y] for y in YEARS})
    print(f"{len(scr)} screening fires, {len(held)} held-out wildfires; family {len(fam)} features {fam}")
    json.dump(dict(family=fam, n_held=len(held), n_screen=len(scr)),
              open(RESULTS_DIR / "se_australia_fire.json", "w"))
    print("\nsaved results/se_australia_fire.json")

if __name__ == "__main__":
    run()
