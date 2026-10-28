"""Test whether a flooded-rice family found in the Arkansas Delta transfers to the Sacramento Valley.

The dictionary is trained only on the crop-atlas Arkansas rice tiles. Pixel groups come from USDA CDL at stride 2: rice
(code 3) and other cropland (codes 1-60 except rice). The family is screened on the Arkansas tiles with the crop
atlas's anchored bar on the rice minus other-cropland gap in 2018-2021. Sampling and analysis use the held-out years
2022-2024 in both regions.
  train    train the dictionary on the Arkansas tiles
  screen   mean activations on rice and other cropland per tile-year (groups of 400 pixels or more), then the family
  sample   up to 3,000 rice and 3,000 other-cropland pixels per held-out tile-year (groups of 200 or more), with raw
           embeddings and family activations. Then runs analyze
  analyze  for each region, how often the family fires on rice and on other cropland, and the AUC of the family sum.
           Also the share of raw variance explained by tile among rice pixels where the family fires (150 per tile),
           and distances between raw group centers, absolute and divided by the pooled RMS spread

Usage: python experiments/rice_invariance.py train|screen|sample|analyze
Writes results/rice_invariance_tables.jsonl, rice_invariance_family.json, rice_invariance.npz and rice_invariance.json,
and caches the dictionary at data/checkpoints/rice_invariance/sae_arkansas.pt.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import CKPT_DIR, RESULTS_DIR, workers
from lib.sae import encode_array, load_sae, train_and_save
from lib.tiles import load_tile, sample_pixels
import experiments.crop_atlas as ca
from lib.stats import group_r2

S = 2
RICE = (3,)
SCREEN_YEARS, HELD = ca.SCREEN_YEARS, ca.HELD
YEARS = ca.YEARS
REGION = {r["tile"]: "AR" if r["subregion"] == "rice" else "CA"
          for r in ca.SEL if r["system"] == "rice"}
TILES = {g: sorted(t for t, r in REGION.items() if r == g) for g in ("AR", "CA")}
CKPT = CKPT_DIR / "rice_invariance" / "sae_arkansas.pt"
TABLES = RESULTS_DIR / "rice_invariance_tables.jsonl"
FAMILY = RESULTS_DIR / "rice_invariance_family.json"
NPZ = RESULTS_DIR / "rice_invariance.npz"
OUT = RESULTS_DIR / "rice_invariance.json"
N_PER = 3000
N_LOC = 150

def phase_train():
    if CKPT.exists():
        print("cached"); return
    X = np.concatenate([sample_pixels([(t, y) for y in YEARS], n_per=5000, seed=0)
                        for t in TILES["AR"]])
    np.random.default_rng(0).shuffle(X)
    print(f"Arkansas corpus {X.shape} from {len(TILES['AR'])} tiles", flush=True)
    train_and_save(X, CKPT, epochs=40, seed=0, log=True)

def _groups(lab):
    rice = np.isin(lab, RICE)
    other = (lab >= 1) & (lab <= ca.CROPLAND_MAX) & ~rice
    return rice, other

def run_tables(tile):
    torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    try:
        sae, cdl, tables = load_sae(path=CKPT), ca.cdl_stack(tile), {}
        for y in ca.CDL_YEARS:
            x = load_tile(tile, y, stride=S)
            H, W, _ = x.shape
            rice, other = _groups(cdl[y][:H, :W])
            f = encode_array(sae, x)
            tables[str(y)] = {k: f[m].mean(0).tolist() for k, m in
                              (("rice", rice), ("_other_crop", other)) if m.sum() >= 400}
        return dict(ok=True, tile=tile, region=REGION[tile], tables=tables)
    except Exception as e:
        return dict(ok=False, tile=tile, error=repr(e))

def phase_screen():
    with open(TABLES, "w") as f, ProcessPoolExecutor(max_workers=workers(12)) as ex:
        for r in ex.map(run_tables, sorted(REGION)):
            f.write(json.dumps(r) + "\n")
            print(r["tile"], r.get("region"), r["ok"], r.get("error", ""), flush=True)
    rows = [r for r in map(json.loads, open(TABLES)) if r["ok"] and r["region"] == "AR"]
    fam = ca.blind_family(rows, "rice")
    json.dump(dict(family=fam, screened_on=[r["tile"] for r in rows], years=SCREEN_YEARS),
              open(FAMILY, "w"), indent=1)
    print(f"rice family, screened on {len(rows)} Arkansas tiles: {len(fam)} features {fam}")

def run_sample(tile):
    torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    fam = json.load(open(FAMILY))["family"]
    sae, cdl, rng, out = load_sae(path=CKPT), ca.cdl_stack(tile), np.random.default_rng(0), []
    for y in HELD:
        x = load_tile(tile, y, stride=S)
        H, W, _ = x.shape
        X = x.reshape(-1, x.shape[-1])
        for cls, m in zip(("rice", "other"), _groups(cdl[y][:H, :W])):
            idx = np.flatnonzero(m)
            if len(idx) < 200:
                continue
            idx = rng.choice(idx, min(N_PER, len(idx)), replace=False)
            out.append(dict(region=REGION[tile], tile=tile, year=y, cls=cls, x=X[idx],
                            f=encode_array(sae, X[idx])[:, fam]))
    return out

def phase_sample():
    rows = []
    with ProcessPoolExecutor(max_workers=workers(12)) as ex:
        for r in ex.map(run_sample, sorted(REGION)):
            rows += r
    rep = lambda k: np.concatenate([[r[k]] * len(r["x"]) for r in rows])
    np.savez_compressed(NPZ, x=np.concatenate([r["x"] for r in rows]).astype(np.float32),
                        f=np.concatenate([r["f"] for r in rows]).astype(np.float32),
                        region=rep("region"), tile=rep("tile"), cls=rep("cls"))
    print(f"saved {sum(len(r['x']) for r in rows)} pixels from {len(rows)} tile-year-classes")
    analyze()

def analyze():
    from sklearn.metrics import roc_auc_score
    z = np.load(NPZ)
    x, f, reg, tile, cls = z["x"], z["f"], z["region"], z["tile"], z["cls"]
    s = f.sum(1)
    on = s > 0
    rng = np.random.default_rng(0)
    out = dict(family_size=int(f.shape[1]),
               n={f"{g}_{c}": int(((reg == g) & (cls == c)).sum())
                  for g in ("AR", "CA") for c in ("rice", "other")})
    print(f"family: {f.shape[1]} features;  pixels: {out['n']}")

    out["family_fires"], out["auc_rice_vs_other"] = {}, {}
    for g in ("AR", "CA"):
        r, o = (reg == g) & (cls == "rice"), (reg == g) & (cls == "other")
        out["family_fires"][f"{g}_rice"] = float(on[r].mean())
        out["family_fires"][f"{g}_other"] = float(on[o].mean())
        out["auc_rice_vs_other"][g] = float(roc_auc_score(np.r_[np.ones(r.sum()), np.zeros(o.sum())],
                                                          np.r_[s[r], s[o]]))
    print("family fires on:", "  ".join(f"{k} {v:.3f}" for k, v in out["family_fires"].items()))
    print("AUC rice vs other cropland:", "  ".join(f"{k} {v:.3f}" for k, v in out["auc_rice_vs_other"].items()))

    m = (cls == "rice") & on
    idx = np.concatenate([rng.choice(ii, N_LOC, replace=False) for t in np.unique(tile[m])
                          if len(ii := np.flatnonzero(m & (tile == t))) >= N_LOC])
    out["location_r2_raw"] = dict(r2=group_r2(x[idx], tile[idx]), n_loc=int(len(np.unique(tile[idx]))),
                                  n_ca=int(len(np.unique(tile[idx][reg[idx] == "CA"]))))
    out["region_r2_raw"] = group_r2(x[idx], reg[idx])
    print(f"family-active rice, {N_LOC}/tile, {out['location_r2_raw']['n_loc']} tiles: "
          f"location R^2 raw {out['location_r2_raw']['r2']:.3f}; region (AR vs CA) R^2 {out['region_r2_raw']:.3f}")

    tv = lambda X: float(((X - X.mean(0)) ** 2).sum(1).mean())
    G = {f"{g}_{c}": x[(reg == g) & (cls == c)] for g in ("AR", "CA") for c in ("rice", "other")}
    out["raw_centroid_dist"] = {}
    print(f"raw centroid distances  {'abs':>6s} {'norm':>6s}")
    for a, b in (("AR_rice", "CA_rice"), ("CA_rice", "CA_other"), ("AR_rice", "AR_other"),
                 ("AR_other", "CA_other")):
        dd = float(np.linalg.norm(G[a].mean(0) - G[b].mean(0)))
        nd = dd / np.sqrt((tv(G[a]) + tv(G[b])) / 2)
        out["raw_centroid_dist"][f"{a}__{b}"] = dict(abs=dd, norm=float(nd))
        print(f"  {a + ' - ' + b:22s} {dd:6.2f} {nd:6.2f}")
    json.dump(out, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    {"train": phase_train, "screen": phase_screen, "sample": phase_sample,
     "analyze": analyze}[sys.argv[1]]()
