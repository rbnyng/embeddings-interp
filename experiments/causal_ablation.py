"""Remove the burn family from the Dixie 2022 embeddings and measure what a linear probe loses.

Three probe tasks on the Dixie tile: burned vs unburned (dNBR), north- vs south-facing steep slopes (Copernicus DEM,
slope above 0.15, northness above 0.3 or below -0.3), and unburned tree vs grass (ESA WorldCover classes 10 and 30).
Each probe is fit on raw embeddings. It is then scored again after removing the decoded part of F566 alone, of the
burn family (lib.config.BURN_FAMILY), or of each of 200 random same-size sets of live features outside the family
(mean and 95% interval). Live features fire on more than 0.5% of 2022 pixels. Also records a probe refit after
removing the family, a probe on family activations alone, and how far each removal moves the embedding on average.

Usage: python experiments/causal_ablation.py
Writes results/ablation_results.json. Run dixie_dictionary.py first.
"""
import json
import numpy as np

from lib.config import (FIRES, DIXIE, BURN_FAMILY, GT_DIR,
                                   RESULTS_DIR, dnbr_path)
from lib.tiles import load_tile
from lib.groundtruth import burn_masks
from lib.sae import load_sae, encode_array, ablate
from lib.ablation import build_probe_set, fit_probe, eval_probe

rng = np.random.default_rng(0)
sae = load_sae()
TILE = FIRES[DIXIE]["tile"]
emb = load_tile(TILE, 2022)
H, W, _ = emb.shape
dnbr = np.load(dnbr_path(DIXIE))[:H, :W]
burn, unburn = burn_masks(dnbr)
wc = np.load(GT_DIR / "gt_worldcover_dixie.npy")[:H, :W]
terr = np.load(GT_DIR / "gt_terrain_dixie.npz")
north, slope = terr["northness"][:H, :W], terr["slope"][:H, :W]
steep = slope > 0.15

TASKS = {
    "burn": (burn, unburn),
    "aspect": (steep & (north > 0.3), steep & (north < -0.3)),
    "tree_grass": ((wc == 10) & unburn, (wc == 30) & unburn),
}

stats = json.load(open(RESULTS_DIR / "feature_stats.json"))
alive = np.array(stats["act_freq"]["2022"]) > 0.005
pool = [i for i in np.where(alive)[0] if i not in BURN_FAMILY]
# random sets the size of the family, drawn from live features outside it (active on more than 0.5% of pixels in 2022)
N_DRAWS = 200
DRAWS = [list(rng.choice(pool, size=len(BURN_FAMILY), replace=False))
         for _ in range(N_DRAWS)]

results = {}
for name, (pos, neg) in TASKS.items():
    X, y, tr = build_probe_set(emb, pos, neg, seed=0)
    clf = fit_probe(X, y, tr)
    res = {"clean": eval_probe(clf, X, y, tr)}
    for tag, feats in (("abl_F566", [566]), ("abl_family", BURN_FAMILY)):
        Xa = ablate(sae, X, feats)
        res[tag] = eval_probe(clf, Xa, y, tr)
        res[tag]["mean_dx_pos"] = float(np.linalg.norm((X - Xa)[y == 1], axis=1).mean())
        res[tag]["mean_dx_neg"] = float(np.linalg.norm((X - Xa)[y == 0], axis=1).mean())
        if tag == "abl_family":
            res["retrained_on_ablated"] = eval_probe(fit_probe(Xa, y, tr), Xa, y, tr)
    # does the same as ablate(), but encodes once and reuses it for every draw
    F = encode_array(sae, X)
    Wd = sae.W_d.data.numpy()
    accs, dxp, dxn = [], [], []
    for S in DRAWS:
        Xa = X - F[:, S] @ Wd[:, S].T
        accs.append(eval_probe(clf, Xa, y, tr)["acc"])
        dxp.append(float(np.linalg.norm((X - Xa)[y == 1], axis=1).mean()))
        dxn.append(float(np.linalg.norm((X - Xa)[y == 0], axis=1).mean()))
    res["abl_random"] = dict(        # random sets the size of the family
        acc=float(np.mean(accs)),
        acc_lo=float(np.percentile(accs, 2.5)),
        acc_hi=float(np.percentile(accs, 97.5)),
        acc_draw0=float(accs[0]), n_draws=N_DRAWS,
        mean_dx_pos=float(np.mean(dxp)), mean_dx_neg=float(np.mean(dxn)))
    fam = F[:, BURN_FAMILY]
    res["family_only"] = eval_probe(fit_probe(fam, y, tr), fam, y, tr)
    results[name] = res
    print(name, {k: round(v["acc"], 3) for k, v in res.items()}, flush=True)

json.dump(results, open(RESULTS_DIR / "ablation_results.json", "w"))
print("saved results/ablation_results.json")
