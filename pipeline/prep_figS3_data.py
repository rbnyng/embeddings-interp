"""Sample and encode the data for Supplementary Fig. S3 (invariance), from the rice, wheat and fire invariance samples.

Each concept has four pixel groups: the concept (c_) and its surroundings (b_), each in the home region where the
family was learned and in a far region.
  rice          Arkansas to California, rice against other cropland
  winter wheat  US to Europe, wheat against other cropland
  fire          California to Europe, burned against unburned, with the burn family
For each group we keep up to N_SCAT (1,500) random pixels to plot and every pixel's family activation sum. For each
concept we also keep the distances between group centers in the raw embedding.

Usage: python pipeline/prep_figS3_data.py
Writes results/figS3_data.npz. load() reads it back.
"""
import json

import numpy as np
import torch

from lib.config import BURN_FAMILY, RESULTS_DIR
from lib.sae import encode_array, load_sae

N_SCAT = 1500
OUT = RESULTS_DIR / "figS3_data.npz"
ORDER = ("b_home", "b_far", "c_home", "c_far")
META = ("name", "home", "far", "c", "b")


def _dict(ckpt):
    return load_sae(path=ckpt)


def _masks(place, cls, home, far, conc, bg):
    return {"c_home": (place == home) & (cls == conc), "c_far": (place == far) & (cls == conc),
            "b_home": (place == home) & (cls == bg), "b_far": (place == far) & (cls == bg)}


def full_sums(s, place, cls, home, far, conc, bg, x):
    """Family sums and raw embeddings of every pixel in each group."""
    m = _masks(place, cls, home, far, conc, bg)
    out = {k: s[v] for k, v in m.items()}
    out["x"] = {k: x[v] for k, v in m.items()}
    return out


def groups(rng, x, code, fam, place, cls, home, far, conc, bg):
    """The plotted subsample of each group: (raw embedding, family sum)."""
    out = {}
    for k, v in _masks(place, cls, home, far, conc, bg).items():
        i = rng.choice(np.flatnonzero(v), min(N_SCAT, v.sum()), replace=False)
        out[k] = (x[i], code(i)[:, fam].sum(1))
    return out


def build():
    import experiments.rice_invariance as ri
    import experiments.wheat_invariance as wi
    rng = np.random.default_rng(0)            # seed and concept order fix the plotted subsample
    torch.set_num_threads(32)
    R = []
    z = np.load(RESULTS_DIR / "rice_invariance.npz"); sae = _dict(ri.CKPT); x = z["x"]
    fam = json.load(open(ri.FAMILY))["family"]
    R.append(dict(name="rice", home="Arkansas", far="California", c="rice", b="cropland",
                  g=groups(rng, x, lambda i: encode_array(sae, x[i]), fam, z["region"], z["cls"],
                           "AR", "CA", "rice", "other"),
                  full=full_sums(z["f"].sum(1), z["region"], z["cls"], "AR", "CA", "rice", "other", x)))
    z = np.load(RESULTS_DIR / "wheat_invariance.npz"); sae = _dict(wi.CKPT); xw = z["x"]
    fam = json.load(open(wi.FAMILY))["family"]
    R.append(dict(name="winter wheat", home="US", far="Europe", c="wheat", b="cropland",
                  g=groups(rng, xw, lambda i: encode_array(sae, xw[i]), fam, z["side"], z["cls"],
                           "US", "EU", "wheat", "other"),
                  full=full_sums(z["f"].sum(1), z["side"], z["cls"], "US", "EU", "wheat", "other", xw)))
    z = np.load(RESULTS_DIR / "fire_invariance.npz")
    xf = np.concatenate([z["xb"], z["xu"]]); cf = np.concatenate([z["fb"], z["fu"]])
    reg = np.concatenate([z["region_b"], z["region_u"]])
    cls = np.r_[np.full(len(z["xb"]), "burn"), np.full(len(z["xu"]), "unburned")]
    fam = list(BURN_FAMILY)
    R.append(dict(name="fire", home="California", far="Europe", c="burn", b="unburned",
                  g=groups(rng, xf, lambda i: cf[i], fam, reg, cls, "CA", "EU", "burn", "unburned"),
                  full=full_sums(cf[:, fam].sum(1), reg, cls, "CA", "EU", "burn", "unburned", xf)))
    return R


def centre_distances(V):
    """Distances between group centers in the raw embedding, divided by the pooled within-group RMS spread."""
    V = {k: np.asarray(v, np.float64).reshape(len(v), -1) for k, v in V.items()}
    mu = {k: v.mean(0) for k, v in V.items()}
    sp = np.sqrt(np.mean([((v - mu[k]) ** 2).sum(1).mean() for k, v in V.items()]))
    dd = lambda a, b: float(np.linalg.norm(mu[a] - mu[b]) / sp)
    return dict(concept_learned_new=dd("c_home", "c_far"), concept_new_vs_surroundings_new=dd("c_far", "b_far"),
                surroundings_learned_new=dd("b_home", "b_far"))


def save(R):
    arrs, meta = {}, []
    for r, row in enumerate(R):
        meta.append(dict({k: row[k] for k in META}, centre=centre_distances(row["full"]["x"])))
        for k in ORDER:
            arrs[f"{r}/g/{k}/x"] = row["g"][k][0]
            arrs[f"{r}/g/{k}/s"] = row["g"][k][1]
            arrs[f"{r}/full/{k}/s"] = row["full"][k]
    np.savez(OUT, meta=json.dumps(meta), **arrs)
    print("wrote", OUT)


def load():
    """Rows for the figure. Each has the name, places and labels, plus g[k] = (embedding, family sum) for the
    plotted subsample, full[k] = every pixel's family sum, and centre = the three center distances."""
    with np.load(OUT) as z:
        z = {k: z[k] for k in z.files}
    rows = []
    for r, m in enumerate(json.loads(str(z["meta"]))):
        g = {k: (z[f"{r}/g/{k}/x"], z[f"{r}/g/{k}/s"]) for k in ORDER}
        full = {k: z[f"{r}/full/{k}/s"] for k in ORDER}
        rows.append(dict(m, g=g, full=full))
    return rows


if __name__ == "__main__":
    save(build())
