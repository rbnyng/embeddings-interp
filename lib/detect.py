"""Detection settings (mines, solar): does a family tell a surface apart from its surroundings at held-out sites?

lib.screen.anchored_bar_family picks the family from the median gap between target and surroundings over the
screening sites. A pixel counts as target when the family's mean activation is above a cut-off. The cut-off is set
once on the screening sites, halfway between the median target mean and the median surroundings mean. A held-out
site validates when its balanced accuracy reaches BAR (0.70, the crop atlas's pass mark). The null draws 200 random
families of the same size, and each gets its own cut-off.

`pairs` has one (target mean activation, surroundings mean activation) pair per screening site, each of shape (M,).
A site to score is (record, target activations, surroundings activations), each (n, M), dense or scipy sparse.
"""
import zlib

import numpy as np

from lib.screen import anchored_bar_family

BAR = 0.70


def screen(pairs):
    """Find the family and its cut-off, using the screening sites only."""
    fam = anchored_bar_family(np.median([t - b for t, b in pairs], axis=0))
    return fam, (cutoff(pairs, fam) if fam else None)


def cutoff(pairs, fam):
    t = np.median([np.asarray(t)[fam].mean() for t, _ in pairs])
    b = np.median([np.asarray(b)[fam].mean() for _, b in pairs])
    return float((t + b) / 2)


def family_mean(A, fam):
    """Mean activation of the family's features at each pixel."""
    return np.asarray(A[:, fam].mean(1)).ravel()


def score(sites, fam, cut):
    """For each site: balanced accuracy at the cut-off, the accuracy on each side, the family's AUC, and a pass flag."""
    from sklearn.metrics import roc_auc_score
    out = []
    for rec, T, B in sites:
        a, b = family_mean(T, fam), family_mean(B, fam)
        ta, ba = float((a > cut).mean()), float((b <= cut).mean())
        acc = (ta + ba) / 2
        auc = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
        out.append(dict(**rec, acc=acc, target_acc=ta, background_acc=ba, auc=auc, ok=bool(acc >= BAR)))
    return out


def null_rates(pairs, sites, size, m, seed, draws=200):
    """Pass rates of `draws` random families of `size` features out of m, seeded by the string `seed`."""
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    rates = []
    for _ in range(draws):
        f = rng.choice(m, size, replace=False).tolist()
        rates.append(float(np.mean([x["ok"] for x in score(sites, f, cutoff(pairs, f))])))
    return rates


def summary(v, null):
    """Pass rate, held-out count, median balanced accuracy and AUC, and the null's median, 95th percentile and p."""
    rate = float(np.mean([x["ok"] for x in v]))
    return dict(rate=rate, n_held=len(v),
                acc_median=float(np.median([x["acc"] for x in v])), auc_median=float(np.median([x["auc"] for x in v])),
                null=dict(median=float(np.median(null)), p95=float(np.percentile(null, 95)),
                          p=float(np.mean(np.array(null) >= rate)), draws=null))
