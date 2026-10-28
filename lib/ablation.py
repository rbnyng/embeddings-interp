"""Linear probes on embeddings, with train and test rows split by a spatial checkerboard."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def checkerboard(H, W, block=64):
    """True on the training blocks of a checkerboard over an (H, W) grid."""
    return (((np.arange(H)[:, None] // block) + (np.arange(W)[None, :] // block)) % 2) == 0

def sample_mask(mask, n, rng):
    r, c = np.where(mask)
    k = rng.choice(r.size, size=min(n, r.size), replace=False)
    return r[k], c[k]

def build_probe_set(emb, pos_mask, neg_mask, n=15000, block=64, seed=0):
    """Sample up to n pixels per class from each checkerboard half of an (H, W, D) embedding, as (X, y, train)."""
    rng = np.random.default_rng(seed)
    H, W = pos_mask.shape
    tr_block = checkerboard(H, W, block)
    Xs, ys, trs = [], [], []
    for lab, m in ((1, pos_mask), (0, neg_mask)):
        for tr_flag, mm in ((True, m & tr_block), (False, m & ~tr_block)):
            r, c = sample_mask(mm, n, rng)
            Xs.append(emb[r, c])
            ys.append(np.full(r.size, lab))
            trs.append(np.full(r.size, tr_flag))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(trs).astype(bool)

def fit_probe(X, y, tr, C=1.0):
    return LogisticRegression(max_iter=3000, C=C).fit(X[tr], y[tr])

def eval_probe(clf, X, y, tr):
    """Score the probe on the held-out half of the checkerboard."""
    Xt, yt = X[~tr], y[~tr]
    return dict(acc=float(clf.score(Xt, yt)),
                auc=float(roc_auc_score(yt, clf.decision_function(Xt))))
