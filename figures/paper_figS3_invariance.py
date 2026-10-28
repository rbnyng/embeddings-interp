"""Supplementary Figure S3: one feature family applied to a region its dictionary never saw, next to the raw embedding.

Rows are rice (Arkansas Delta to Sacramento Valley), winter wheat (US southern Plains to Europe) and fire (California
to Europe). Each panel shows the concept and its surroundings, both where the family was learned and where it is tested.
a, d, g  raw embeddings of the four groups on their first two principal components
b, e, h  the same points colored by the family's summed activation
c, f, i  each group's density along the first principal component of the four group centers (top) and along log
         family activation (bottom)

Writes figures/figS3_invariance.png and .pdf, and results/figS3_numbers.json.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from lib.config import FIG_DIR
from lib.numbers import FigureNumbers
from lib.style import js

js.use()
INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
COL = {"c_home": "#0b6a63", "b_home": "#9a9a9a", "c_far": "#6a3596", "b_far": "#cfcfcf"}
from matplotlib.colors import LinearSegmentedColormap
ZERO = "#dcdad4"
ACT = LinearSegmentedColormap.from_list("act", ["#f3d9c9", "#e57f52", "#8f2a08"])

# ---- data -----------------------------------------------------------------------------
from pipeline.prep_figS3_data import load
R = load()

def auc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))

# ---- figure ---------------------------------------------------------------------------
from matplotlib.lines import Line2D
FS_LET, FS_ANN, FS_TICK = 8.5, 7.0, 6.4
FIGW = 7.2
X_LAB = 0.40
GS = 0.12
GY = 0.50
WS = (FIGW - X_LAB - 2 * GS - GY - 0.04) / 3
HS = 2.0
HL = 0.36
GR = 0.20
HCB = 0.46
FIGH = HCB + 3 * (HS + HL) + 2 * GR + 0.04
fig = plt.figure(figsize=(FIGW, FIGH), dpi=300)
box = lambda x0, y0, w, h: fig.add_axes([x0 / FIGW, y0 / FIGH, w / FIGW, h / FIGH])
XS = [X_LAB, X_LAB + WS + GS, X_LAB + 2 * (WS + GS) + GY]

def letter(x, y, t):
    fig.text(x / FIGW, y / FIGH, t, fontsize=FS_LET, fontweight="bold", ha="left", va="bottom")

NUM = FigureNumbers("figS3")
ORDER = ("b_home", "b_far", "c_home", "c_far")
for r, row in enumerate(R):
    g = row["g"]
    H, F, c_, b_ = row["home"], row["far"], row["c"], row["b"]
    y0 = HCB + (2 - r) * (HS + HL + GR); ytop = y0 + HS
    allx = np.concatenate([g[k][0] for k in ORDER]); mu = allx.mean(0)
    _, _, Vt = np.linalg.svd(allx - mu, full_matrices=False)
    P = {k: (g[k][0] - mu) @ Vt[:2].T for k in g}

    # a, d, g: raw embedding, by group
    ax = box(XS[0], y0, WS, HS)
    for k in ORDER:
        ax.scatter(P[k][:, 0], P[k][:, 1], s=1.2, c=COL[k], lw=0,
                   alpha=0.6 if k.startswith("c") else 0.45, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("embedding PC 1", fontsize=FS_TICK, labelpad=1)
    ax.set_ylabel("embedding PC 2", fontsize=FS_TICK, labelpad=1)
    lim = (ax.get_xlim(), ax.get_ylim())
    fig.text(0.04 / FIGW, (y0 + HS / 2) / FIGH, row["name"], rotation=90, ha="left", va="center",
             fontsize=FS_ANN + 0.8, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], marker="s", ls="", ms=5, mfc=COL[k], mec="none", label=l)
                        for k, l in (("c_home", f"{H} {c_}"), ("b_home", f"{H} {b_}"),
                                     ("c_far", f"{F} {c_}"), ("b_far", f"{F} {b_}"))],
               loc="lower left", bbox_to_anchor=((XS[0] - 0.04) / FIGW, (ytop + 0.01) / FIGH),
               ncol=2, frameon=False, fontsize=FS_TICK, handletextpad=0.2, columnspacing=1.0,
               borderaxespad=0.0, labelspacing=0.15)
    letter(XS[0] - 0.20, ytop + 0.20, "adg"[r])

    # b, e, h: same points, by family activation
    ax = box(XS[1], y0, WS, HS)
    allp = np.concatenate([P[k] for k in ORDER]); alla = np.log1p(np.concatenate([g[k][1] for k in ORDER]))
    o = np.argsort(alla); z0 = alla[o] <= 0
    ax.scatter(allp[o][z0, 0], allp[o][z0, 1], s=1.2, c=ZERO, lw=0, rasterized=True)
    sc = ax.scatter(allp[o][~z0, 0], allp[o][~z0, 1], s=1.2, c=alla[o][~z0], cmap=ACT, lw=0,
                    vmin=0, vmax=np.percentile(alla[alla > 0], 98), rasterized=True)
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("embedding PC 1", fontsize=FS_TICK, labelpad=1)
    letter(XS[1] - 0.02, ytop + 0.20, "beh"[r])

    # c, f, i: one line per basis, in pooled within-group SDs, flipped so the learned concept sits
    # left of the new one. A curve's area is its share of the group, so silent pixels still count in the total.
    from scipy.stats import gaussian_kde
    C = np.stack([g[k][0].mean(0) for k in ORDER]); C = C - C.mean(0)
    u = np.linalg.svd(C, full_matrices=False)[2][0]
    E = {k: g[k][0] @ u for k in ORDER}
    if E["c_home"].mean() > E["c_far"].mean():
        E = {k: -v for k, v in E.items()}
    Fa = {k: np.log(g[k][1][g[k][1] > 0]) for k in ORDER}
    ntot = {k: len(g[k][1]) for k in ORDER}
    def standardise(V):
        sd = np.sqrt(np.mean([np.var(V[k]) for k in ORDER if len(V[k]) > 1]))
        c = V["c_home"].mean()
        return {k: (V[k] - c) / sd for k in ORDER}
    E, Fa = standardise(E), standardise(Fa)
    lo_ = min(np.percentile(np.concatenate([V[k] for k in ORDER]), 0.5) for V in (E, Fa))
    hi_ = max(np.percentile(np.concatenate([V[k] for k in ORDER]), 99.5) for V in (E, Fa))
    xs_ = np.linspace(lo_, hi_, 400)
    dens = {(nm_, k): (100 * gaussian_kde(V[k], bw_method=0.15)(xs_) * len(V[k]) / ntot[k]
                       if len(V[k]) > 5 and np.ptp(V[k]) > 0 else np.zeros_like(xs_))
            for nm_, V in (("E", E), ("F", Fa)) for k in ORDER}
    hgap = 0.30
    hh = (HS - hgap) / 2

    def curves(ax, nm, V):
        for k in ORDER:
            conc = k.startswith("c")
            ax.fill_between(xs_, dens[nm, k], color=COL[k], alpha=0.30 if conc else 0.22, lw=0)
            ax.plot(xs_, dens[nm, k], color=COL[k], lw=1.1 if conc else 0.8)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])

    def centroids(ax, V):
        for k in ("c_home", "c_far"):
            ax.plot([V[k].mean()], [0], marker="^", ms=5, color=COL[k], mec="white", mew=0.4,
                    clip_on=False, zorder=6, transform=ax.get_xaxis_transform())

    # top: embedding
    ax = box(XS[2], y0 + hh + hgap, WS, hh)
    curves(ax, "E", E)
    ax.set_xlim(lo_, hi_); ax.set_ylim(0, max(dens["E", k].max() for k in ORDER) * 1.15)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("embedding PC 1", fontsize=FS_TICK, labelpad=4)
    centroids(ax, E)

    # bottom: family. Silent pixels have no log activation, so they are drawn as a narrow spike left of
    # the active range. That way each group's density adds up to 100%. The axis extends left to fit it.
    XSIL, EXT, SIG = 0.45, 0.80, 0.04                     # spike's distance left of lo_, axis extension, spike width
    ax = box(XS[2] - EXT * WS / (hi_ - lo_), y0, WS + EXT * WS / (hi_ - lo_), hh)
    xmin = lo_ - EXT
    xg = np.linspace(xmin, hi_, 3000)
    xs_sil = lo_ - XSIL
    full = {}
    for k in ORDER:
        act = (100 * gaussian_kde(Fa[k], bw_method=0.15)(xg) * len(Fa[k]) / ntot[k]
               if len(Fa[k]) > 5 and np.ptp(Fa[k]) > 0 else np.zeros_like(xg))
        sil = 100 * (1 - len(Fa[k]) / ntot[k]) * np.exp(-0.5 * ((xg - xs_sil) / SIG) ** 2) / (SIG * np.sqrt(2 * np.pi))
        full[k] = act + sil
    for k in ORDER:
        conc = k.startswith("c")
        ax.fill_between(xg, full[k], color=COL[k], alpha=0.30 if conc else 0.22, lw=0)
        ax.plot(xg, full[k], color=COL[k], lw=1.1 if conc else 0.8)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xmin, hi_); ax.set_ylim(0, max(full[k].max() for k in ORDER) * 1.08)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    tr = ax.get_xaxis_transform()
    lab_y = -(4 / 72) / hh
    ax.text((lo_ + hi_) / 2, lab_y, "family activation (log scale)", transform=tr, ha="center", va="top",
            fontsize=FS_TICK)
    f0 = (lo_ - xmin) / (hi_ - xmin)
    ins = ax.inset_axes([f0, 0.30, 1 - f0, 0.70])
    curves(ins, "F", Fa)
    ins.set_xlim(lo_, hi_); ins.set_ylim(0, max(dens["F", k].max() for k in ORDER) * 1.18)
    for sp in ins.spines.values():
        sp.set_linewidth(0.4); sp.set_color("#9a9a9a")
    ins.text(0.02, 0.95, "active pixels", transform=ins.transAxes, ha="left", va="top",
             fontsize=FS_TICK - 0.8, color=INK2)
    centroids(ins, Fa)
    letter(XS[2] - 0.45, ytop + 0.20, "cfi"[r])
    key = row["name"].split()[-1]
    fs = row["full"]                                    # AUCs on every sampled pixel, not the plotted subsample
    a_home = auc(fs["c_home"], fs["b_home"]); a_far = auc(fs["c_far"], fs["b_far"])
    NUM.put(f"figS3.{key}.auc_learned", round(a_home, 3), f"{H}: {c_} vs {b_}")
    NUM.put(f"figS3.{key}.auc_new", round(a_far, 3), f"{F}: {c_} vs {b_}")
    NUM.put(f"figS3.{key}.gap_concept_embedding_sd", round(float(abs(E["c_far"].mean() - E["c_home"].mean())), 2),
            "learned vs new concept centroid, in group widths, on PC 1 of the four group centroids")
    NUM.put(f"figS3.{key}.gap_concept_family_sd", round(float(abs(Fa["c_far"].mean() - Fa["c_home"].mean())), 2),
            "learned vs new concept centroid, in group widths, log family activation over active pixels")
    for k_, nm_ in (("c_home", "concept_learned"), ("c_far", "concept_new"), ("b_home", "surroundings_learned"), ("b_far", "surroundings_new")):
        NUM.put(f"figS3.{key}.silent.{nm_}", round(float((fs[k_] == 0).mean()), 3))
    for nm_, v_ in row["centre"].items():
        NUM.put(f"figS3.{key}.embedding_full.{nm_}", round(v_, 2),
                "all 128 dimensions, center distance over pooled within-group spread")
    print(f"{row['name']:13s} AUC {H} {a_home:.2f} {F} {a_far:.2f};  concept gap learned<->new: embedding {abs(E['c_far'].mean()-E['c_home'].mean()):.2f} sd, "
          f"family {abs(Fa['c_far'].mean()-Fa['c_home'].mean()):.2f} sd;  new concept vs new surroundings: "
          f"embedding {abs(E['c_far'].mean()-E['b_far'].mean()):.2f}, family {abs(Fa['c_far'].mean()-Fa['b_far'].mean()):.2f}")

cax = box(XS[1] + 0.45, 0.06, WS - 0.9, 0.06)
cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_ticks([]); cb.outline.set_linewidth(0.4)
fig.text((XS[1] + 0.40) / FIGW, 0.09 / FIGH, "family activation:  low", ha="right", va="center", fontsize=FS_TICK - 0.4)
fig.text((XS[1] + WS - 0.40) / FIGW, 0.09 / FIGH, "high", ha="left", va="center", fontsize=FS_TICK - 0.4)
NUM.save()
out = FIG_DIR / "figS3_invariance.png"
fig.savefig(out, dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print("wrote", out, fig.get_size_inches())
