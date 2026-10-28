"""Figure 5: what a linear probe loses when the burn family is removed, and which distinctions the families carry.

a  change in linear-probe accuracy on three Dixie tasks when we remove the burn family, its strongest single feature
   (F566) or a random set of the same size
b  SE US wildfires and prescribed burns, including those beyond the cap of 100 per type, ranked by the wildfire
   family's contrast and cut into tenths: share of wildfires in each tenth
c  European tile-years with both windthrow and harvest, ranked by wind-only minus harvest-only feature activation and
   cut into tenths: share of windthrow in each tenth

Writes figures/fig5_boundaries.png and .pdf, and results/fig5_numbers.json.
"""
import json

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.config import FIG_DIR, RESULTS_DIR
from lib.numbers import FigureNumbers
from lib.style import js

INK, INK2, FAINT = "#111111", "#555555", "#c9c6bc"
FAMC, RANDC, SOLOC = "#c1440e", "#8a8f94", "#e39b6b"
# drawn 7.9 in wide and printed at 183 mm (scaled by 0.912)
FS_L, FS_LAB, FS_ANN, FS_TICK = 8.8, 7.6, 7.0, 6.6

def jsonl(name):
    return [r for r in map(json.loads, open(RESULTS_DIR / name)) if r.get("ok")]

ab = json.load(open(RESULTS_DIR / "ablation_results.json"))

# b: both fire types are scored with the wildfire family. Each event's score is the larger
# burned minus unburned contrast of the event year and the year after.
wf_fam = json.load(open(RESULTS_DIR / "se_us_fire.json"))["family"]

def contrast(r):
    t = {int(y): float(np.mean([r["burn"][y][f] for f in wf_fam])
                       - np.mean([r["ref"][y][f] for f in wf_fam]))
         for y in r["burn"]}
    return max(t.get(r["year"], 0), t.get(r["year"] + 1, 0))

fires = jsonl("se_us_fire.jsonl") + jsonl("se_us_fire_extra.jsonl")
sev = {kind: np.array([contrast(r) for r in fires if r["fire_type"] == kind])
       for kind in ("wildfire", "prescribed")}

# c: tile-years with both windthrow and harvest, so the two share a tile and year. Contrasts
# are disturbed pixels minus undisturbed (intact) forest on the same tile.
A = json.load(open(RESULTS_DIR / "europe_atlas.json"))["agents"]
W, H = set(A["wind_beetle"]["family"]), set(A["harvest"]["family"])
w_only, h_only = sorted(W - H), sorted(H - W)
cw_all, ch_all = [], []
for r in jsonl("europe_atlas.jsonl"):
    for y in range(2018, 2023):
        T = r["tables"].get(str(y), {})
        if f"wind_beetle_{y}" in T and f"harvest_{y}" in T:
            base = np.array(T["intact"])
            cw_all.append(np.array(T[f"wind_beetle_{y}"]) - base)
            ch_all.append(np.array(T[f"harvest_{y}"]) - base)
cw_all, ch_all = np.array(cw_all), np.array(ch_all)
# agent score: wind-only minus harvest-only activation. Above 0 reads as windthrow.
agent = {"windthrow": cw_all[:, w_only].mean(1) - cw_all[:, h_only].mean(1),
         "harvest": ch_all[:, w_only].mean(1) - ch_all[:, h_only].mean(1)}
wind_fam = {"windthrow": cw_all[:, sorted(W)].mean(1),
            "harvest": ch_all[:, sorted(W)].mean(1)}

js.use()
fig = plt.figure(figsize=(7.9, 4.8), dpi=200)
gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 1.0], hspace=0.62)
gc = gs[1, 0].subgridspec(1, 2, wspace=0.42)

def letter(ax, s, x0=-0.14, y0=1.06):
    ax.text(x0, y0, s, transform=ax.transAxes, fontsize=FS_L, fontweight="bold",
            color=INK, va="bottom")

def bare(ax, keep_left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, colors=INK)

# ---- a: ablation -------------------------------------------------------------
axa = fig.add_subplot(gs[0, 0]); letter(axa, "a", x0=-0.085, y0=1.16)
TASKS = [("burn history", "burn"), ("terrain aspect", "aspect"),
         ("tree vs grass", "tree_grass")]
yp = np.arange(len(TASKS))[::-1]
ABLATIONS = [("abl_F566", SOLOC, "^", 6.8, "strongest single feature removed"),
             ("abl_family", FAMC, "o", 8.5, "burn family removed"),
             ("abl_random", RANDC, "o", 7.5, "same-size random set removed")]
for yi, (nm, k) in zip(yp, TASKS):
    c = ab[k]["clean"]["acc"]
    dfam = 100 * (ab[k]["abl_family"]["acc"] - c)
    axa.plot([0, dfam], [yi, yi], "-", color=FAINT, lw=1.5, zorder=1)
    for j, (key, col, mk, ms, _) in enumerate(ABLATIONS):
        yj = yi + 0.20 - 0.20 * j
        r = ab[k][key]
        if key == "abl_random":
            axa.plot([100 * (r["acc_lo"] - c), 100 * (r["acc_hi"] - c)],
                     [yj, yj], "-", color=col, lw=2.6, alpha=0.45, zorder=2,
                     solid_capstyle="butt")
        axa.plot([100 * (r["acc"] - c)], [yj], mk, ms=ms,
                 color=col, mec="white", mew=0.9, zorder=3 + j)
axa.axvline(0, color=INK, lw=1.0, zorder=2)
axa.set_yticks(yp); axa.set_yticklabels([t for t, _ in TASKS])
axa.set_xlabel("Change in linear-probe accuracy (percentage points)",
               fontsize=FS_LAB)
axa.set_xlim(-21.5, 4.5)
axa.set_ylim(-0.55, len(TASKS) - 0.45)
bare(axa, keep_left=False)
axa.tick_params(left=False)
axa.legend(handles=[Line2D([], [], marker=mk, ls="", mfc=col, mec="white",
                           ms=7.5, label=lab)
                    for _, col, mk, _, lab in ABLATIONS],
           fontsize=FS_ANN, frameon=False, loc="lower left",
           bbox_to_anchor=(0.0, 1.02), ncol=3, handlelength=0.9,
           handletextpad=0.4, columnspacing=1.1, borderaxespad=0.0)

# ---- b, c: group mix in each tenth of the pooled ranking --------------------
def decile_mix(ax, first, second, cols, labels, noun, ranked_by,
               ends=("weakest\ntenth", "strongest\ntenth")):
    v = np.r_[first, second]
    lab = np.r_[np.ones(len(first)), np.zeros(len(second))]
    order = np.argsort(v, kind="stable")
    share = np.array([g.mean() for g in np.array_split(lab[order], 10)]) * 100
    x = np.arange(1, 11)
    ax.bar(x, share, width=0.82, color=cols[0], alpha=0.9, lw=0, zorder=2,
           label=labels[0])
    ax.bar(x, 100 - share, bottom=share, width=0.82, color=cols[1], alpha=0.9,
           lw=0, zorder=2, label=labels[1])
    ax.axhline(100 * len(first) / len(v), color="white", lw=0.9,
               ls=(0, (3, 2)), zorder=3)
    ax.set_xlim(0.4, 10.6); ax.set_ylim(0, 100)
    ax.set_xticks([1, 10]); ax.set_xticklabels(ends)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(f"% of {noun}", fontsize=FS_LAB)
    ax.set_xlabel(f"{noun[0].upper() + noun[1:]} ranked by {ranked_by}", fontsize=FS_LAB)
    ax.legend(fontsize=FS_TICK, frameon=False, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=0.9,
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
    bare(ax)
    ax.grid(axis="both", visible=False)
    ax.tick_params(axis="x", length=0)
    return share

axb = fig.add_subplot(gc[0]); letter(axb, "b", x0=-0.22, y0=1.10)
share_b = decile_mix(axb, sev["wildfire"], sev["prescribed"], ("#b3261e", "#1b7a4b"),
           ("wildfire", "prescribed burn"), "fires",
           "wildfire-family activation")

axc = fig.add_subplot(gc[1]); letter(axc, "c", x0=-0.22, y0=1.10)
share_c = decile_mix(axc, agent["windthrow"], agent["harvest"], ("#3b6ea8", "#d98a2b"),
           ("windthrow", "harvest"), "disturbed stands",
           "wind-only −\nharvest-only feature activation",
           ends=("most\nharvest-like", "most\nwindthrow-like"))

# ---- numbers the caption and text quote -------------------------------------
def best_cut(a, b):
    """Find the observed value that best separates a (above it) from b (at or below it). Returns (threshold, balanced
    accuracy)."""
    c = np.r_[a, b]
    acc = [(np.mean(a > t) + np.mean(b <= t)) / 2 for t in c]
    i = int(np.argmax(acc))
    return c[i], float(acc[i])

NUM = FigureNumbers("fig5")
for t in ("burn", "aspect", "tree_grass"):
    c = ab[t]["clean"]["acc"]
    NUM.put(f"fig5.ablation.{t}.clean", round(c, 4))
    NUM.put(f"fig5.ablation.{t}.drop_family", round(ab[t]["abl_family"]["acc"] - c, 4))
    NUM.put(f"fig5.ablation.{t}.drop_random", round(ab[t]["abl_random"]["acc"] - c, 4),
            "mean over 200 size-matched random draws from the alive pool")
    NUM.put(f"fig5.ablation.{t}.drop_random_ci",
            [round(ab[t]["abl_random"]["acc_lo"] - c, 4),
             round(ab[t]["abl_random"]["acc_hi"] - c, 4)],
            "95% interval over the 200 draws")
    NUM.put(f"fig5.ablation.{t}.drop_single", round(ab[t]["abl_F566"]["acc"] - c, 4),
            "strongest single family feature (F566) removed")
    NUM.put(f"fig5.ablation.{t}.retrained_on_ablated",
            round(ab[t]["retrained_on_ablated"]["acc"], 4),
            "probe refit on family-ablated embeddings: recovery means the family "
            "is what the linear readout uses, not the only place the information "
            "is kept")

w, p = sev["wildfire"], sev["prescribed"]
cut, acc = best_cut(w, p)
NUM.put("fig5.severity.n", {"wildfire": len(w), "prescribed": len(p)})
NUM.put("fig5.severity.balanced_acc", round(acc, 3),
        "best single activation threshold, wildfire vs prescribed; chance 0.5")
NUM.put("fig5.severity.cut", {"x_median_wildfire": round(float(cut / np.median(w)), 3),
                              "wildfire_above": round(float(np.mean(w > cut)), 3),
                              "prescribed_at_or_below": round(float(np.mean(p <= cut)), 3)},
        "that threshold, and the share of each fire type on its side")
NUM.put("fig5.severity.median_ratio", round(float(np.median(p) / np.median(w)), 4),
        "prescribed median / wildfire median")
NUM.put("fig5.severity.rx_median_wildfire_pctile",
        round(float(np.mean(w < np.median(p))), 4),
        "share of wildfires scoring below the median prescribed burn")
NUM.put("fig5.severity.nonpositive", {k: int(np.sum(v <= 0)) for k, v in sev.items()})
NUM.put("fig5.severity.tenths_pct_wildfire", [round(float(x), 1) for x in share_b],
        f"panel b bars, weakest to strongest; pooled share {100 * len(w) / (len(w) + len(p)):.1f}%")

NUM.put("fig5.agent.balanced_acc", round(best_cut(agent["windthrow"], agent["harvest"])[1], 3),
        "best single threshold on the wind-only minus harvest-only score; chance 0.5")
NUM.put("fig5.agent.family_balanced_acc",
        round(best_cut(wind_fam["windthrow"], wind_fam["harvest"])[1], 3),
        "same, on the whole wind/beetle family's strength: windthrow responds "
        "more strongly, a magnitude effect like 5b, not agent identity")
NUM.put("fig5.agent.family_median_ratio",
        round(float(np.median(wind_fam["harvest"]) / np.median(wind_fam["windthrow"])), 3),
        "harvest / windthrow median wind-family response")
NUM.put("fig5.agent.tenths_pct_windthrow", [round(float(x), 1) for x in share_c],
        "panel c bars, most harvest-like to most windthrow-like; pooled share 50%")

# on the same tile-year pairs, do the features found only in one agent's family favor that agent's disturbance?
NUM.put("fig5.identity.n_pairs", len(cw_all))
for k, own, other, only, fam in (("wind", cw_all, ch_all, w_only, W),
                                 ("harvest", ch_all, cw_all, h_only, H)):
    margin = own[:, only].mean(1) - other[:, only].mean(1)
    NUM.put(f"fig5.identity.{k}.median_pct",
            round(float(np.median(100 * margin / np.median(own[:, sorted(fam)].mean(1)))), 2),
            "% of family signal")
    NUM.put(f"fig5.identity.{k}.auc", round(float(np.mean(margin > 0)), 3),
            "paired AUC: share of same tile-year pairs where the exclusive "
            "features prefer their own disturbance; 0.5 = no preference")
    NUM.put(f"fig5.identity.{k}.n_features", len(only))
NUM.save()

out = FIG_DIR / "fig5_boundaries.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", out)
