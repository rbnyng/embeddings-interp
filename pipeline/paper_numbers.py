"""Collect every number the manuscript quotes, each with its source file and a note.

Most numbers are read from the results files, including the figN_numbers.json that each figure script writes. So
run this after the figures. Some are computed here, such as the overlaps between the Europe families, tile-bootstrap
intervals on the Europe and Amazon arc rates, and the harvest latitude split. The split compares harvest units north
of 50 degrees with those south of 42 degrees, on the share of dated units that peak the year after the event.

Usage: python pipeline/paper_numbers.py
Writes results/paper_numbers.json and paper/numbers.tex, which defines a LaTeX macro for each number.
"""
import itertools
import json

import numpy as np
import pandas as pd

from lib.config import RESULTS_DIR
from lib.stats import rate, tile_bootstrap

FIGURES = ("fig1", "fig4", "fig5", "fig6", "figS1", "figS2", "figS3")

N = {}

def put(key, value, source, note=""):
    N[key] = dict(value=value, source=source, note=note)

def load(name):
    return json.load(open(RESULTS_DIR / name))

# ---- the corpus ------------------------------------------------------------
df = pd.read_csv(RESULTS_DIR / "paper_events.csv")
dist = df[df.kind != "management"]
mgmt = df[df.kind == "management"]
put("corpus.units", int(len(df)), "paper_events.csv", "all reference units")
put("corpus.validated", int(df.ok.sum()), "paper_events.csv")
put("corpus.rate", round(float(df.ok.mean()), 4), "paper_events.csv")
put("corpus.disturbance_units", int(len(dist)), "paper_events.csv")
put("corpus.disturbance_validated", int(dist.ok.sum()), "paper_events.csv")
put("corpus.disturbance_rate", round(float(dist.ok.mean()), 4), "paper_events.csv",
    "the dated settings only; crop settings are classified, not dated")
put("corpus.management_units", int(len(mgmt)), "paper_events.csv")
put("corpus.management_rate", round(float(mgmt.ok.mean()), 4), "paper_events.csv")
put("corpus.settings", int(df.region.nunique()), "paper_events.csv")
for r, g in df.groupby("region"):
    put(f"region.{r}.n", int(len(g)), "paper_events.csv")
    if g.ok.notna().any():
        put(f"region.{r}.rate", round(float(g.ok.mean()), 4), "paper_events.csv")
        put(f"region.{r}.validated", int(g.ok.sum()), "paper_events.csv")
    else:
        # no unit of the setting has a peak-year outcome
        put(f"region.{r}.rate", None, "crop_atlas.json",
            "no per-unit outcome: quote crop.macro_mean and the crop.* classes")

# ---- nulls -----------------------------------------------------------------
an = load("all_nulls.json")
for k, v in an.items():
    if k.startswith("_"):
        continue
    put(f"null.{k}.observed", round(v["observed"], 4), "all_nulls.json")
    put(f"null.{k}.median", round(v["null_median"], 4), "all_nulls.json")
    put(f"null.{k}.ci", [round(v["null_lo"], 4), round(v["null_hi"], 4)],
        "all_nulls.json")
    put(f"null.{k}.p", round(v["p"], 4), "all_nulls.json")
    put(f"null.{k}.obs_ci", [round(v["obs_lo"], 4), round(v["obs_hi"], 4)], "all_nulls.json",
        "95% interval on the observed rate (tile bootstrap; Wilson where every unit or none is dated)")
    put(f"null.{k}.n", int(v["n"]), "all_nulls.json")
put("null.settings_with_null", len([k for k in an if not k.startswith("_")]),
    "all_nulls.json")
put("null.no_null", list(an.get("_not_computable", {})), "all_nulls.json",
    "crop settings are scored by held-out accuracy, not a peak offset")

# ---- fig 2: feature vs axis ------------------------------------------------
f2 = load("fig2_data.json")
put("fig2.corn_feature", int(f2["corn"]["f"]), "fig2_data.json")
put("fig2.soy_feature", int(f2["soy"]["f"]), "fig2_data.json")
for k, v in f2["stability"].items():
    put(f"fig2.{k}", round(v, 4), "fig2_data.json", "F1, selected by F1 not purity")
put("fig2.corn_base_rate", round(f2["base"][0], 4), "fig2_data.json")
put("fig2.polysemantic_dim", int(f2["poly"]["d"]), "fig2_data.json",
    f"top tail -> {f2['poly']['top_cls']}, bottom tail -> {f2['poly']['bot_cls']}")
for k, v in f2["swap"].items():
    put(f"fig2.swap.{k}", round(v, 4) if isinstance(v, float) else int(v),
        "fig2_data.json")

# ---- atlases --------------------------------------------------------------
eu = load("europe_atlas.json")
for a in eu["agents"]:
    put(f"atlas.europe.{a}.rate", round(eu["agents"][a]["rate"], 4),
        "europe_atlas.json")
    put(f"atlas.europe.{a}.n", int(eu["agents"][a]["n"]), "europe_atlas.json")
    put(f"atlas.europe.{a}.family_size", len(eu["agents"][a]["family"]),
        "europe_atlas.json")
arc = load("amazon_arc.json")
put("atlas.arc.rate", round(arc["pooled"], 4), "amazon_arc.json")
put("atlas.arc.n", int(arc["n"]), "amazon_arc.json")
put("atlas.arc.y0_share", round(arc["y0_share"], 4), "amazon_arc.json")

# ---- family overlap and the latitude split ---------------------------------
fams = {a: set(eu["agents"][a]["family"]) for a in eu["agents"]}
for a, b in itertools.combinations(sorted(fams), 2):
    A, B = fams[a], fams[b]
    inter = len(A & B)
    put(f"overlap.{a}_vs_{b}.jaccard", round(inter / len(A | B), 4),
        "europe_atlas.json (recomputed)")
    put(f"overlap.{a}_vs_{b}.coefficient", round(inter / min(len(A), len(B)), 4),
        "europe_atlas.json (recomputed)",
        f"{inter} of {min(len(A), len(B))} shared; Jaccard's ceiling here is "
        f"{min(len(A), len(B))}/{max(len(A), len(B))}")

# Among harvest units dated in the expected window, the share that peak the year after the event.
# Compares north (lat 50 and above) with south (lat below 42), leaving out the middle.
V = eu["outcomes"]
LAT_N, LAT_S = 50.0, 42.0

def y1_share(recs):
    return float(np.mean([r["off"] == 1 for r in recs])) if recs else np.nan

dated = [r for r in V if r["ok"]]
harvest = [r for r in dated if r["agent"] == "harvest"
           and (r["lat"] >= LAT_N or r["lat"] < LAT_S)]
north = [r for r in harvest if r["lat"] >= LAT_N]
south = [r for r in harvest if r["lat"] < LAT_S]
put("latitude.harvest_north.y1", round(y1_share(north), 4),
    "europe_atlas.json (recomputed)", f"lat>=50, n={len(north)} dated units")
put("latitude.harvest_south.y1", round(y1_share(south), 4),
    "europe_atlas.json (recomputed)", f"lat<42, n={len(south)} dated units")
put("latitude.n_tiles_north", len({r["tile"] for r in north}),
    "europe_atlas.json (recomputed)")
put("latitude.n_tiles_south", len({r["tile"] for r in south}),
    "europe_atlas.json (recomputed)")

def _diff(recs):
    n = [r for r in recs if r["lat"] >= LAT_N]
    s = [r for r in recs if r["lat"] < LAT_S]
    if not n or not s:
        return np.nan
    return y1_share(n) - y1_share(s)

d_obs, d_lo, d_hi = tile_bootstrap(harvest, _diff)
put("latitude.harvest_diff", round(d_obs, 4), "europe_atlas.json (recomputed)",
    "north minus south Y+1 share, harvest only (controls agent mix)")
put("latitude.harvest_diff_ci", [round(d_lo, 4), round(d_hi, 4)],
    "europe_atlas.json (recomputed)",
    "tile-clustered bootstrap, 10,000 draws")

for a in sorted(fams):
    recs = [r for r in V if r["agent"] == a]
    o, lo, hi = tile_bootstrap(recs, rate)
    put(f"atlas.europe.{a}.rate_ci", [round(lo, 4), round(hi, 4)],
        "europe_atlas.json (recomputed)", "tile-clustered, 10,000 draws")

arc_v = arc["outcomes"]
o, lo, hi = tile_bootstrap(arc_v, rate)
put("atlas.arc.rate_ci", [round(lo, 4), round(hi, 4)],
    "amazon_arc.json (recomputed)", "tile-clustered, 10,000 draws")

# ---- invariance table: location R^2 among family-active pixels (rice and wheat by tile, fire by location) ----
for k, fn, sub in (("rice", "rice_invariance.json", None), ("wheat", "wheat_invariance.json", None),
                   ("fire", "fire_invariance.json", "100")):
    v = load(fn)["location_r2_raw"]
    put(f"invariance.{k}.location_r2", round((v[sub] if sub else v)["r2"], 4), fn,
        "share of raw embedding variance among family-active pixels explained by location"
        + (", 100 pixels per location" if sub else ""))

# ---- negative results and crops ---------------------------------------------
ra = load("amazon_arc_radd.json")
put("neg.subannual.diff", round(ra["subannual_diff"], 4), "amazon_arc_radd.json",
    "early-late P(peak=Y) difference; the decisive sub-annual null")
put("neg.subannual.ci", [round(x, 4) for x in ra["subannual_ci"]], "amazon_arc_radd.json")
put("neg.subannual.n_plots", int(ra["n_plots"]), "amazon_arc_radd.json")
put("neg.perplot.by_size", {k: {"n": v["n"], "exact": round(v["exact"], 3),
                                "window": round(v["window"], 3)}
                            for k, v in ra["size_bands"].items()},
    "amazon_arc_radd.json", "quote as a size curve, not one average")
pr = load("precursor.json")
put("neg.precursor.y2_median", round(pr["q2_median"], 4), "precursor.json")
put("neg.precursor.y2_ci", [round(x, 4) for x in pr["q2_ci"]], "precursor.json",
    "tile-clustered; includes zero")
put("neg.precursor.positive_share", round(pr["q2"], 4), "precursor.json",
    "share of held-out units with a positive Y-2 contrast")
put("neg.precursor.depth_profile",
    {k: round(v, 4) for k, v in pr["q5"].items()}, "precursor.json",
    "mean contrast at Y-4, Y-3 and Y-2")
rxa = load("se_us_fire_aef.json")
put("neg.aef_blind.family_size", len(rxa["family"]), "se_us_fire_aef.json",
    "zero features from the identical screen")
put("neg.aef_blind.raw_ratio",
    round(load("raw_retention.json")["classes"]["SE US wildfire | AlphaEarth"]["snr_med"], 4),
    "raw_retention.json",
    "dictionary-free 64-d control: in-window/pre-event median, on the full "
    "event set rather than the 10 the hand-made se_us_fire_aef.json used")
put("neg.aef_blind.n_events", int(rxa["n_screen_events"]), "se_us_fire_aef.json")
ca = load("crop_atlas.json")
put("crop.cornsoy_heldout", round(ca["c2"]["mean"], 4), "crop_atlas.json",
    "multiclass held-out, corn/soy mean")
put("crop.corn", round(ca["c2"]["corn"], 4), "crop_atlas.json")
put("crop.soy", round(ca["c2"]["soy"], 4), "crop_atlas.json")
for sys_ in ("cornsoy", "winter_wheat", "rice", "cotton", "perennial", "fallow"):
    oks = [v["ok"] for v in ca["tiles"].values() if sys_ in v["systems"] and v["ok"] is not None]
    put(f"crop.tiles.{sys_}.n", len(oks), "crop_atlas.json")
    put(f"crop.tiles.{sys_}.pass", int(sum(oks)), "crop_atlas.json",
        "tiles at or above crop_atlas.TILE_PASS held-out accuracy on their own classes")
put("crop.winter_wheat", round(ca["c3_winter_wheat"], 4), "crop_atlas.json")
put("crop.rice", round(ca["c3_rice"], 4), "crop_atlas.json")
put("crop.cotton", round(ca["c3_cotton"], 4), "crop_atlas.json")
put("crop.macro_mean",
    round((ca["c2"]["corn"] + ca["c2"]["soy"] + ca["c3_winter_wheat"]
           + ca["c3_rice"] + ca["c3_cotton"]) / 5, 4), "crop_atlas.json",
    "Table S1 cell: unweighted mean over the five classes, so corn/soy "
    "pixel dominance cannot bury the minors")
put("crop.flip_perennial", round(ca["c4"]["perennial"], 4), "crop_atlas.json",
    "label-flip rate where ground truth says nothing changed")
put("crop.flip_rotation", round(ca["c4"]["rotation"], 4), "crop_atlas.json",
    "label-flip rate where ground truth says the crop rotated")

# ---- western US fire: the rate on the null's 207 units and on all 242 mapped fires ----
# 35 fires have too few pixels at the screening stride to give a contrast per feature. A rate compared with a
# null must use the null's units.
put("western_us.on_null_units.rate", round(an["Western US fire"]["observed"], 4),
    "all_nulls.json", "the units the null is scored on")
put("western_us.on_null_units.n", int(an["Western US fire"]["n"]), "all_nulls.json")
put("western_us.all_mapped.rate", round(float(df[df.region == "Western US fire"].ok.mean()), 4),
    "paper_events.csv", "every mapped fire, including those with no per-feature contrast")
put("western_us.all_mapped.n", int((df.region == "Western US fire").sum()),
    "paper_events.csv", "every mapped fire, as on the Fig 3 map")

# ---- specificity: the western US family on European disturbances (SI table) -
for name, o in load("specificity.json").items():
    for k in ("n", "pct_median", "above95", "window"):
        put(f"spec.{name}.{k}", o[k], "specificity.json")

# ---- seed robustness of the western US family ------------------------------
sd = load("dixie_seeds.json")
seeds = list(sd["seeds"].values())
others = [v for k, v in sd["seeds"].items() if k != "0"]
put("seeds.n", len(seeds), "dixie_seeds.json")
put("seeds.family_min", min(v["family_size"] for v in seeds), "dixie_seeds.json")
put("seeds.family_max", max(v["family_size"] for v in seeds), "dixie_seeds.json")
put("seeds.rate_min", min(v["rate"] for v in seeds), "dixie_seeds.json")
put("seeds.rate_max", max(v["rate"] for v in seeds), "dixie_seeds.json")
put("seeds.null_min", min(v["null_median"] for v in seeds), "dixie_seeds.json")
put("seeds.null_max", max(v["null_median"] for v in seeds), "dixie_seeds.json")
put("seeds.feature_match_median", float(np.median([v["match_cos_median"] for v in others])),
    "dixie_seeds.json", "median over seeds 1-4 of each family's median best-match cosine to seed 0")
put("seeds.direction_min", min(v["family_direction_cos"] for v in others), "dixie_seeds.json")
put("seeds.direction_max", max(v["family_direction_cos"] for v in others), "dixie_seeds.json")
put("seeds.dixie_rate", sd["dixie"]["rate"], "dixie_seeds.json")

# ---- detection settings (SI): mining land and solar farms, scored by lib.detect ----
for key, fn in (("mines", "mines.json"), ("solar", "solar.json")):
    d = load(fn)
    v = d["outcomes"]
    src = f"{fn} (experiments/{key}.py)"
    put(f"detect.{key}.n", int(d["n_held"]), src, "held-out sites/tiles")
    put(f"detect.{key}.n_screen", int(d["n_screen"]), src)
    put(f"detect.{key}.family_size", len(d["family"]), src)
    put(f"detect.{key}.rate", round(float(d["rate"]), 4), src, "share validating at balanced accuracy >= 0.70")
    put(f"detect.{key}.null_median", round(float(d["null"]["median"]), 4), src, "200 random families of the same size")
    put(f"detect.{key}.null_p95", round(float(d["null"]["p95"]), 4), src)
    put(f"detect.{key}.acc_median", round(float(np.median([x["acc"] for x in v])), 4), src, "median balanced accuracy")
    put(f"detect.{key}.auc_median", round(float(np.median([x["auc"] for x in v])), 4), src, "median family AUC")
    put(f"detect.{key}.xregion.rate", round(float(d["cross_region"]["rate"]), 4), src, "screened on some regions, scored on the rest")
    put(f"detect.{key}.xregion.n", int(d["cross_region"]["n"]), src)

# ---- numbers each figure computed ------------------------------------------
for fig in FIGURES:
    for k, v in load(f"{fig}_numbers.json").items():
        assert k not in N, f"{k} is written both here and by {fig}"
        put(k, v["value"], f"{fig}_numbers.json", v["note"])

# Derived from the figure numbers: AlphaEarth's retention two years on, divided by Tessera's
for s in ("Western US fire", "SE US wildfire", "SE US prescribed fire"):
    put(f"fig6.persist.ratio.{s}", round(N[f"fig6.persist.aef.{s}"]["value"]
                                        / N[f"fig6.persist.tessera.{s}"]["value"], 4),
        "fig6_numbers.json", "AlphaEarth / Tessera")
put("fig5.ablation.burn.after", round(N["fig5.ablation.burn.clean"]["value"]
                                    + N["fig5.ablation.burn.drop_family"]["value"], 4),
    "fig5_numbers.json", "burn probe accuracy with the family ablated")
put("capacity.bar_multiple_largest_int", int(round(N["capacity.bar_multiple_largest"]["value"])), "figS2_numbers.json")
d = [N[k]["value"] for k in N if k.startswith("detection.")]
put("detection.p_y0_min", min(d), "fig4_numbers.json")
put("detection.p_y0_max", max(d), "fig4_numbers.json")

out = RESULTS_DIR / "paper_numbers.json"
json.dump(N, open(out, "w"), indent=1)


# ---- the same numbers as LaTeX macros, keyed by the JSON key ----------------
# \pn{key}        the value. Floats get 3 decimals, integers get thousands separators.
# \pnpct{key}     100 times the value, to one decimal, with \%.
# \pnpcti{key}    100 times the value, rounded, with \%.
# \pnpts{key}     100 times the absolute value, rounded, for changes in percentage points.
# \pntwo, \pnone  the value to two or one decimals.
# \pnord{key}     100 times the value as an ordinal (87th), for percentiles.
# Dict entries are addressed as key.field, and list entries as key.0, key.1. An unknown key stops the LaTeX build.
def _scalars(key, v):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _scalars(f"{key}.{k}", x)
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            yield from _scalars(f"{key}.{i}", x)
    elif isinstance(v, (bool, str)) or v is None:
        return
    else:
        yield key, v

def _int(x):
    return f"{int(x):,}".replace(",", "{,}")

def _forms(v):
    if isinstance(v, (int, np.integer)):
        yield "", _int(v)
    else:
        yield "", f"{v:.3f}"
        yield "pct@", f"{100 * v:.1f}\\%"
        yield "pcti@", f"{round(100 * v)}\\%"
        yield "pts@", f"{round(abs(100 * v))}"
        yield "two@", f"{v:.2f}"
        yield "one@", f"{v:.1f}"
        k = round(100 * v)
        yield "ord@", f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"

lines = [r"% generated by pipeline/paper_numbers.py from results/paper_numbers.json; do not edit",
         r"\makeatletter",
         r"\newcommand{\pn@get}[2]{\ifcsname pn@#1#2\endcsname\csname pn@#1#2\endcsname"
         r"\else\errmessage{paper number '#2' (#1) is not in paper_numbers.json}\fi}",
         r"\newcommand{\pn}[1]{\pn@get{}{#1}}",
         r"\newcommand{\pnpct}[1]{\pn@get{pct@}{#1}}",
         r"\newcommand{\pnpcti}[1]{\pn@get{pcti@}{#1}}",
         r"\newcommand{\pnpts}[1]{\pn@get{pts@}{#1}}",
         r"\newcommand{\pntwo}[1]{\pn@get{two@}{#1}}",
         r"\newcommand{\pnone}[1]{\pn@get{one@}{#1}}",
         r"\newcommand{\pnord}[1]{\pn@get{ord@}{#1}}"]
for key in sorted(N):
    for k, v in _scalars(key, N[key]["value"]):
        for form, txt in _forms(v):
            lines.append(r"\expandafter\def\csname pn@" + form + k + r"\endcsname{" + txt + "}")
lines.append(r"\makeatother")
TEX = RESULTS_DIR.parent / "paper" / "numbers.tex"
TEX.parent.mkdir(exist_ok=True)
TEX.write_text("\n".join(lines) + "\n")
print(f"wrote {TEX}")
print(f"{len(N)} numbers -> {out}\n")
for k in sorted(N):
    v = N[k]
    print(f"  {k:<46} {str(v['value']):<22} [{v['source']}]")
