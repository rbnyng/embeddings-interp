"""Test the unchanged burn family (lib.config.BURN_FAMILY) on European fire, harvest and windthrow, through the Dixie
dictionary.

Units are tile-years: the disturbed pixels against the tile's undisturbed forest.
  European fire  EFDA fire tile-years in the Europe-atlas tiles
  Harvest        EFDA harvest tile-years in three European boxes
  Windthrow      the held-out Friederike, Xavier and Vaia tiles of the windthrow setting
For each unit it finds the peak of the family's mean contrast over 2017-2025 and compares it with 200 random feature
sets of the same size: the peak's percentile, whether it is above their 95th percentile, and whether it falls in the
event year or the year after.

Usage: python experiments/specificity.py encode|table
  encode encodes the windthrow tiles with the Dixie dictionary. table computes the statistics.
Writes results/windthrow_dixie.jsonl, results/specificity.json and paper/si_specificity_table.tex.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from lib.config import BURN_FAMILY, RESULTS_DIR, REPO_ROOT, workers
from lib.sae import load_sae, encode_array
from lib.tiles import load_tile
import experiments.windthrow as ws

FAM = list(BURN_FAMILY)
YEARS = list(range(2017, 2026))
STORM_DIXIE = RESULTS_DIR / "windthrow_dixie.jsonl"
OUT = RESULTS_DIR / "specificity.json"
TEX = REPO_ROOT / "paper" / "si_specificity_table.tex"
DRAWS = 200

def _encode(rec):
    torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    try:
        dmg, intact = ws.tile_masks(rec)
        sae = load_sae()
        td, ti = {}, {}
        for y in YEARS:
            x = load_tile(rec["tile"], y, stride=ws.S)
            H, W, _ = x.shape
            f = encode_array(sae, x)
            td[str(y)] = f[dmg[:H, :W]].mean(0).tolist()
            ti[str(y)] = f[intact[:H, :W]].mean(0).tolist()
        return dict(ok=True, tile=rec["tile"], storm=rec["storm"], dmg=td, ref=ti)
    except Exception as e:
        return dict(ok=False, tile=rec["tile"], storm=rec["storm"], error=repr(e))

def phase_encode():
    with open(STORM_DIXIE, "w") as out, ProcessPoolExecutor(max_workers=workers(40)) as ex:
        for r in ex.map(_encode, ws.SEL):
            out.write(json.dumps(r) + "\n")
    print("wrote", STORM_DIXIE.name)

def _units():
    """Collect each group's units. Returns {group: [(event_year, {year: disturbed minus undisturbed mean activation,
    shape (M,)})]}."""
    U = {"European fire": [], "Harvest": [], "Windthrow": []}
    for r in map(json.loads, open(RESULTS_DIR / "europe_fire_dixie.jsonl")):
        if not r.get("ok"):
            continue
        T = r["tables"]
        for g in T["2018"]:
            if g.startswith("fire_"):
                U["European fire"].append((int(g[5:]), {int(y): np.array(T[y][g]) - np.array(T[y]["intact"])
                                                        for y in T if g in T[y]}))
    for r in json.load(open(RESULTS_DIR / "europe_agents_dixie.json")):
        T = r["tables"]
        for g in sorted(set().union(*[set(v) for v in T.values()])):
            if g.startswith("harvest_"):
                U["Harvest"].append((int(g[8:]), {int(y): np.array(T[y][g]) - np.array(T[y]["intact"])
                                                  for y in T if g in T[y]}))
    held = {r["tile"] for r in map(json.loads, open(ws.JSONL)) if r.get("ok") and not r.get("screen")}
    for r in map(json.loads, open(STORM_DIXIE)):
        if r.get("ok") and r["tile"] in held:
            U["Windthrow"].append((ws.EVENT_YEAR[r["storm"]],
                                   {int(y): np.array(r["dmg"][y]) - np.array(r["ref"][y]) for y in r["dmg"]}))
    return U

def phase_table():
    rng = np.random.default_rng(0)
    null = [rng.choice(1024, len(FAM), replace=False) for _ in range(DRAWS)]
    out = {}
    for name, units in _units().items():
        rows = []
        for Y, C in units:
            ys = sorted(C)
            A = np.stack([C[y] for y in ys])                     # (years, M)
            tr = A[:, FAM].mean(1)
            pk = float(tr.max()); off = ys[int(tr.argmax())] - Y
            nl = np.array([A[:, f].mean(1).max() for f in null])
            rows.append(dict(peak=pk, null_med=float(np.median(nl)), pct=float((nl < pk).mean()),
                             above95=bool(pk > np.percentile(nl, 95)), window=bool(off in (0, 1))))
        pk = np.array([r["peak"] for r in rows])
        out[name] = dict(n=len(rows), peak_median=float(np.median(pk)), peak_max=float(pk.max()),
                         null_median=float(np.median([r["null_med"] for r in rows])),
                         pct_median=float(np.median([r["pct"] for r in rows])),
                         above95=float(np.mean([r["above95"] for r in rows])),
                         window=float(np.mean([r["window"] for r in rows])))
        o = out[name]
        print(f"{name:14s} n={o['n']:4d}  peak {o['peak_median']:.3f} ({o['peak_max']:.2f})  null {o['null_median']:.3f}  "
              f"pct {o['pct_median']:.2f}  >p95 {o['above95']:.2f}  window {o['window']:.2f}")
    json.dump(out, open(OUT, "w"), indent=1)
    cols = list(out)
    pct = lambda v: f"{round(100 * v)}\\%"
    ordn = lambda v: (lambda k: f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}")(round(100 * v))
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{\textbf{Western US burn family on European disturbances.} The unchanged " + f"{len(FAM)}" + r"-feature family, scored through the Dixie dictionary. Each unit is a tile-year contrasted against the tile's undisturbed forest; percentiles are relative to 200 random " + f"{len(FAM)}" + r"-feature sets from the same dictionary, per unit. Window is the share peaking in $\{Y, Y{+}1\}$. Windthrow is the corpus windthrow set (held-out Friederike, Xavier and Vaia tiles).}",
             r"\label{tab:specificity}", r"\small", r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(cols) + r" \\", r"\midrule",
             "Units & " + " & ".join(f"{out[c]['n']:,}".replace(",", "{,}") for c in cols) + r" \\",
             "Peak contrast, median (max) & " + " & ".join(f"{out[c]['peak_median']:.3f} ({out[c]['peak_max']:.2f})" for c in cols) + r" \\",
             "Random-set peak, median & " + " & ".join(f"{out[c]['null_median']:.3f}" for c in cols) + r" \\",
             "Percentile vs random, median & " + " & ".join(ordn(out[c]['pct_median']) for c in cols) + r" \\",
             "Above random 95th percentile & " + " & ".join(pct(out[c]['above95']) for c in cols) + r" \\",
             "Peaks in window & " + " & ".join(pct(out[c]['window']) for c in cols) + r" \\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TEX.parent.mkdir(exist_ok=True)
    TEX.write_text("\n".join(lines) + "\n")
    print("wrote", TEX)

if __name__ == "__main__":
    {"encode": phase_encode, "table": phase_table}[sys.argv[1]]()
