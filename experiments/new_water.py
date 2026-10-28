"""New permanent water: find a family for land that becomes permanent water, and test it on held-out tiles.

Reference product: JRC Global Surface Water yearly history v1.4, detection-dated. A 30 m pixel has onset year Y
(2018-2021) when it is land in each of the three years before Y, water in Y and every year after up to 2021, and
permanent water in 2021.
Each unit is a 0.1 degree tile, dated to its onset year with the most new water. It needs at least 25 ha of new water
in that year and Tessera for Y-1, Y and Y+1. Up to 100 tiles are kept per onset year, largest first. The expected
window is Y or Y+1. Pixel groups use every second pixel, up to 5,000 per tile-year:
  disturbed    onset in Y, eroded by one 20 m pixel to drop the edges of the 30 m labels, at least 300 pixels
  undisturbed  never water in 2015-2021 and more than 60 m from any pixel that ever was, at least 2,000 pixels
One dictionary is trained on these tiles. The family is screened on the 2018 units and tested on the 2019-2021 units.
The screen drops years from onset that fewer than 3 screening units cover.

Usage: python -I experiments/new_water_labels.py candidates|onsets   (needs earthengine-api and $EE_PROJECT)
       python experiments/new_water.py select
       python -I experiments/new_water_labels.py tiles
       python experiments/new_water.py fetch|train|screen|verdicts
Writes results/new_water_selection.json, new_water.jsonl and new_water.json, and the dictionary checkpoint.
"""
import json
import sys
import zlib

import numpy as np

from lib.config import CKPT_DIR, DATA_DIR, GT_DIR, RESULTS_DIR
from lib.groundtruth import crop_to_grid
from lib.tiles import available_years
from lib.sae import load_sae, train_and_save
from lib.sampling import corpus, mean_activation, sample_tile_years
from lib.screen import aggregate_contrasts, anchored_family, peak_outcome, row_contrasts, row_units, outcome_summary

YEARS = list(range(2017, 2026))
S, WINDOW = 2, (0, 1)
CAP, N_TRAIN = 5000, 5000
MIN_HA, CAP_TILES, SCREEN_YEAR, MIN_UNITS = 25.0, 100, 2018, 3
LAB = GT_DIR / "new_water"
PIX = DATA_DIR / "intermediate" / "new_water"
CKPT = CKPT_DIR / "new_water" / "sae_new_water.pt"
SEL_PATH = RESULTS_DIR / "new_water_selection.json"
JSONL = RESULTS_DIR / "new_water.jsonl"
SEL = json.load(open(SEL_PATH)) if SEL_PATH.exists() else []


def _npz(rec):
    return PIX / f"{rec['tile']}.npz"


def phase_select():
    z = np.load(LAB / "survey.npz")
    best = {}
    for y, t, h in zip(z["year"], z["tile"], z["ha"]):
        if h > best.get(str(t), (0, 0.0))[1]:
            best[str(t)] = (int(y), float(h))
    out = []
    for Y in sorted({y for y, _ in best.values()}):
        cand = sorted([t for t, (y, h) in best.items() if y == Y and h >= MIN_HA], key=lambda t: -best[t][1])
        have = available_years(cand[:3 * CAP_TILES], YEARS)
        keep = [t for t in cand[:3 * CAP_TILES] if {Y - 1, Y, Y + 1} <= have[t]][:CAP_TILES]
        out += [dict(tile=t, year=Y, ha=round(best[t][1], 1), years=sorted(have[t]), screen=Y == SCREEN_YEAR) for t in keep]
        print(f"onset {Y}: {len(cand)} tiles >= {MIN_HA} ha, kept {len(keep)} with Tessera for Y-1..Y+1", flush=True)
    json.dump(out, open(SEL_PATH, "w"), indent=1)
    print(f"{len(out)} units; {sum(r['screen'] for r in out)} screening ({SCREEN_YEAR}), {sum(not r['screen'] for r in out)} held out")


def _masks(rec):
    """Return the disturbed and undisturbed masks on the tile grid, at stride S."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    z = np.load(LAB / "tiles" / f"{rec['tile']}.npz")
    onset = crop_to_grid(z["onset"].astype(np.int16), z["transform"], rec["tile"], -1)[::S, ::S]
    never = crop_to_grid(z["never"].astype(np.int16), z["transform"], rec["tile"], -1)[::S, ::S]
    new = binary_erosion(onset == rec["year"])
    ever = never == 0                                          # water in some year 2015-2021
    ref = (never == 1) & (distance_transform_edt(~ever) * 10.0 * S > 60)
    return new, ref


def fetch_one(rec):
    out = _npz(rec)
    if out.exists():
        return rec["tile"], "cached"
    try:
        new, ref = _masks(rec)
        if new.sum() < 300 or ref.sum() < 2000:
            return rec["tile"], "below the pixel floors"
        rng = np.random.default_rng(zlib.crc32(rec["tile"].encode()))
        store = sample_tile_years(rec["tile"], rec["years"], {"dmg": new, "ref": ref}, S, CAP, N_TRAIN, rng)
        if not store:
            return rec["tile"], "no years"
        PIX.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **store)
        return rec["tile"], ""
    except Exception as e:
        return rec["tile"], repr(e)


def phase_fetch():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch_one, SEL))
    bad = [(t, e) for t, e in res if e and e != "cached"]
    print(f"{len(res) - len(bad)}/{len(res)} units sampled; not sampled: {len(bad)} {bad[:8]}", flush=True)


def phase_train():
    if CKPT.exists():
        print("cached"); return
    X = corpus([_npz(r) for r in SEL])
    print(f"new-water corpus {X.shape}", flush=True)
    train_and_save(X, CKPT)


def phase_screen():
    import torch
    sae = load_sae(path=CKPT)
    torch.set_num_threads(16)
    with open(JSONL, "w") as out:
        for r in SEL:
            if not _npz(r).exists():
                out.write(json.dumps(dict(ok=False, tile=r["tile"], error="not sampled")) + "\n")
                continue
            z = np.load(_npz(r))
            dmg, ref, n = {}, {}, {}
            for y in YEARS:
                if f"{y}_dmg" not in z.files:
                    continue
                (a, na), (b, nb) = mean_activation(sae, z[f"{y}_dmg"], 300), mean_activation(sae, z[f"{y}_ref"], 2000)
                if a is None or b is None:
                    continue
                dmg[str(y)], ref[str(y)], n[str(y)] = a.tolist(), b.tolist(), [na, nb]
            ok = {str(r["year"]), str(r["year"] + 1)} <= set(dmg)
            out.write(json.dumps(dict(ok=ok, tile=r["tile"], year=r["year"], ha=r["ha"], screen=r["screen"], n=n,
                                      dmg=dmg, ref=ref, **({} if ok else {"error": "gap at the window"}))) + "\n")
    verdicts()


def verdicts():
    rows = [r for r in map(json.loads, open(JSONL)) if r.get("ok")]
    scr, held = [r for r in rows if r["screen"]], [r for r in rows if not r["screen"]]
    fam = anchored_family(aggregate_contrasts(row_units(scr), min_units=MIN_UNITS), window=WINDOW)
    v = peak_outcome(row_contrasts(held, key="tile"), fam, WINDOW) if fam else {}
    summ = outcome_summary(v, {r["tile"]: r["year"] for r in held})
    print(f"{len(scr)} screening tiles ({SCREEN_YEAR}), {len(held)} held out; family {len(fam)} features {fam}")
    for k, s in summ.items():
        label = "all" if k == "all" else f"onset {k}"
        print(f"  {label}: dated to {{Y, Y+1}} {s['rate']:.3f} (n={s['n']}; Y share among dated {s['y0_share']:.2f})")
    json.dump(dict(family=fam, n_screen=len(scr), n_held=len(v), rate=summ["all"]["rate"], summary=summ,
                   outcomes={u: dict(peak=pk, ok=ok) for u, (pk, ok) in v.items()}),
              open(RESULTS_DIR / "new_water.json", "w"), indent=1, default=float)
    print("saved results/new_water.json")


if __name__ == "__main__":
    {"select": phase_select, "fetch": phase_fetch, "train": phase_train, "screen": phase_screen,
     "verdicts": verdicts}[sys.argv[1]]()
