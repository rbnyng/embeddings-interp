"""Rebuild each dated setting's family at 1/4 to 4 times the anchored bar and score its held-out units (Table S2).

The screening units' contrasts are combined by their median and screened at 0.25, 0.5, 1, 2 and 4 times ANCHOR with
the setting's expected window. Each family is scored on the held-out units the null uses. The family at 1 times
ANCHOR must equal the committed family, or the script stops and writes nothing. Screening units are:
  western US fire    the Dixie Fire, burned minus unburned
  windthrow          the 12 Friederike screening tiles, in 2018
  landslides         the 15 Emilia-Romagna screening tiles
  boreal fire        the 2018 fires
  SE Australia fire  the wildfires that ignited in September or October 2019
  the rest           the units from the setting's screening years

Usage: python pipeline/anchor_sweep.py
Writes results/anchor_sweep.json."""
import json

import numpy as np

from lib import measure
from lib import settings as S
from lib.config import RESULTS_DIR
from lib.measure import Unit, family, held_out
from lib.screen import ANCHOR, anchored_family, median_contrasts, peak_outcome

MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def score(units, family, window):
    if not family:
        return float("nan")
    v = peak_outcome({u.uid: u.offsets() for u in units}, list(family), window)
    return float(np.mean([ok for _, ok in v.values()]))


# ---- screening units that results/units/ leaves out, read again from the screen outputs ----

def jsonl_screen(fn, year=lambda r: r["year"], uid=lambda r: r["uid"]):
    """Screening rows (flagged "screen") of a screen output storing dmg (disturbed) and ref (undisturbed) by year."""
    out = []
    for r in map(json.loads, open(RESULTS_DIR / fn)):
        if not r.get("ok") or not r.get("screen"):
            continue
        y0 = year(r)
        con = {int(y) - y0: np.asarray(r["dmg"][y], float) - np.asarray(r["ref"][y], float)
               for y in r["dmg"] if y in r["ref"]}
        out.append(Unit(uid(r), r["tile"], y0, con))
    return out


def se_australia_screen():
    out = []
    for r in map(json.loads, open(RESULTS_DIR / "se_australia_fire.jsonl")):
        if not r.get("ok") or r["ig_date"][:7] not in ("2019-09", "2019-10"):
            continue
        con = {int(y) - 2019: np.asarray(v, float) for y, v in r["traj"].items()}
        out.append(Unit(r["uid"], r.get("tile", r["uid"]), 2019, con))
    return out


def boreal_screen():
    out = []
    for r in map(json.loads, open(RESULTS_DIR / "boreal.jsonl")):
        if not r.get("ok") or r.get("which") != "local" or r["year"] != 2018:
            continue
        con = {int(y) - 2018: np.asarray(v, float) for y, v in r["traj"].items()}
        out.append(Unit(r["uid"], r.get("tile", r["uid"]), 2018, con))
    return out


def dixie_screen():
    """The western US screening set: the Dixie fire as one unit, burned minus unburned (results/feature_stats.json)."""
    d = json.load(open(RESULTS_DIR / "feature_stats.json"))
    con = {int(y) - 2021: np.asarray(d["mean_burn"][y], float) - np.asarray(d["mean_unburn"][y], float)
           for y in d["mean_burn"]}
    return [Unit("dixie2021", "dixie", 2021, con)]


def screen_units(name, setting):
    if name == "Western US fire":
        return dixie_screen()
    if name == "Windthrow":                 # 12 Friederike tiles, in the storm's year
        return jsonl_screen("windthrow.jsonl", year=lambda r: 2018, uid=lambda r: r["tile"])
    if name == "Boreal fire":
        return boreal_screen()
    if name == "SE Australia fire":
        return se_australia_screen()
    if name == "Landslides":                # 15 Emilia-Romagna tiles
        return jsonl_screen("landslides.jsonl")
    return [u for u in measure.load(setting.unit_key)
            if u.event_year in setting.screening_years]


def sweep_setting(name, dictionary, setting):
    eval_units = held_out(measure.load(setting.unit_key), setting)
    traj = median_contrasts([u.offsets() for u in screen_units(name, setting)], dictionary.min_units)
    fam1 = anchored_family(traj, anchor=ANCHOR, window=setting.window)
    com = family(name)
    if sorted(fam1) != sorted(com):
        raise AssertionError(
            f"{name}: family re-derived at x1 has {len(fam1)} features, the "
            f"committed family {len(com)}; the sweep no longer reproduces the "
            f"paper's screen, so its other multipliers mean nothing")
    return {m: (fam := anchored_family(traj, anchor=m * ANCHOR, window=setting.window),
                len(fam), score(eval_units, fam, setting.window))[1:]
            for m in MULTS}, len(eval_units)


def main():
    out = {}
    for name, (dictionary, setting) in S.SETTINGS.items():
        if not setting.null:
            continue
        row, n = sweep_setting(name, dictionary, setting)
        out[name] = {"n": n, **{f"{m:g}x": dict(family=f, rate=round(r, 4))
                                for m, (f, r) in row.items()}}
        cells = "  ".join(f"x{m:g} {f:>3}/{r:.3f}" for m, (f, r) in row.items())
        print(f"  {name:<24} n={n:<4} {cells}")
    json.dump(out, open(RESULTS_DIR / "anchor_sweep.json", "w"), indent=1)
    print(f"\n{len(out)} settings swept; x1 reproduced every committed family")


if __name__ == "__main__":
    main()
