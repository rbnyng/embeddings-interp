"""Collect the validation units of every dated setting from the screen outputs, in the format of lib/measure.py.

Europe atlas and Amazon arc units are tile-years: one tile's pixels disturbed in one year, against its intact pixels.
Every other unit is one record of the screen output. Contrasts cover the years 2017-2025. The screen outputs store
the disturbed side as dmg or burn and the undisturbed side as ref, or only the contrast, as traj or con. Run again
whenever a screen output changes.

Usage: python pipeline/units.py
Writes results/units/<unit_key>.jsonl, one file per setting."""
import json

from lib import settings as S
from lib.measure import Unit, save
from lib.config import RESULTS_DIR
STORM_YEAR = dict(friederike=2018, xavier=2017, vaia=2018)   # FORWIND event dates

YEARS = range(2017, 2026)

def rows(fn, ok=True):
    p = RESULTS_DIR / fn
    if fn.endswith(".jsonl"):
        return [r for r in map(json.loads, open(p)) if not ok or r.get("ok")]
    d = json.load(open(p))
    return [r for r in d if isinstance(r, dict) and (not ok or r.get("ok"))] \
        if isinstance(d, list) else []

def from_tables(fn, prefix, which=None, skip_years=None):
    """Tile-year units from tables[year][group], with groups named <prefix><event year>. One record holds several."""
    out = []
    for r in rows(fn):
        if which is not None and r.get("which") != which:
            continue
        groups = {g for y in YEARS if str(y) in r["tables"]
                  for g in r["tables"][str(y)] if g.startswith(prefix)}
        for g in sorted(groups):
            try:
                ey = int(g.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if skip_years and ey in skip_years:
                continue
            dis, und, con = {}, {}, {}
            for y in YEARS:
                t = r["tables"].get(str(y), {})
                if g in t and "intact" in t:
                    o = y - ey
                    dis[o], und[o] = t[g], t["intact"]
                    con[o] = [a - b for a, b in zip(t[g], t["intact"])]
            if con:
                out.append(Unit(f"{r['tile']}:{g}", r["tile"], ey, con, dis, und))
    return out

def from_pair(fn, ck, rk="ref", ykey="year", uid="event_id", flt=None):
    """One unit per record, from the disturbed (ck) and undisturbed (rk) sides stored by year."""
    out = []
    for r in rows(fn):
        if flt and not flt(r):
            continue
        ey = ykey(r) if callable(ykey) else (ykey if isinstance(ykey, int) else r[ykey])
        dis, und, con = {}, {}, {}
        for y in YEARS:
            s = str(y)
            if s in r[ck] and s in r[rk]:
                o = y - ey
                dis[o], und[o] = r[ck][s], r[rk][s]
                con[o] = [a - b for a, b in zip(r[ck][s], r[rk][s])]
        if con:
            u = str(r.get(uid, r.get("tile")))
            out.append(Unit(u, str(r.get("tile", u)), ey, con, dis, und))
    return out

def from_contrast(fn, key, ykey="year", uid="uid", flt=None):
    """One unit per record, from a stored contrast (the two sides were not stored)."""
    out = []
    for r in rows(fn):
        if flt and not flt(r):
            continue
        ey = ykey(r) if callable(ykey) else r[ykey]
        con = {y - ey: r[key][str(y)] for y in YEARS if str(y) in r[key]}
        if con:
            u = str(r.get(uid, r.get("tile")))
            out.append(Unit(u, str(r.get("tile", u)), ey, con))
    return out

def se_units(d, s, kind):
    return (from_pair(d.jsonl, "burn", flt=lambda r: r["fire_type"] == kind)
            + from_pair("se_us_fire_extra.jsonl", "burn",
                        flt=lambda r: r["fire_type"] == kind
                        and r["year"] > max(s.screening_years)))

# Maps each setting to builder(dictionary, setting). File names and group prefixes come from lib/settings.py.
JOBS = {
    "Europe harvest":     lambda d, s: from_tables(d.jsonl, s.group_prefix),
    "Europe wind/beetle": lambda d, s: from_tables(d.jsonl, s.group_prefix),
    "Europe fire":        lambda d, s: from_tables(d.jsonl, s.group_prefix),
    "Amazon arc clearing": lambda d, s: from_tables(d.jsonl, s.group_prefix),
    # se_us_fire_extra.jsonl has the events left out by the cap of 100 per fire type. They only join as held-out
    # units. Extras from a screening year are left out.
    "SE US wildfire":     lambda d, s: se_units(d, s, "wildfire"),
    "SE US prescribed fire": lambda d, s: se_units(d, s, "prescribed"),
    # every storm tile the family was not screened on, each in its own storm's event year
    "Windthrow":          lambda d, s: from_pair(d.jsonl, "dmg",
                                                  ykey=lambda r: STORM_YEAR[r["storm"]],
                                                  uid="tile", flt=lambda r: not r.get("screen")),
    # the 2018 units are the screening units, not held out
    "Boreal fire":        lambda d, s: from_contrast(d.jsonl, "traj",
                                                      flt=lambda r: r.get("which") == "local"
                                                      and r["year"] not in s.screening_years),
    # The screening units are fires that started in Sep-Oct 2019. So the split is by ignition month,
    # not by Setting.screening_years.
    "SE Australia fire":  lambda d, s: from_contrast(d.jsonl, "traj",
                                                      ykey=lambda r: int(r["ig_date"][:4]),
                                                      flt=lambda r: r["ig_date"][:7] not in ("2019-09", "2019-10")),
    "Western US fire":    lambda d, s: from_contrast(d.jsonl, "con", uid="event_id"),
    # every landslide tile the family was not screened on (as for Windthrow), each in its own event's year
    "Landslides":         lambda d, s: from_pair(d.jsonl, "dmg", uid="uid", flt=lambda r: not r.get("screen")),
    # The 2018 units are the screening units. held_out drops them using Setting.screening_years.
    "New water":          lambda d, s: from_pair(d.jsonl, "dmg", uid="tile"),
    "Africa forest loss": lambda d, s: from_pair(d.jsonl, "dmg", uid="uid"),
    "East Asia forest loss": lambda d, s: from_pair(d.jsonl, "dmg", uid="uid"),
}


def check_registry():
    """Check that every setting with a unit file has a builder, and that every builder has a unit file."""
    declared = set(S.UNIT_SETTINGS)
    built = {S.SETTINGS[n][1].unit_key for n in JOBS}
    if declared != built:
        raise KeyError(f"settings.py declares unit files {sorted(declared - built)} "
                       f"with no builder; units.py builds {sorted(built - declared)} "
                       f"with no Setting.unit_key")


if __name__ == "__main__":
    check_registry()
    total = 0
    for name, build in JOBS.items():
        dictionary, setting = S.SETTINGS[name]
        key = setting.unit_key
        us = build(dictionary, setting)
        n = save(key, us)
        total += n
        both = sum(u.can_diff_of_median for u in us)
        print(f"  {key:<20} {n:>5} units   ({both} with both sides stored)")
    print(f"\n{total} units written to results/units/")
