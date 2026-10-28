"""Build the corpus table, one row per validation unit: place, event year, date type, expected window and outcome.

Dated settings take their peak-year outcomes from lib.measure.outcomes. This script adds coordinates, ignition month
and disturbance kind. Three settings are built differently:
  western US fire  all 242 mapped fires, with the screen's own outcomes. The 207 fires that have a contrast per
                   feature must agree with lib.measure.outcomes.
  crop atlas       a tile validates when its held-out accuracy reaches crop_atlas.TILE_PASS
  crop rotation    a tile validates when its held-out accuracy is at least 0.80

Usage: python pipeline/events.py
Writes results/paper_events.csv.
"""
import json

import numpy as np
import pandas as pd

from lib import measure
from lib import settings as S
from lib.config import RESULTS_DIR
from lib.tiles import tile_center

def load(name):
    return json.load(open(RESULTS_DIR / name))

WINDOW = {(0, 1): "{Y,Y+1}", (1,): "Y+1"}

# ---- (lon, lat, ignition month) for each unit ----
def at_tile(month=np.nan):
    return lambda u: (*tile_center(u.tile), month)

# the selection without the cap: the screened events plus the ones the cap of 100 per fire type left out
rx_sel = {r["event_id"]: r for r in load("se_us_fire_selection_nocap.json")}
au_inv = {str(r["uid"]): r for r in load("se_australia_inventory.json")}
nbac_inv = {str(r["uid"]): r for r in load("nbac_inventory.json")}
storm_sel = {r["tile"]: r for r in load("windthrow_selection.json")}

def rx_meta(u):
    r = rx_sel[u.uid]
    return r["lon"], r["lat"], int(r["ig_date"][5:7])

STORM_MONTH = dict(friederike=1, xavier=11, vaia=10)

# maps each setting to (kind, metadata)
META = {
    "SE Australia fire":  ("fire", lambda u: (*tile_center(au_inv[u.uid]["tile"]),
                                              int(au_inv[u.uid]["ig_date"][5:7]))),
    "Boreal fire":        ("fire", lambda u: (*tile_center(nbac_inv[u.uid]["tile"]),
                                              np.nan)),
    "Europe harvest":     ("harvest", at_tile()),
    "Europe wind/beetle": ("wind_beetle", at_tile()),
    "Europe fire":        ("fire", at_tile()),
    "Amazon arc clearing": ("clearing", at_tile()),
    "SE US wildfire":     ("fire", rx_meta),
    "SE US prescribed fire": ("prescribed", rx_meta),
    # Friederike 18 Jan 2018, Xavier 10 Nov 2017, Vaia 28 Oct 2018
    "Windthrow":          ("windthrow", lambda u: (storm_sel[u.tile]["lon"], storm_sel[u.tile]["lat"],
                                                   STORM_MONTH[storm_sel[u.tile]["storm"]])),
    "Landslides":         ("landslide", at_tile()),
    "New water":          ("water", at_tile()),
    "Africa forest loss": ("forest loss", at_tile()),
    "East Asia forest loss": ("forest loss", at_tile()),
}

rows = []

# ---- western US: all 242 mapped fires, with western_us_fire's outcomes ----
# peak is the family's peak year. dated means it falls in the expected window.
conus = [r for r in map(json.loads, open(RESULTS_DIR / "western_us_fire.jsonl"))
         if r.get("ok")]
by_id = {r["event_id"]: r for r in conus}
for u, off, ok in measure.outcomes("Western US fire"):
    r = by_id[u.uid]
    assert (r["peak"], bool(r["dated"])) == (u.event_year + off, ok), \
        f"western US fire {u.uid}: the screen's outcome and the unit's outcome disagree"
for r in conus:
    rows.append(dict(region="Western US fire", kind="fire", lon=r["lon"], lat=r["lat"],
                     event_year=r["year"], ig_month=int(r["ig_date"][5:7]),
                     peak_year=r["peak"], ok=bool(r["dated"]),
                     window="Y+1", date_type="event"))

# ---- every other disturbance setting, through the units ----
def setting_rows(name):
    kind, meta = META[name]
    _, setting = S.SETTINGS[name]
    for u, off, ok in measure.outcomes(name):
        lon, lat, month = meta(u)
        yield dict(region=name, kind=kind, lon=lon, lat=lat,
                   event_year=u.event_year, ig_month=month,
                   peak_year=u.event_year + off, ok=bool(ok),
                   window=WINDOW[setting.window], date_type=setting.date_type,
                   event=storm_sel[u.tile]["storm"].capitalize() if name == "Windthrow" else ""), u

def emit(names, order=None):
    """Append the settings' rows, sorted by `order` if given. Row order decides which map point is drawn on top."""
    batch = [ru for n in names for ru in setting_rows(n)]
    if order:
        batch.sort(key=lambda ru: order(*ru))
    rows.extend(r for r, _ in batch)

eu_order = {(v["agent"], v["tile"], v["year"]): i
            for i, v in enumerate(load("europe_atlas.json")["outcomes"])}
rx_order = {json.loads(l)["event_id"]: i for i, l in enumerate(
    [*open(RESULTS_DIR / "se_us_fire.jsonl"), *open(RESULTS_DIR / "se_us_fire_extra.jsonl")])}

emit(["SE Australia fire"])
emit(["Boreal fire"])
emit(["Europe harvest", "Europe wind/beetle", "Europe fire"],
     order=lambda r, u: eu_order[(r["kind"], u.tile, u.event_year)])
emit(["Amazon arc clearing"])
emit(["SE US wildfire", "SE US prescribed fire"], order=lambda r, u: rx_order[u.uid])
emit(["Windthrow"])
emit(["Landslides"])
emit(["New water"])
emit(["Africa forest loss"])
emit(["East Asia forest loss"])

# ---- crop atlas: a tile validates when held-out accuracy is at least crop_atlas.TILE_PASS ----
# a tile with no pixels to score keeps ok as NaN
crop_tiles = load("crop_atlas.json")["tiles"]
for t, v in crop_tiles.items():
    lon, lat = tile_center(t)
    rows.append(dict(region="Crop atlas", kind="management", lon=lon, lat=lat,
                     event_year=2024, ig_month=np.nan, peak_year=np.nan,
                     ok=(np.nan if v["ok"] is None else bool(v["ok"])),
                     window="held-out acc", date_type="none"))

# ---- crop rotation (crop_rotation): a tile validates when held-out accuracy is at least 0.80 ----
for r in load("crop_rotation.json")["rows"]:
    if not r.get("ok"):
        continue
    lon, lat = tile_center(r["tile"])
    rows.append(dict(region="Crop rotation", kind="management", lon=lon, lat=lat,
                     event_year=2024, ig_month=np.nan, peak_year=np.nan,
                     ok=bool(r.get("held_acc", 0) >= 0.80), window="held-out acc",
                     date_type="none"))

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "paper_events.csv", index=False)
print(df.groupby("region")["ok"].agg(["count", "mean"]).round(3))
print("\nby date type:")
print(df.groupby("date_type")["ok"].agg(["count", "mean"]).round(3))
print(f"\nTOTAL units: {len(df)}   dated/validated: {int(df.ok.sum())} "
      f"({df.ok.mean():.3f})")
