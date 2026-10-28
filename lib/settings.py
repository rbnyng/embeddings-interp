"""The registry of settings and dictionaries. Each Setting is one row of Table S1. Each Dictionary is one trained
dictionary with its screen tables.

A Setting names its reference product, date type, expected window and held-out units. Settings screened on the same
dictionary share it. For example, the three Europe atlas settings share one."""
from dataclasses import dataclass, field
from typing import Optional

EVENT, DETECTION = "event", "detection"     # date types: event-dated or detection-dated

@dataclass(frozen=True)
class Setting:
    name: str                    # registry key, also used in the results files
    display: str                 # as printed in Table S1
    product: str                 # reference product that gives the dates
    date_type: str               # EVENT or DETECTION
    window: tuple                # expected window, in years from the event
    group_prefix: str = ""       # prefix of the disturbed group's keys in the stored tables
    unit_key: str = ""           # results/units/<unit_key>.jsonl, or "" if there is none
    screening_years: tuple = ()  # event year(s) of the screening units
    null: bool = True            # False where a peak-year null has no meaning
    n_expected: Optional[int] = None   # number of held-out units
    null_seeds: tuple = ()       # random seeds of the null and the live-feature null

@dataclass(frozen=True)
class Dictionary:
    key: str
    model: str                   # "tessera" (128-d) or "aef" (64-d)
    m: int                       # dictionary width
    jsonl: str                   # stored per-feature tables
    min_units: int = 0           # year offsets that fewer units cover are dropped
    settings: tuple = field(default_factory=tuple)

W01 = (0, 1)     # the event year and the year after
W1 = (1,)        # the year after the event

DICTIONARIES = (
    Dictionary(
        key="europe_atlas", model="tessera", m=1024, jsonl="europe_atlas.jsonl",
        settings=(
            Setting("Europe harvest", "Europe harvest", "EFDA", DETECTION, W01,
                "harvest_", "europe_harvest", (2018,), n_expected=826,
                null_seeds=(171799861, 344212842)),
            Setting("Europe wind/beetle", "Europe wind/bark beetle", "EFDA", DETECTION,
                W01, "wind_beetle_", "europe_wind_beetle", (2018,), n_expected=352,
                null_seeds=(3135510692, 1463010323)),
            Setting("Europe fire", "Europe fire", "EFDA", DETECTION, W01,
                "fire_", "europe_fire", (2018,), n_expected=151,
                null_seeds=(124560588, 395875184)),
        )),
    Dictionary(
        key="amazon_arc", model="aef", m=512, jsonl="amazon_arc.jsonl",
        settings=(Setting("Amazon arc clearing", "Amazon arc clearing", "Hansen GFC",
                  DETECTION, W01, "clear_", "amazon_arc", (2018,), n_expected=546,
                  null_seeds=(2223226316, 2291783536)),)),
    Dictionary(
        key="se_us_fire", model="tessera", m=1024, jsonl="se_us_fire.jsonl",
        min_units=3,
        settings=(
            # held-out units include the events left out by the cap of 100 per fire type (se_us_fire_extra.jsonl)
            Setting("SE US wildfire", "SE US wildfire", "MTBS", EVENT, W01,
                unit_key="se_us_wildfire", screening_years=(2018,), n_expected=154,
                null_seeds=(2719751502, 994398574)),
            Setting("SE US prescribed fire", "SE US prescribed fire", "MTBS", EVENT, W01,
                unit_key="se_us_prescribed", screening_years=(2018, 2019),
                n_expected=109,
                null_seeds=(2862943127, 3364096914)),
        )),
    Dictionary(
        key="windthrow", model="tessera", m=1024, jsonl="windthrow.jsonl",
        settings=(
            # One family, screened on 12 Friederike tiles. It is scored on the held-out tiles of
            # Friederike, Xavier and Vaia, each unit in its own storm's year.
            Setting("Windthrow", "Windthrow (Friederike, Xavier, Vaia)", "FORWIND", EVENT,
                W01, unit_key="windthrow", n_expected=89,
                null_seeds=(4274212703, 2535082016)),
        )),
    Dictionary(
        key="boreal", model="aef", m=512, jsonl="boreal.jsonl",
        settings=(Setting("Boreal fire", "Boreal fire", "NBAC", EVENT, W1,
                  unit_key="boreal", screening_years=(2018,), n_expected=309, null_seeds=(1774025236, 530804844)),)),
    Dictionary(
        key="se_australia_fire", model="aef", m=512, jsonl="se_australia_fire.jsonl",
        settings=(Setting("SE Australia fire", "SE Australia fire", "GA Bushfire", EVENT,
                  W01, unit_key="se_australia_fire", n_expected=274, null_seeds=(3972303626, 3816626076)),)),
    Dictionary(
        # 207 of the 242 mapped fires. The rest have too few pixels at the screening stride to give a contrast
        # per feature.
        key="western_us_fire", model="tessera", m=1024, jsonl="western_us_fire_perfeat.jsonl",
        settings=(Setting("Western US fire", "Western US fire", "MTBS", EVENT, W1,
                  unit_key="western_us_fire", n_expected=207, null_seeds=(3902246210, 2751663653)),)),
    Dictionary(
        # One family, screened on 15 random Emilia-Romagna tiles. It is scored on the other Emilia-Romagna tiles and
        # on every Beijing (2023) and Zixing (2024) tile, each unit in its own event's year.
        key="landslides", model="tessera", m=1024, jsonl="landslides.jsonl",
        settings=(Setting("Landslides", "Rainfall landslides (Italy, China)", "RER2023, RLBJ, RLZX", EVENT, W01,
                  unit_key="landslides", n_expected=71, null_seeds=(2676900338, 4225114177)),)),
    Dictionary(
        # land that turns into permanent water (filling reservoirs, rising lakes), starting in 2018-2021
        key="new_water", model="tessera", m=1024, jsonl="new_water.jsonl", min_units=3,
        settings=(Setting("New water", "New permanent water", "JRC GSW", DETECTION, W01,
                  unit_key="new_water", screening_years=(2018,), n_expected=158,
                  null_seeds=(2065722032, 1193428437)),)),
    Dictionary(
        key="africa_forest_loss", model="tessera", m=1024, jsonl="forest_loss_africa.jsonl",
        settings=(Setting("Africa forest loss", "Africa forest loss", "Hansen GFC", DETECTION, W01,
                  unit_key="africa_forest_loss", screening_years=(2018,), n_expected=140,
                  null_seeds=(473136535, 3453041070)),)),
    Dictionary(
        key="eastasia_forest_loss", model="tessera", m=1024, jsonl="forest_loss_eastasia.jsonl",
        settings=(Setting("East Asia forest loss", "East Asia forest loss", "Hansen GFC", DETECTION, W01,
                  unit_key="eastasia_forest_loss", screening_years=(2018,), n_expected=317,
                  null_seeds=(2532495265, 97156655)),)),
    Dictionary(
        # scored by held-out classification, so a peak-year null does not apply
        key="crop_atlas", model="tessera", m=1024, jsonl="crop_atlas.json",
        settings=(
            Setting("Crop atlas", "Crop atlas", "USDA CDL", "---", (),
                null=False, n_expected=102),
            Setting("Crop rotation", "Crop rotation", "USDA CDL", "---", (),
                null=False, n_expected=100),
        )),
)

SETTINGS = {s.name: (d, s) for d in DICTIONARIES for s in d.settings}
BY_KEY = {d.key: d for d in DICTIONARIES}
UNIT_SETTINGS = {s.unit_key: (d, s) for d in DICTIONARIES for s in d.settings if s.unit_key}

TABLE_ORDER = ("Western US fire", "SE US wildfire", "SE US prescribed fire", "Boreal fire",
               "SE Australia fire", "Europe harvest", "Europe wind/beetle",
               "Europe fire", "Windthrow",
               "Amazon arc clearing", "Africa forest loss", "East Asia forest loss",
               "Landslides", "New water", "Crop atlas",
               "Crop rotation")

def table_rows():
    """Return (name, dictionary, setting) for each row, in Table S1 order."""
    missing = [n for n in TABLE_ORDER if n not in SETTINGS]
    if missing:
        raise KeyError(f"TABLE_ORDER names no setting: {missing}")
    absent = [s.name for d in DICTIONARIES for s in d.settings if s.name not in TABLE_ORDER]
    if absent:
        raise KeyError(f"setting missing from TABLE_ORDER: {absent}")
    return [(n, *SETTINGS[n]) for n in TABLE_ORDER]
