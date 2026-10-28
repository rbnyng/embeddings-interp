"""Validation units, the committed family of each setting, and each unit's outcome.

A unit is one thing that gets an outcome: an event or a tile-year. Units are stored in results/units/<unit_key>.jsonl:

    uid, tile, event_year, contrast{offset: [M]}, disturbed?, undisturbed?

contrast maps each year counted from the event (0 is the event year) to an (M,) vector of disturbed minus
undisturbed, per feature. disturbed and undisturbed hold the two sides when the screen stored them. The outcomes do not use them."""
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from lib import settings as S
from lib.config import BURN_FAMILY, RESULTS_DIR
from lib.screen import peak_outcome

UNIT_DIR = RESULTS_DIR / "units"

# Maps each setting to the results file and key path of its committed family. None marks the western US setting,
# which is scored with lib.config.BURN_FAMILY.
FAMILIES = {
    "Europe harvest":         ("europe_atlas.json", ("agents", "harvest", "family")),
    "Europe wind/beetle":     ("europe_atlas.json", ("agents", "wind_beetle", "family")),
    "Europe fire":            ("europe_atlas.json", ("agents", "fire", "family")),
    "Amazon arc clearing":    ("amazon_arc.json", ("family",)),
    "SE US wildfire":         ("se_us_fire.json", ("family",)),
    "SE US prescribed fire":  ("se_us_fire.json", ("family_rx",)),
    "Windthrow":              ("windthrow.json", ("family",)),
    "Boreal fire":            ("boreal.json", ("family",)),
    "SE Australia fire":      ("se_australia_fire.json", ("family",)),
    "Western US fire":        (None, None),
    "Landslides":             ("landslides.json", ("family",)),
    "New water":              ("new_water.json", ("family",)),
    "Africa forest loss":     ("forest_loss_africa.json", ("family",)),
    "East Asia forest loss":  ("forest_loss_eastasia.json", ("family",)),
}

@dataclass
class Unit:
    uid: str
    tile: str
    event_year: int
    contrast: dict
    disturbed: Optional[dict] = None
    undisturbed: Optional[dict] = None

    @property
    def can_diff_of_median(self):
        return self.disturbed is not None and self.undisturbed is not None

    def offsets(self):
        """Return {offset: (M,) contrast array}."""
        return {int(o): np.asarray(v, float) for o, v in self.contrast.items()}

def load(setting_key):
    """Load every unit of one setting."""
    p = UNIT_DIR / f"{setting_key}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run pipeline/units.py")
    out = []
    for line in open(p):
        d = json.loads(line)
        d["contrast"] = {int(k): v for k, v in d["contrast"].items()}
        for k in ("disturbed", "undisturbed"):
            if d.get(k):
                d[k] = {int(kk): vv for kk, vv in d[k].items()}
        out.append(Unit(**d))
    return out

def save(setting_key, units):
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNIT_DIR / f"{setting_key}.jsonl", "w") as f:
        for u in units:
            f.write(json.dumps({k: v for k, v in u.__dict__.items()
                                if v is not None}) + "\n")
    return len(units)


def family(name):
    """The committed family a setting is scored with, as feature indices."""
    fn, path = FAMILIES[name]
    if fn is None:
        return list(BURN_FAMILY)
    d = json.load(open(RESULTS_DIR / fn))
    for k in path:
        d = d[k]
    return list(d)

def aef_dixie_family():
    """The AlphaEarth burn family: the anchored screen on the Dixie Fire's burned-minus-unburned trajectory in the
    AlphaEarth Dixie dictionary, with the year after the fire as the expected window."""
    from lib.screen import anchored_family
    T = json.load(open(RESULTS_DIR / "alphaearth_stats.json"))["tables"]["dixie2021"]
    traj = {int(y) - T["fire_year"]: np.array(T["mean_burn"][str(y)]) - np.array(T["mean_unburn"][str(y)])
            for y in T["years"]}
    return anchored_family(traj, window=(1,))


def held_out(units, setting):
    """Held-out units: those after the setting's last screening year, or all units if it lists none."""
    if setting.screening_years:
        return [u for u in units if u.event_year > max(setting.screening_years)]
    return units

def outcomes(name):
    """(unit, peak offset, ok) for every held-out unit of one setting, from lib.screen.peak_outcome with the setting's
    family and expected window."""
    _, setting = S.SETTINGS[name]
    units = held_out(load(setting.unit_key), setting)
    v = peak_outcome({u.uid: u.offsets() for u in units}, family(name), setting.window)
    return [(u, *v[u.uid]) for u in units if u.uid in v]
