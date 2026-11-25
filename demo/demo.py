"""Demo to find the western US burn family on the Dixie Fire and score it on held out fires.

Usage: python demo/demo.py   (from the repository folder)
"""
import gzip
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo" / "data"

# Unpack the demo data and point the code's results folder at it before lib is imported.
tmp = Path(tempfile.mkdtemp(prefix="tessera_interp_demo_"))
for src in DATA.rglob("*.gz"):
    dst = tmp / src.relative_to(DATA).with_suffix("")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f, open(dst, "wb") as g:
        shutil.copyfileobj(f, g)
os.environ["TESSERA_INTERP_RESULTS"] = str(tmp)
sys.path.insert(0, str(ROOT))

from lib import measure, settings as S
from lib.config import BURN_FAMILY
from lib.screen import ANCHOR, anchored_family, median_contrasts
from pipeline.anchor_sweep import dixie_screen
from pipeline.nulls import compute

NAME = "Western US fire"
t0 = time.time()
dictionary, setting = S.SETTINGS[NAME]

# 1. screen: the anchored screen on the Dixie Fire's burned minus unburned trajectory, expected window Y+1
traj = median_contrasts([u.offsets() for u in dixie_screen()], dictionary.min_units)
fam = anchored_family(traj, anchor=ANCHOR, window=setting.window)
print(f"1. Burn family found on the Dixie Fire: {len(fam)} of {dictionary.m} features")
print(f"   same as the paper's family (lib.config.BURN_FAMILY): {sorted(fam) == sorted(BURN_FAMILY)}")

# 2. score: does the family's mean contrast peak in the year after each held-out fire?
out = measure.outcomes(NAME)
dated = sum(ok for *_, ok in out)
print(f"2. Held-out western US fires: {len(out)}, peaking in the year after the fire: {dated} ({dated / len(out):.3f})")

# 3. null: 200 random families of the same size, scored on the same fires
r = compute(NAME, dictionary, setting)
print(f"3. Observed rate {r['observed']:.3f}, 95% interval [{r['obs_lo']:.3f}, {r['obs_hi']:.3f}] ({r['obs_interval']})")
print(f"   Random families: median {r['null_median']:.3f}, 95% interval [{r['null_lo']:.3f}, {r['null_hi']:.3f}], "
      f"share at or above the observed rate {r['p']:.3f}")
print(f"   Random live features outside the family: median {r['live']['null_median']:.3f}")
print(f"Done in {time.time() - t0:.0f} s")
shutil.rmtree(tmp)
