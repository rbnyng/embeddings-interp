"""Measure how strong each fire class's signal is in raw Tessera and AlphaEarth embeddings, and how much of it is left
two years after the peak. No dictionary is used.

Classes are SE US wildfire, SE US prescribed fire and western US fire, on those settings' fires and masks in both
models. For each fire and year 2017-2025, d(y) is the distance between the mean embeddings of burned and unburned
pixels at stride 4. A fire needs at least 30 burned and 200 unburned pixels. d_pre is the largest d before the fire
year Y, and s(y) is d(y) minus the median d before Y.
  SNR  = max(d[Y], d[Y+1]) / d_pre                (1.0 is the noise level before the event)
  P    = s(peak + 2) / s(peak), peak is Y or Y+1  (clipped to 0-1.2, and only where s(peak) > 0.1 d_pre)

Usage: python experiments/raw_retention.py
Writes results/raw_retention.json.
"""
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from lib.config import GT_DIR, RESULTS_DIR, workers
from lib.tiles import load_tile
from lib.alphaearth import load_aef

YEARS = list(range(2017, 2026))
SE_CLASS = {"wildfire": "SE US wildfire", "prescribed": "SE US prescribed fire"}
STRIDE = 4


def traj_metrics(d, Y):
    pre = [d[y] for y in d if y < Y]
    if not pre or Y not in d or Y + 1 not in d:
        return None
    d_pre = max(pre)
    snr = max(d[Y], d[Y + 1]) / max(d_pre, 1e-9)
    base = float(np.median(pre))
    s = {y: d[y] - base for y in d}
    peak = Y if s[Y] >= s[Y + 1] else Y + 1
    P = None
    if peak + 2 in s and s[peak] > 0.1 * max(d_pre, 1e-9):
        P = float(np.clip(s[peak + 2] / s[peak], 0, 1.2))
    return dict(snr=float(snr), P=P)


def raw_d(loader, tile, dist, ref):
    d = {}
    for y in YEARS:
        try:
            x = loader(tile, y, stride=STRIDE)
        except Exception:
            continue
        x = np.nan_to_num(np.asarray(x, dtype=np.float32))
        H, W = x.shape[:2]
        dd, rr = dist[:H, :W], ref[:H, :W]
        if dd.sum() < 30 or rr.sum() < 200:
            return {}
        d[y] = float(np.linalg.norm(x[dd].mean(0) - x[rr].mean(0)))
    return d


def half(m):
    return m[::2, ::2]


def work(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    kind, payload = job
    out = []
    try:
        if kind == "se":
            import geopandas as gpd
            import se_us_fire
            rec = payload
            others = gpd.read_file(GT_DIR / "mtbs" / "mtbs_perims_DD.shp",
                                   bbox=(rec["lon"] - 0.08, rec["lat"] - 0.08,
                                         rec["lon"] + 0.08, rec["lat"] + 0.08))
            b, u = se_us_fire.event_masks(rec, others)
            b, u = half(b), half(u)
            for model, loader in (("Tessera", load_tile), ("AlphaEarth", load_aef)):
                m = traj_metrics(raw_d(loader, rec["tile"], b, u), rec["year"])
                if m:
                    out.append((SE_CLASS[rec["fire_type"]], model, m))
        elif kind == "conus":
            import western_us_fire
            rec = payload
            b, u = western_us_fire.fire_mask(rec)
            b, u = half(b), half(u)
            for model, loader in (("Tessera", load_tile), ("AlphaEarth", load_aef)):
                m = traj_metrics(raw_d(loader, rec["tile"], b, u), rec["year"])
                if m:
                    out.append(("Western US fire", model, m))
    except Exception as e:
        print(f"  {kind} job failed: {e!r}", flush=True)
    return out


def main():
    jobs = []
    for rec in json.load(open(RESULTS_DIR / "se_us_fire_selection.json")):
        jobs.append(("se", rec))
    import western_us_fire
    for rec in western_us_fire.FIRES:
        jobs.append(("conus", rec))
    print(f"{len(jobs)} jobs", flush=True)
    units = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers(48)) as ex:
        for i, res in enumerate(ex.map(work, jobs), 1):
            for cls, model, m in res:
                units[(cls, model)].append(m)
            if i % 100 == 0:
                print(f"  [{i}/{len(jobs)}]", flush=True)

    print(f"\n{'class':<18} {'model':<8} {'n':>5}  {'SNR med [IQR]':<24} "
          f"{'P med (n_P)':<14}")
    out = {}
    for (cls, model), u in sorted(units.items()):
        snrs = np.array([x["snr"] for x in u])
        Ps = np.array([x["P"] for x in u if x["P"] is not None])
        key = f"{cls} | {model}"
        out[key] = dict(n=len(u), snr_med=float(np.median(snrs)),
                        snr_q1=float(np.percentile(snrs, 25)),
                        snr_q3=float(np.percentile(snrs, 75)),
                        P_med=float(np.median(Ps)) if len(Ps) else None,
                        n_P=int(len(Ps)))
        o = out[key]
        print(f"{cls:<18} {model:<8} {o['n']:>5}  "
              f"{o['snr_med']:>6.2f} [{o['snr_q1']:.2f}, {o['snr_q3']:.2f}]     "
              f"{o['P_med'] if o['P_med'] is not None else float('nan'):>5.2f} "
              f"({o['n_P']})")

    json.dump(dict(classes=out), open(RESULTS_DIR / "raw_retention.json", "w"), default=float)
    print("saved results/raw_retention.json")


if __name__ == "__main__":
    main()
