"""Find feature families and score units by the year their family peaks.

Years are counted from the event (0 is the event year). A feature's gap is its mean activation on disturbed minus
undisturbed pixels. A feature joins the family when all three hold:
  in window  its largest gap in the expected window is above the bar
  before     its absolute gap at every offset before the event is below half the bar
  later      its gap at every offset from LATER_FROM (3) on is below half its in-window gap
The bar is ANCHOR (29) times the median absolute in-window gap over the dictionary's active features (those with a
nonzero gap at some offset).

Units are combined by the median over units of (disturbed - undisturbed), not the difference of the two medians.
`min_units` drops year offsets that fewer units cover. The SE US fire and new water screens set it to 3."""
import numpy as np

ANCHOR = 29.0          # bar = ANCHOR times the median absolute in-window gap
WINDOW = (0, 1)        # default expected window: the event year and the year after
LATER_FROM = 3         # fading is checked from this year offset on

def median_contrasts(units, min_units=0):
    """Combine units, each {offset: (M,) contrast}, into {offset: (M,) median over units}. Offsets that fewer than
    min_units units cover are dropped."""
    per_offset = {}
    for u in units:
        for off, c in u.items():
            per_offset.setdefault(off, []).append(np.asarray(c, dtype=float))
    return {off: np.median(cs, axis=0) for off, cs in per_offset.items() if len(cs) >= max(min_units, 1)}

def aggregate_contrasts(units, min_units=0, m=None):
    """Combine units, each {offset: (disturbed, undisturbed)}, into {offset: (M,) median of disturbed - undisturbed}."""
    out = median_contrasts(({off: np.asarray(d, dtype=float) - np.asarray(r, dtype=float) for off, (d, r) in u.items()}
                            for u in units), min_units)
    if m is not None:
        for off, v in out.items():
            if v.shape[0] != m:
                raise ValueError(f"offset {off}: width {v.shape[0]} != m={m}")
    return out

def anchored_family(traj, anchor=ANCHOR, window=WINDOW, later_from=LATER_FROM):
    """Return the features that pass the screen on an {offset: (M,)} trajectory. `window` is the expected window.

    A missing offset in the window counts as zero. Missing offsets before the event or later add no condition.
    """
    if not traj:
        return []
    m = len(next(iter(traj.values())))
    stack = np.array(list(traj.values()))

    z = np.zeros(m)
    inwin = np.maximum.reduce([traj.get(o, z) for o in window])
    pre_offsets = [o for o in traj if o < 0]
    pre = np.max([np.abs(traj[o]) for o in pre_offsets], axis=0) if pre_offsets else z
    late_offsets = [o for o in traj if o >= later_from]
    later = np.max([traj[o] for o in late_offsets], axis=0) if late_offsets else z

    act = np.abs(stack).max(0) > 0
    if not act.any():
        return []
    bar = anchor * np.median(np.abs(inwin[act]))
    return [int(f) for f in
            np.where((inwin > bar) & (pre < bar / 2) & (later < 0.5 * inwin))[0]]

def peak_outcome(traj_by_unit, family, window=WINDOW):
    """The peak-year outcome, {unit: (peak_offset, ok)}: does the family's mean contrast peak in `window`?

    The same function scores the committed family and the null's random families.
    """
    out = {}
    for uid, traj in traj_by_unit.items():
        if not traj:
            continue
        offs = sorted(traj)
        score = [float(np.mean(traj[o][family])) for o in offs]
        peak = offs[int(np.argmax(score))]
        out[uid] = (peak, peak in window)
    return out

def row_units(rows, disturbed="dmg", undisturbed="ref", year="year"):
    """Turn screen rows into one {offset: (disturbed, undisturbed)} per unit, the form aggregate_contrasts takes.
    The rows store both sides by calendar year. The defaults are the stored keys (dmg is disturbed, ref undisturbed)."""
    return [{int(y) - r[year]: (r[disturbed][y], r[undisturbed][y]) for y in r[disturbed] if y in r[undisturbed]}
            for r in rows]

def row_contrasts(rows, key="uid", disturbed="dmg", undisturbed="ref", year="year"):
    """Turn the same rows into {unit id: {offset: disturbed - undisturbed}}, the form peak_outcome takes."""
    return {r[key]: {o: np.asarray(c, float) - np.asarray(f, float) for o, (c, f) in u.items()}
            for r, u in zip(rows, row_units(rows, disturbed, undisturbed, year))}

def outcome_summary(v, group=None):
    """Share of units dated in the expected window, and of those, the share that peak in the event year.

    Given overall and per group. `v` is peak_outcome's {unit: (peak, ok)}. `group` maps each unit to its group,
    if given.
    """
    def one(items):
        ok = [o for _, o in items]
        rate = float(np.mean(ok)) if ok else float("nan")
        y0 = float(np.mean([pk == 0 for pk, o in items if o])) if any(ok) else float("nan")
        return dict(n=len(items), rate=rate, y0_share=y0)
    out = dict(all=one(list(v.values())))
    if group:
        for g in sorted({group[u] for u in v}):
            out[g] = one([pv for u, pv in v.items() if group[u] == g])
    return out

def anchored_bar_family(m, act=None, anchor=ANCHOR):
    """Features whose contrast m is above the anchored bar, with no timing conditions.

    Used for classification and detection. The bar is anchor times the median absolute m over active features.
    By default `act` is the features with nonzero m.
    """
    m = np.asarray(m, float)
    act = np.abs(m) > 0 if act is None else act
    bar = anchor * np.median(np.abs(m[act]))
    return [int(f) for f in np.where(m > bar)[0]]
