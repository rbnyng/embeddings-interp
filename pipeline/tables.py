"""Write Supplementary Tables S1 (corpus), S2 (anchored-bar sweep) and S3 (null vs live-feature null) as LaTeX.

Stops if a setting's unit count differs from its Setting.n_expected. Table S1 sits on a landscape page, so the paper
must load the pdflscape package. The tables also use booktabs.

Usage: python pipeline/tables.py
Writes paper/si_corpus_table.tex, si_anchor_table.tex and si_null_live_table.tex.
"""
import json

import pandas as pd

from lib import settings as S
from lib.config import RESULTS_DIR

OUT = RESULTS_DIR.parent / "paper" / "si_corpus_table.tex"
OUT_ANCHOR = RESULTS_DIR.parent / "paper" / "si_anchor_table.tex"
OUT_LIVE = RESULTS_DIR.parent / "paper" / "si_null_live_table.tex"
WIN = {(0, 1): r"\{Y, Y$+$1\}", (1,): r"\{Y$+$1\}", (): "---"}

CAPTION = (r"Full results over {n_settings} settings, {n_units} reference units. "
           r"Validation is the share of held-out units whose family peak falls "
           r"in the expected window (disturbance settings) or held-out "
           r"classification accuracy (crop settings), with its 95\% interval for the "
           r"disturbance settings (tile-clustered bootstrap; Wilson score interval where "
           r"every unit or none is dated). Nulls are same-size "
           r"random-family draws (median and 95\% interval over 200 draws)")

def build():
    nulls = json.load(open(RESULTS_DIR / "all_nulls.json"))
    df = pd.read_csv(RESULTS_DIR / "paper_events.csv")
    rows, total = [], 0
    for name, dictionary, setting in S.table_rows():
        if name in nulls:
            v = nulls[name]
            n, rate = v["n"], v["observed"]
            validated = f"{rate:.3f} [{v['obs_lo']:.3f}, {v['obs_hi']:.3f}]"
            null = f"{v['null_median']:.3f} [{v['null_lo']:.3f}, {v['null_hi']:.3f}]"
        else:
            g = df[df.region == name]
            n, rate, null = len(g), float(g.ok.mean()), "---"
            validated = f"{rate:.3f}"
        if setting.n_expected is not None and n != setting.n_expected:
            raise ValueError(f"{name}: {n} units, registry expects "
                             f"{setting.n_expected}. Refusing to print a table that "
                             f"disagrees with the registered design.")
        total += n
        model = "Tessera" if dictionary.model == "tessera" else "AlphaEarth"
        dates = setting.date_type + "-dated" if setting.date_type in ("event", "detection") else setting.date_type
        rows.append(f"{setting.display} & {setting.product} & {model} & {dates} & "
                    f"{WIN[tuple(setting.window)]} & {n:,} & {validated} & {null} \\\\")
    return rows, total

ANCHOR_CAPTION = (
    r"Sensitivity of the held-out validation rate to the screening "
    r"multiplier. For each disturbance setting the family is rederived from "
    r"its screening tile-years with the multiplier in Supplementary "
    r"Equation~\eqref{eq:bar} set to the column's value, and the same "
    r"held-out units are re-scored under the unchanged expected window. Cells "
    r"give the validation rate with the family size in parentheses")

def build_anchor():
    sw = json.load(open(RESULTS_DIR / "anchor_sweep.json"))
    order = [name for name, *_ in S.table_rows() if name in sw]
    rows = []
    for name in order:
        _, _, setting = next(r for r in S.table_rows() if r[0] == name)
        cells = " & ".join(
            f"{sw[name][m]['rate']:.3f} ({sw[name][m]['family']})"
            for m in ("0.25x", "0.5x", "1x", "2x", "4x"))
        rows.append(f"{setting.display} & {sw[name]['n']:,} & {cells} \\\\")
    return rows

def build_live():
    """Compare the null as implemented with draws of only live features outside the family."""
    nulls = json.load(open(RESULTS_DIR / "all_nulls.json"))
    rows, dead = [], {}
    for name, _, setting in S.table_rows():
        if name not in nulls or name.startswith("_"):
            continue
        r = nulls[name]
        rows.append(f"{setting.display} & {r['null_median']:.2f} & {r['live']['null_median']:.2f} "
                    f"& {r['observed']:.2f} \\\\")
        dead[setting.display] = r["live"]["dead_share"]
    keys = [n for n in nulls if not n.startswith("_")]
    lowered = sum(nulls[n]["live"]["null_median"] < nulls[n]["null_median"] for n in keys)
    d = sorted(dead.values())
    caption = (r"\textbf{Random family null is conservative.} Null validation rate (median "
               r"over 200 draws) as implemented, with random feature sets drawn from the "
               r"whole dictionary including family features, and with draws restricted to "
               r"live features outside the family, against the observed rate. A feature is "
               r"live if its contrast is nonzero in at least one held-out unit; dead features "
               rf"are {100 * d[0]:.1f}--{100 * d[-1]:.1f}\% of each dictionary. "
               + f"Restricting draws lowers the null in {lowered} of {len(keys)} settings.")
    return ("\\begin{table}[h]\n\\centering\n\\caption{" + caption + "}\n\\label{tab:null_live}\n"
            "\\small\n\\begin{tabular}{lrrr}\n\\toprule\n"
            "Setting & Null (as implemented) & Null (live, non-family) & Observed \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def main():
    rows, total = build()
    n_units = len(pd.read_csv(RESULTS_DIR / "paper_events.csv"))
    # use replace, not str.format, so the LaTeX braces survive
    caption = (CAPTION.replace("{n_settings}", str(len(rows)))
                      .replace("{n_units}", f"{n_units:,}".replace(",", "{,}")))
    # on its own landscape page, since the eight columns (two with intervals) are too wide for a portrait page.
    # pdflscape turns the page itself, so it reads upright on screen.
    tex = ("\\begin{landscape}\n\\begin{table}\n\\centering\n\\caption{" + caption + "}\n"
           "\\label{tab:corpus}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{@{}lllllrrl@{}}\n\\toprule\n"
           "Setting & Product & Embeddings & Dates & Window & $n$ & Validated [95\\% int.] & "
           "Null (median [95\\% int.]) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\\end{landscape}\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(tex)
    print(f"wrote {OUT} ({len(rows)} rows, {total:,} units)")

    arows = build_anchor()
    from lib.screen import ANCHOR
    heads = " & ".join(rf"\textbf{{{m * ANCHOR:g}}}" if m == 1 else
                       f"{m * ANCHOR:g}" for m in (0.25, 0.5, 1, 2, 4))
    atex = ("\\begin{table}[ht]\n\\centering\n\\caption{" + ANCHOR_CAPTION
            + "}\n\\label{tab:anchor}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}lrlllll@{}}\n\\toprule\n"
            "Setting & $n$ & " + heads + " \\\\\n\\midrule\n"
            + "\n".join(arows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    OUT_ANCHOR.write_text(atex)
    print(f"wrote {OUT_ANCHOR} ({len(arows)} rows)")

    OUT_LIVE.write_text(build_live())
    print(f"wrote {OUT_LIVE}")

if __name__ == "__main__":
    main()
