# -*- coding: utf-8 -*-
"""
Imputation audit: what share of the panel did cleaning actually touch?

Counts every modification the cleaner makes, per variable, as a share of the
county-year CELLS that exist for that variable, and flags anything over a
tolerance (default 10%).  The denominator matters: a "county-year" is one row,
but cleaning acts on FIVE variables independently, so the honest unit is the
county-year x variable CELL.  Both are reported.

Categories (mutually exclusive per cell):
  set_NA_invalid   0 / <1 / low-jump rules            (4b)
  boundary_NA      boundary run dropped               (4c, mode=na or non-decimal)
  boundary_rescale boundary run spliced onto the core (4c)
  placeholder_NA   cross-county placeholder value     (4d)
  imputed_auto     spike-and-revert, provincial trend (6)
  imputed_manual   hand-listed multi-year stretch     (7)

Run:  python src/05_imputation_audit.py [--tol 0.10]
Reads only outputs, changes nothing.
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.10)
    a = ap.parse_args()

    m = pd.read_csv(C.CLEAN_PANEL)
    n_rows = len(m)
    n_cty = m.countyid.nunique()
    print(f"panel: {n_rows:,d} county-years, {n_cty:,d} counties, "
          f"{m.year.min()}-{m.year.max()}, {len(C.IO_VARS)} variables")
    print(f"       {n_rows*len(C.IO_VARS):,d} county-year x variable cells\n")

    alog = pd.read_csv(C.ANOMALY_LOG) if os.path.exists(C.ANOMALY_LOG) else pd.DataFrame()
    rsc = os.path.join(C.DQ_DIR, "boundary_run_rescale.csv")
    rescale = pd.read_csv(rsc) if os.path.exists(rsc) else pd.DataFrame()

    rows = []
    for v in C.IO_VARS:
        raw = pd.to_numeric(m[v + "_raw"], errors="coerce")
        cur = pd.to_numeric(m[v], errors="coerce")
        imp = m[v + "_imp"] == True
        present_raw = raw.notna() & (raw > 0)          # the cells cleaning could act on
        n_present = int(present_raw.sum())

        n_imp = int(imp.sum())
        n_manual = 0
        if len(alog):
            man = alog[(alog["var"] == v) & (alog.get("method", "").astype(str)
                                             .str.contains("manual", case=False, na=False))]
            n_manual = len(man)
        n_auto = n_imp - n_manual if n_imp >= n_manual else n_imp

        n_resc = int((rescale["var"] == v).sum()) if len(rescale) else 0
        # became NA although a positive raw value existed = dropped by some rule
        n_na = int((present_raw & cur.isna()).sum())
        touched = n_imp + n_resc + n_na

        rows.append(dict(variable=v, cells_present=n_present,
                         imputed_auto=n_auto, imputed_manual=n_manual,
                         rescaled=n_resc, set_to_NA=n_na,
                         touched=touched,
                         share_touched=touched / max(n_present, 1),
                         share_imputed=n_imp / max(n_present, 1),
                         share_NA=n_na / max(n_present, 1)))
    t = pd.DataFrame(rows)

    tot_present = t.cells_present.sum()
    tot_touch = t.touched.sum()
    print(f"{'variable':18s} {'present':>9s} {'imp_auto':>9s} {'imp_man':>8s} "
          f"{'rescaled':>9s} {'->NA':>7s} {'touched':>8s} {'% touched':>10s}")
    for r in t.itertuples():
        flag = "  <-- OVER" if r.share_touched > a.tol else ""
        print(f"{r.variable:18s} {r.cells_present:9,d} {r.imputed_auto:9,d} "
              f"{r.imputed_manual:8,d} {r.rescaled:9,d} {r.set_to_NA:7,d} "
              f"{r.touched:8,d} {100*r.share_touched:9.2f}%{flag}")
    print(f"{'ALL':18s} {tot_present:9,d} {t.imputed_auto.sum():9,d} "
          f"{t.imputed_manual.sum():8,d} {t.rescaled.sum():9,d} {t.set_to_NA.sum():7,d} "
          f"{tot_touch:8,d} {100*tot_touch/tot_present:9.2f}%")

    print(f"\ntolerance {100*a.tol:.0f}% of present cells")
    over = t[t.share_touched > a.tol]
    if len(over):
        print("OVER TOLERANCE: " + ", ".join(
            f"{r.variable} ({100*r.share_touched:.1f}%)" for r in over.itertuples()))
        print("  -> loosen the detector (raise JUMP5 / TAU_END) or narrow the rules")
    else:
        print("all variables within tolerance")

    # per-county concentration: is the touching spread out or concentrated?
    print("\nconcentration — share of counties by how much of their panel was touched:")
    per = []
    for v in C.IO_VARS:
        raw = pd.to_numeric(m[v + "_raw"], errors="coerce")
        cur = pd.to_numeric(m[v], errors="coerce")
        tch = (m[v + "_imp"] == True) | (raw.notna() & (raw > 0) & cur.isna())
        per.append(pd.DataFrame({"countyid": m.countyid, "v": v, "t": tch,
                                 "p": raw.notna() & (raw > 0)}))
    per = pd.concat(per)
    g = per.groupby("countyid").agg(t=("t", "sum"), p=("p", "sum"))
    g = g[g.p > 0]; g["share"] = g.t / g.p
    for lo, hi in [(0, .001), (.001, .05), (.05, .10), (.10, .25), (.25, 1.01)]:
        n = int(((g.share >= lo) & (g.share < hi)).sum())
        print(f"   {100*lo:5.1f}-{100*hi:5.1f}% touched : {n:5,d} counties "
              f"({100*n/len(g):5.1f}%)")

    # ------------------------------------------------------------------
    # Category breakdown, and the same footprint at the ROW level.
    #
    # Two denominators, because they answer different questions.  A cell is one
    # county-year x variable and is the unit cleaning actually acts on -- it is the
    # honest denominator for "how much of the data did we touch".  A ROW is one
    # county-year and counts as touched if ANY of its five variables was; that is
    # the denominator that matters for the frontier, which needs all five present
    # and drops the whole observation if one is missing.  The row share is always
    # the larger of the two and is the number to quote when asked how much of the
    # panel the cleaning moved.
    # ------------------------------------------------------------------
    floor_p = os.path.join(C.DQ_DIR, "extreme_low_floor.csv")
    floor = pd.read_csv(floor_p) if os.path.exists(floor_p) else pd.DataFrame()
    man_p = os.path.join(C.CLEAN_DIR, "manual_impute_log.csv")
    man = pd.read_csv(man_p) if os.path.exists(man_p) else pd.DataFrame()

    n_cells = n_rows * len(C.IO_VARS)
    n_present_cells = int(t.cells_present.sum())
    n_auto = int(sum(int((m[v + "_imp"] == True).sum()) for v in C.IO_VARS))
    n_manual = len(man)
    n_resc = len(rescale)
    n_floor = len(floor)
    n_na_all = int(t.set_to_NA.sum())

    print("\n" + "=" * 74)
    print("FOOTPRINT BY CATEGORY")
    print("=" * 74)
    print(f"{'category':30s} {'cells':>9s} {'% of present':>13s} {'% of all cells':>15s}")
    cats = [("rescaled to a power of ten", n_resc),
            ("imputed (auto, spike+revert)", n_auto),
            ("imputed (manual list)", n_manual),
            ("deleted -> NA (all rules)", n_na_all),
            ("   of which extreme-low floor", n_floor)]
    for lab, k in cats:
        print(f"{lab:30s} {k:9,d} {100*k/n_present_cells:12.2f}% {100*k/n_cells:14.2f}%")
    tot = n_resc + n_auto + n_manual + n_na_all
    print("-" * 74)
    print(f"{'TOTAL modified':30s} {tot:9,d} {100*tot/n_present_cells:12.2f}% "
          f"{100*tot/n_cells:14.2f}%")
    print(f"\ndenominators: {n_present_cells:,d} present cells, {n_cells:,d} total cells "
          f"({n_rows:,d} county-years x {len(C.IO_VARS)} variables)")

    # row level: a county-year is touched if ANY variable in it was
    touched_row = pd.Series(False, index=m.index)
    imputed_row = pd.Series(False, index=m.index)
    na_row = pd.Series(False, index=m.index)
    for v in C.IO_VARS:
        raw = pd.to_numeric(m[v + "_raw"], errors="coerce")
        cur = pd.to_numeric(m[v], errors="coerce")
        imp = m[v + "_imp"] == True
        gone = raw.notna() & (raw > 0) & cur.isna()
        imputed_row |= imp
        na_row |= gone
        touched_row |= (imp | gone)
    if len(rescale):
        key = set(zip(rescale.countyid, rescale.year))
        resc_row = pd.Series([ (c, y) in key for c, y in zip(m.countyid, m.year) ],
                             index=m.index)
        touched_row |= resc_row
    else:
        resc_row = pd.Series(False, index=m.index)

    print("\n" + "=" * 74)
    print("FOOTPRINT AT THE COUNTY-YEAR (ROW) LEVEL")
    print("=" * 74)
    for lab, s in [("rows with any rescale", resc_row),
                   ("rows with any imputation", imputed_row),
                   ("rows with any deletion", na_row),
                   ("rows touched in ANY way", touched_row)]:
        k = int(s.sum())
        print(f"{lab:30s} {k:9,d} of {n_rows:,d} county-years  = {100*k/n_rows:6.2f}%")

    t.to_csv(os.path.join(C.DQ_DIR, "imputation_audit.csv"), index=False)
    print(f"\nsaved -> dq/imputation_audit.csv")


if __name__ == "__main__":
    main()
