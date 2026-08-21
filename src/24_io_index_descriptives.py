# -*- coding: utf-8 -*-
"""
Descriptive statistics for the input-output variables, the Tornqvist aggregate
input index, and every derived ratio the frontier figures actually plot.

WHY THIS EXISTS ALONGSIDE 22_descriptives.do.  That do-file reports the moments
of the five raw I-O variables (levels and logs, skewness, kurtosis, by reform
stage).  This adds the three things it cannot:

  1. the AGGREGATE INPUT INDEX x, which is built in Python
     (_frontier.aggregate_index) and so does not exist in the Stata panel;
  2. the PLOTTING RANGES -- for every quantity that appears on an axis in
     52_frontier_figs.py / 56_weather_figs.py, the 0.5th and 99.5th percentiles
     that those scripts use as axis limits, so a figure's scale can be checked
     against a number instead of read off the picture;
  3. the SCALE DIAGNOSTIC that decides whether the land and capital LEVELS are
     usable as values at all (see below).

THE SCALE QUESTION, stated precisely.
  The codebook defines three of the four inputs as VALUES at 2005 prices:
      Land_serv_q     hedonic nominal farmland rent / Fixed_PI_2005   [yuan]
      capital_serv_q  PIM user-cost capital service / Fixed_PI_2005   [yuan]
      Inter_all_real  intermediate inputs, already deflated           [yuan]
  If that is true, all three are in the SAME unit, so the ratios Land/M and K/M
  are unit-free and ought to be order 0.1-1: a county spends the same broad
  order of magnitude on land, on capital services and on seed-fertiliser-fuel.

  They are not.  Both come out two to three orders of magnitude below M.  The
  question is then whether that is a UNIT problem (fixable) or a CONSTRUCTION
  problem (not).  The test is whether ONE constant reconciles the whole panel:

      a UNIT error is a single multiplicative constant, so the ratio is wrong by
      the same factor in 1981 and in 2015 and the ratio does NOT drift;

      a CONSTRUCTION error -- e.g. a perpetual-inventory stock started near zero,
      which then grows far faster than the flow it should track -- makes the
      ratio DRIFT, and no constant can fix the early and late years together.

  Table 4 runs exactly that test on both series.  This is what "land is off by
  about 1000x but capital cannot be fixed by any constant" means, and it is the
  reason the Tornqvist index is still valid for both (it uses only ratios within
  a series, where any constant cancels) while the observed COST SHARES are not.

Run:  python src/24_io_index_descriptives.py
Output: src/dq/desc_io_index.csv, desc_plot_ranges.csv, desc_scale_check.csv
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import _common as C
import _tfp as T
import _frontier as F

XCOLS = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
SHORT = {"real_gvp": "Y (real GVP)", "Laborday_impute": "L (labour days)",
         "Land_serv_q": "Land (service)", "capital_serv_q": "K (service)",
         "Inter_all_real": "M (intermediates)", "x": "x (Tornqvist index)"}
Q = [.005, .01, .25, .5, .75, .99, .995]


def desc(s, name):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    q = s.quantile(Q)
    return dict(variable=name, n=len(s), mean=s.mean(), sd=s.std(),
                min=s.min(), p0_5=q[.005], p1=q[.01], p25=q[.25], median=q[.5],
                p75=q[.75], p99=q[.99], p99_5=q[.995], max=s.max(),
                skew=s.skew(), kurt=s.kurt() + 3.0)


def show(t, cols, fmt="{:>13,.4g}"):
    hdr = f"{'variable':22s}" + "".join(f"{c:>13s}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for r in t.itertuples():
        print(f"{r.variable:22s}" + "".join(fmt.format(getattr(r, c)) for c in cols))


def main():
    os.makedirs(C.DQ_DIR, exist_ok=True)
    d = T.load_panel()
    d["x"], w = F.aggregate_index(d, XCOLS, dict(T.BETA_AG))
    print(f"agricultural sample: {len(d):,d} county-years, "
          f"{d.countyid.nunique():,d} counties, {d.year.min()}-{d.year.max()}")
    print("Tornqvist weights (renormalised to 1): "
          + ", ".join(f"{k.split('_')[0]}={v:.4f}" for k, v in w.items()))

    # ------------------------------------------------------------- Table 1/2
    lv = pd.DataFrame([desc(d[c], SHORT[c]) for c in ["real_gvp"] + XCOLS + ["x"]])
    lg = pd.DataFrame([desc(np.log(d[c].where(d[c] > 0)), "ln " + SHORT[c])
                       for c in ["real_gvp"] + XCOLS + ["x"]])
    print("\n" + "=" * 96)
    print("TABLE 1  LEVELS  (Y and M in 2005-price yuan; L in worked days; "
          "Land and K in 2005-price yuan; x unit-free)")
    print("=" * 96)
    show(lv, ["n", "mean", "sd", "min", "p25", "median", "p75", "max"])
    print("\n" + "=" * 96)
    print("TABLE 2  LOGS  -- this is the scale every frontier figure is drawn on")
    print("=" * 96)
    show(lg, ["mean", "sd", "p0_5", "p25", "median", "p75", "p99_5", "kurt"],
         "{:>13.3f}")

    # --------------------------------------------------------------- Table 3
    # exactly the quantities the figures put on an axis, and the 0.5/99.5 limits
    # those scripts use, so a plotted range can be checked against a number
    dr = {
        "Y  (levels panel, vertical)": d.real_gvp,
        "x  (levels panel, horizontal)": d.x,
        "Y/L  (per-worker panels, vertical)": d.real_gvp / d.Laborday_impute,
        "x/L  (aggregate per labour day)": d.x / d.Laborday_impute,
        "x/Land  (aggregate per land unit)": d.x / d.Land_serv_q,
        "K/L  (fig9 KL horizontal)": d.capital_serv_q / d.Laborday_impute,
        "M/L  (fig9 ML horizontal)": d.Inter_all_real / d.Laborday_impute,
        "Land/L  (fig9 LandL horizontal)": d.Land_serv_q / d.Laborday_impute,
        "L/Y  (isoquant horizontal)": d.Laborday_impute / d.real_gvp,
        "K/Y  (isoquant vertical)": d.capital_serv_q / d.real_gvp,
        "M/Y  (isoquant vertical)": d.Inter_all_real / d.real_gvp,
        "Land/Y  (isoquant vertical)": d.Land_serv_q / d.real_gvp,
    }
    t3 = pd.DataFrame([desc(v, k) for k, v in dr.items()])
    t3["decades_spanned"] = np.log10(t3.p99_5 / t3.p0_5)
    print("\n" + "=" * 96)
    print("TABLE 3  PLOTTED RANGES -- axis limits are the p0.5 and p99.5 columns")
    print("=" * 96)
    show(t3, ["p0_5", "p25", "median", "p75", "p99_5", "decades_spanned"],
         "{:>13,.4g}")
    print("\n  'decades_spanned' = log10(p99.5 / p0.5): how many orders of magnitude the "
          "axis covers.\n  Every one of these is why the figures are on LOG axes.")

    # --------------------------------------------------------------- Table 4
    # Land, K and M are ALL 2005-price yuan by the codebook, so these ratios are
    # unit-free and should be O(0.1-1).  A UNIT error is one constant and does
    # not drift; a CONSTRUCTION error drifts and no constant repairs it.
    r = pd.DataFrame({"year": d.year,
                      "Land/M": d.Land_serv_q / d.Inter_all_real,
                      "K/M": d.capital_serv_q / d.Inter_all_real,
                      "K/Land": d.capital_serv_q / d.Land_serv_q})
    med = r.groupby("year").median()
    yrs = [y for y in (1981, 1990, 2000, 2005, 2010, 2015) if y in med.index]
    print("\n" + "=" * 96)
    print("TABLE 4  SCALE CHECK -- all three series are 2005-price yuan, so these "
          "ratios are unit-free")
    print("=" * 96)
    print(f"{'ratio':10s}" + "".join(f"{y:>12d}" for y in yrs)
          + f"{'end/start':>11s}{'max/min':>10s}{'1/median':>11s}")
    print("-" * (10 + 12 * len(yrs) + 32))
    for k in ("Land/M", "K/M", "K/Land"):
        v = med[k]
        print(f"{k:10s}" + "".join(f"{med.loc[y, k]:12.5f}" for y in yrs)
              + f"{v.iloc[-1]/v.iloc[0]:>10.2f}x{v.max()/v.min():>9.1f}x"
              + f"{1/v.median():>11,.1f}")
    print("\n  READING IT.  A pure UNIT error is ONE constant, so the ratio would be "
          "wrong by the\n  same factor in every year and max/min would be ~1.  Neither "
          "series passes that test:")
    for k, lab in (("Land/M", "Land"), ("K/M", "K")):
        v = med[k]
        print(f"   {k:7s} median {v.median():.5f}  ->  {lab} is {1/v.median():,.0f}x too "
              f"small to be a value share alongside M,")
        print(f"           and it MOVES: max/min = {v.max()/v.min():.1f}x over the panel "
              f"(min {v.min():.5f} in {int(v.idxmin())}, "
              f"max {v.max():.5f} in {int(v.idxmax())}).")
    print("\n  So the earlier 'land is a clean 1000x unit slip, capital is not' reading "
          "is NOT\n  supported on this panel: BOTH ratios drift, land downward and "
          "capital upward, and\n  K/Land moves by a factor of "
          f"{(med['K/Land'].max()/med['K/Land'].min()):.0f} between them.  Land is closer "
          "to a round\n  constant and moves less, but neither is repairable by rescaling "
          "alone.  What can be\n  said without ambiguity is only the first half: the "
          "LEVELS of both series are two to\n  three orders of magnitude away from being "
          "value shares, so they cannot be used as such.")
    print("\n  CONSEQUENCE.  The Tornqvist index uses only within-series ratios, so any "
          "constant\n  cancels exactly and x is valid whatever the level problem is.  "
          "Observed COST SHARES\n  need the LEVELS, so they remain usable for labour and "
          "intermediates and not for land\n  and capital -- which is why --weights "
          "costshare is offered only as a reported bound.")

    lv.to_csv(os.path.join(C.DQ_DIR, "desc_io_index.csv"), index=False)
    lg.to_csv(os.path.join(C.DQ_DIR, "desc_io_index_logs.csv"), index=False)
    t3.to_csv(os.path.join(C.DQ_DIR, "desc_plot_ranges.csv"), index=False)
    med.to_csv(os.path.join(C.DQ_DIR, "desc_scale_check.csv"))
    print(f"\nsaved -> dq/desc_io_index.csv, desc_io_index_logs.csv, "
          f"desc_plot_ranges.csv, desc_scale_check.csv")


if __name__ == "__main__":
    main()
