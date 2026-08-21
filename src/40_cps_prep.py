# -*- coding: utf-8 -*-
"""
STEP 40 — CPS (2020) step 1: scalar aggregate input index for the ~1,800-county
agricultural-county sample, plus the analysis frame (y, x, w) the
decomposition consumes.

INPUT INDEX
  Tornqvist with FIXED weights.  With constant shares the Tornqvist
  ln(x_t/x_s) = SUM_k 0.5(s_kt+s_ks) ln(X_kt/X_ks) collapses to
  SUM_k beta_k ln(X_kt/X_ks), so the level index is
        ln x = SUM_k beta_k ln X_k ,  SUM_k beta_k = 1
  i.e. a CRS geometric (Cobb-Douglas) aggregator.

OUTPUT
  src/clean/cps/cps_frame.csv   countyid, year, y, x, w1(GDD), w2(rzsm 0-28cm)
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import _common as C

OUT = os.path.join(C.CLEAN_DIR, "cps")
os.makedirs(OUT, exist_ok=True)

# fixed CRS weights from the converged BC92 SFA -- SEE PLACEHOLDER WARNING
BETA = {"Laborday_impute": 0.439, "Inter_all_real": 0.436,
        "capital_serv_q": 0.069, "Land_serv_q": 0.056}
Y_VAR = "real_gvp"
W1, W2 = "gdd_growing_season", "sm_0_28"

BANNER = """
*******************************************************************************
 PLACEHOLDER INPUT WEIGHTS
   Tornqvist weights beta = {b}
   estimated by BC92 SFA on the ~2,600-county sample, applied here to the
   1,800-county AGRICULTURAL-COUNTY filter.  The ag filter selects on economic
   structure, so the input mix plausibly differs systematically.
   => Re-estimate SFA / cost shares on the 1,800-county sample before these
      numbers are used for anything final.  This is the next task, not a
      someday item.
*******************************************************************************
"""

def main():
    print(BANNER.format(b=BETA))
    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    need = [Y_VAR, W1, W2] + list(BETA)
    n0 = len(d)
    d = d.dropna(subset=need)
    d = d[(d[[Y_VAR] + list(BETA)] > 0).all(axis=1)]
    print(f"rows {n0:,d} -> {len(d):,d} usable (y, 4 inputs, both weather all present & >0)")

    s = sum(BETA.values())
    if abs(s - 1.0) > 1e-9:
        print(f"NOTE: weights sum to {s:.6f}, renormalising to impose CRS")
    b = {k: v / s for k, v in BETA.items()}

    d["x"] = np.exp(sum(b[k] * np.log(d[k]) for k in b))    # CRS geometric aggregate
    d = d.rename(columns={Y_VAR: "y", W1: "w1", W2: "w2"})
    frame = d[["countyid", "year", "y", "x", "w1", "w2"]].copy()
    frame = frame.sort_values(["year", "countyid"]).reset_index(drop=True)
    frame.to_csv(os.path.join(OUT, "cps_frame.csv"), index=False)

    print(f"\ncounties {frame.countyid.nunique():,d}  years {frame.year.min()}-{frame.year.max()}"
          f"  county-years {len(frame):,d}")
    print(f"  y  (real GVP)  median {frame.y.median():,.0f}")
    print(f"  x  (agg input) median {frame.x.median():,.0f}")
    print(f"  w1 (GDD)       {frame.w1.min():.0f} - {frame.w1.max():.0f}")
    print(f"  w2 (rzsm 0-28) {frame.w2.min():.3f} - {frame.w2.max():.3f}")
    ky = frame.groupby("year").countyid.nunique()
    print(f"\n|K_j| by year: min {ky.min():,d}  median {int(ky.median()):,d}  max {ky.max():,d}")
    print("  (unbalanced: |K_j| varies by year, as specified)")

    # adjacent-pair evaluation set sizes  A = K_{t-1} ∩ K_t
    yrs = sorted(frame.year.unique())
    sets = {y: set(frame.loc[frame.year == y, "countyid"]) for y in yrs}
    rows = [(a, b_, len(sets[a] & sets[b_])) for a, b_ in zip(yrs[:-1], yrs[1:]) if b_ == a + 1]
    A = pd.DataFrame(rows, columns=["year_prev", "year", "n_eval"])
    A.to_csv(os.path.join(OUT, "eval_set_sizes.csv"), index=False)
    print(f"\nadjacent pairs {len(A)}; |A| min {A.n_eval.min():,d} "
          f"median {int(A.n_eval.median()):,d} max {A.n_eval.max():,d}")
    print(f"total county-pairs to decompose: {A.n_eval.sum():,d}")
    print(f"saved -> {os.path.join(OUT,'cps_frame.csv')}")


if __name__ == "__main__":
    main()
