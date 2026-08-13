# -*- coding: utf-8 -*-
"""
Cleaning audit: how much of the panel did 03_clean_panel.py actually touch?

The unit of account is the CELL (county-year x variable), because every rule in 03
acts on one variable at a time.  A county-year is reported as "touched" if ANY of
its five I-O variables was modified, so the two numbers bracket the footprint.

Target: total modified cells should sit under ~10%.  If it does not, the honest
lever is the DETECTOR threshold (JUMP5 / TAU_END / TAU_UP / TAU_DOWN in 03), not a
post-hoc filter -- a detector that fires on a tenth of the data is describing
normal variation, not anomalies.

Reads only outputs, changes nothing:
  src/clean/county_panel_clean.csv   <var>_raw vs <var>, <var>_imp flags
  src/clean/anomaly_log.csv          spike-and-revert + manual imputations
  src/dq/boundary_run_rescale.csv    boundary-run splices
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import _common as C

RTOL = 1e-9


def main():
    m = pd.read_csv(C.CLEAN_PANEL)
    V = C.IO_VARS
    n_cy = len(m)
    n_cells = n_cy * len(V)
    print(f"panel: {n_cy:,d} county-years x {len(V)} variables = {n_cells:,d} cells")
    print(f"       {m.countyid.nunique():,d} counties, {m.year.min()}-{m.year.max()}\n")

    rows = []
    touched = np.zeros(len(m), dtype=bool)          # any variable modified
    for v in V:
        raw = pd.to_numeric(m[v + "_raw"], errors="coerce")
        cur = pd.to_numeric(m[v], errors="coerce")
        obs = raw.notna()                            # cells that carried a raw value
        imp = m[v + "_imp"].fillna(False).astype(bool)
        # value changed but not flagged as an imputation => a rescale/unit fix
        both = raw.notna() & cur.notna()
        changed = both & (np.abs(cur - raw) > RTOL * np.abs(raw).clip(lower=1))
        rescaled = changed & ~imp
        nulled = raw.notna() & cur.isna()            # raw existed, now NA
        rows.append(dict(variable=v, raw_obs=int(obs.sum()),
                         imputed=int((imp & both).sum()),
                         rescaled=int(rescaled.sum()),
                         set_to_NA=int(nulled.sum())))
        touched |= (imp & both).to_numpy() | rescaled.to_numpy() | nulled.to_numpy()

    t = pd.DataFrame(rows)
    t["modified"] = t.imputed + t.rescaled + t.set_to_NA
    t["pct_of_raw_obs"] = 100 * t.modified / t.raw_obs.clip(lower=1)
    t["pct_of_all_cells"] = 100 * t.modified / n_cy
    print("BY VARIABLE (cells)")
    print(t[["variable", "raw_obs", "imputed", "rescaled", "set_to_NA",
             "modified", "pct_of_raw_obs", "pct_of_all_cells"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    tot_mod = int(t.modified.sum()); tot_raw = int(t.raw_obs.sum())
    print(f"\nTOTAL modified cells: {tot_mod:,d}")
    print(f"   as a share of cells that HAD a raw value ({tot_raw:,d}): "
          f"{100*tot_mod/tot_raw:.2f}%")
    print(f"   as a share of all {n_cells:,d} cells:                    "
          f"{100*tot_mod/n_cells:.2f}%")
    print(f"county-years with >=1 variable modified: {int(touched.sum()):,d} "
          f"({100*touched.mean():.2f}% of {n_cy:,d})")

    verdict = "WITHIN" if 100*tot_mod/tot_raw <= 10 else "ABOVE"
    print(f"\n=> {verdict} the 10% target "
          f"({100*tot_mod/tot_raw:.2f}% of observed cells)")

    if os.path.exists(C.ANOMALY_LOG):
        a = pd.read_csv(C.ANOMALY_LOG)
        print("\nanomaly log by action:", a["action"].value_counts().to_dict())
        if "method" in a.columns:
            print("            by method:", a["method"].value_counts().to_dict())
    f = os.path.join(C.DQ_DIR, "boundary_run_rescale.csv")
    if os.path.exists(f):
        b = pd.read_csv(f)
        print(f"boundary-run splices applied: {len(b):,d} cells in "
              f"{b.drop_duplicates(['countyid','var']).shape[0]:,d} runs")

    t.to_csv(os.path.join(C.DQ_DIR, "cleaning_audit.csv"), index=False)
    print(f"\nsaved -> dq/cleaning_audit.csv")


if __name__ == "__main__":
    main()
