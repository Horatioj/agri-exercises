# -*- coding: utf-8 -*-
"""
STEP 0 — Assemble the base county production panel from THREE sources, then save
to data/base_panel.dta for the cleaning step (01_clean_panel.py).

  output GVP      <- data/agr_GVP.dta        (GVP_allagr_impute, 10k RMB)
  labour          <- data/agr_labor.dta      (agr_labor1 -> labor number)
  land/capital/   <- data/cty_prod_account_agg_forAXN.dta
   intermediate      (Land_serv_q, capital_serv_q, Inter_all_real, Inter_all_nom)

The cropland>=15% agricultural-county filter (data/ag_counties_crop15.csv) is NOT
applied here -- it is applied in 01 AFTER administrative harmonisation, so old
county codes are first mapped to their canonical (2010-list) codes.  County names
may differ across sources/years; everything is merged on (countyid, year) and the
name is taken from whichever source has it.
Run:  python src/00_build_base_panel.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import _common as C

KEYS = ["countyid", "year"]
META = ["SID", "state", "county_name"]

gvp = pd.read_stata(C.GVP_DTA, convert_categoricals=False).drop_duplicates(KEYS)
lab = (pd.read_stata(C.LABOR_DTA, convert_categoricals=False).drop_duplicates(KEYS)
         .rename(columns={"agr_labor1": "Laborday_impute"}))
agg = pd.read_stata(C.AGG_DTA, convert_categoricals=False).drop_duplicates(KEYS)
print(f"GVP {gvp.shape} | labour {lab.shape} | agg {agg.shape}")

agg3 = agg[KEYS + META + ["Land_serv_q", "capital_serv_q", "Inter_all_real", "Inter_all_nom"]]
g    = gvp[KEYS + META + ["GVP_allagr_impute"]]

# outer-merge GVP + the 3 agg vars, coalesce metadata
base = g.merge(agg3, on=KEYS, how="outer", suffixes=("", "_a"))
for c in META:
    base[c] = base[c].fillna(base[c + "_a"])
base = base.drop(columns=[c + "_a" for c in META])

# add labour (key only), then backfill metadata for labour-only county-years
base = base.merge(lab[KEYS + ["Laborday_impute"]], on=KEYS, how="outer")
meta = (pd.concat([g[["countyid"] + META], agg3[["countyid"] + META]])
          .dropna(subset=["SID"]).drop_duplicates("countyid"))
base = base.merge(meta.rename(columns={c: c + "_m" for c in META}), on="countyid", how="left")
for c in META:
    base[c] = base[c].fillna(base[c + "_m"])
base = base.drop(columns=[c + "_m" for c in META])

cols = KEYS + META + ["GVP_allagr_impute", "Laborday_impute",
                      "Land_serv_q", "capital_serv_q", "Inter_all_real", "Inter_all_nom"]
base = base[cols].sort_values(KEYS).reset_index(drop=True)
base["countyid"] = base["countyid"].astype(int)
base["year"] = base["year"].astype(int)

base.to_stata(C.BASE_DTA, write_index=False, version=118)
print(f"Saved base panel -> {os.path.relpath(C.BASE_DTA, C.ROOT)}  "
      f"({len(base)} rows, {base.countyid.nunique()} counties, "
      f"{int(base.year.min())}-{int(base.year.max())})")
print("non-NA by variable:")
for v in ["GVP_allagr_impute", "Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]:
    print(f"   {v:18s} {int(base[v].notna().sum())}")
