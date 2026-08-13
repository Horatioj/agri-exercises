# -*- coding: utf-8 -*-
"""
STEP 2 — apply the resolved roster (01) to the base panel to produce the
CORRECTED raw county I-O data: right countyid + county_name, merges summed,
prefecture and urban districts removed.  Output feeds the cleaning step (03).

SCOPE: this keeps EVERY resolved rural county-level unit (~2,570), not just the
cropland-defined agricultural ones.  The cropland>=15% membership is carried as
an `ag_county` 0/1 FLAG rather than applied as a filter, so that the cleaned
panel is a reusable product for any county-level work, and the agricultural
restriction is a one-line selection made by whoever needs it:

    python                 _tfp.load_panel()              # ag only (analysis default)
    python                 _tfp.load_panel(ag_only=False) # all counties
    Stata                  keep if ag_county == 1

The flag is county-level and time-invariant, so a county that qualifies is
flagged for ALL its years — there is no year-by-year entry/exit.

  data/io_raw_corrected.dta / .csv   (countyid, county_name, SID, state, year,
        ag_county, GVP_allagr_impute, Laborday_impute, Land_serv_q,
        capital_serv_q, Inter_all_real, Inter_all_nom)
Run:  python src/02_apply_resolution.py
      python src/02_apply_resolution.py --ag-only   # drop non-ag units outright
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import _common as C

VALVARS = ["GVP_allagr_impute", "Laborday_impute", "agr_labor_num", "Land_serv_q",
           "capital_serv_q", "Inter_all_real", "Inter_all_nom"]
KEEP = {"keep_adopt_name", "recode_name_match", "combine",
        "recode_prefix2_REVIEW", "keep_own", "keep_rural"}

res = pd.read_csv(os.path.join(C.DATA, "roster_resolution.csv"), encoding="utf-8-sig")
res = res[res["action"].isin(KEEP)].copy()
res["final_code"] = res["final_code"].astype(int)
xwalk = res.set_index("prod_countyid")[["final_code", "final_name", "action"]]

base = pd.read_stata(C.BASE_DTA, convert_categoricals=False)
base = base[base["countyid"].isin(xwalk.index)].copy()
base["final_code"] = base["countyid"].map(xwalk["final_code"])
base["final_name"] = base["countyid"].map(xwalk["final_name"])

# province/state metadata from the new (final) code's province
meta = base.dropna(subset=["SID"]).drop_duplicates("final_code").set_index("final_code")[["SID", "state"]]

# aggregate to (final_code, year): sum additive vars (merges); min_count=1 keeps NA
agg = (base.groupby(["final_code", "year"])[VALVARS].sum(min_count=1).reset_index())
agg["county_name"] = agg["final_code"].map(res.drop_duplicates("final_code").set_index("final_code")["final_name"])
agg["SID"] = agg["final_code"].map(meta["SID"])
agg["state"] = agg["final_code"].map(meta["state"])
agg = agg.rename(columns={"final_code": "countyid"})
agg = agg[["countyid", "county_name", "SID", "state", "year"] + VALVARS].sort_values(["countyid", "year"])
agg["countyid"] = agg["countyid"].astype(int)
agg["year"] = agg["year"].astype(int)

# ---------------------------------------------------------------------------
# AGRICULTURAL-COUNTY FLAG (not a filter -- see the scope note in the docstring)
# `ag_county` = cropland >= 15% in the ag_counties_crop15 set, via the
# `in_ag_list` column of county_roster_final.csv.
# ---------------------------------------------------------------------------
roster = pd.read_csv(os.path.join(C.DATA, "county_roster_final.csv"), encoding="utf-8-sig")
ag_codes = set(roster.loc[roster["in_ag_list"] == True, "final_code"].astype(int))
agg["ag_county"] = agg["countyid"].isin(ag_codes).astype("int8")
agg = agg[["countyid", "county_name", "SID", "state", "year", "ag_county"] + VALVARS]

n_ag = agg.loc[agg.ag_county == 1, "countyid"].nunique()
n_all = agg["countyid"].nunique()
print(f"\nAG-COUNTY FLAG: {n_ag:,d} of {n_all:,d} counties are cropland>=15% agricultural "
      f"({n_all - n_ag:,d} flagged 0, retained for reuse)")

if "--ag-only" in sys.argv:
    agg = agg[agg.ag_county == 1].copy()
    print(f"  --ag-only: dropped the {n_all - n_ag:,d} non-ag units from the output")

agg.to_csv(os.path.join(C.DATA, "io_raw_corrected.csv"), index=False, encoding="utf-8-sig")
agg.to_stata(os.path.join(C.DATA, "io_raw_corrected.dta"), write_index=False, version=118)

merged = res[res["prod_countyid"] != res["final_code"]]
print(f"Corrected raw I-O data: {agg['countyid'].nunique()} counties, {len(agg)} rows, "
      f"{int(agg.year.min())}-{int(agg.year.max())}")
print(f"  production codes re-coded/merged into a different final code: {len(merged)}")
print(f"  final codes built from >1 production code (summed): "
      f"{int((res.groupby('final_code')['prod_countyid'].nunique() > 1).sum())}")
print("Saved -> data/io_raw_corrected.dta, data/io_raw_corrected.csv")
print("\nnon-NA by variable:")
for v in VALVARS:
    print(f"   {v:18s} {int(agg[v].notna().sum())}")
