# -*- coding: utf-8 -*-
"""
STEP 0e — apply the resolved roster (00d) to the base panel to produce the
CORRECTED raw agricultural I-O data: right countyid + county_name, merges summed,
prefecture/urban-non-ag units removed.  Output feeds the cleaning step (01).

  data/io_raw_corrected.dta / .csv   (countyid, county_name, SID, state, year,
        GVP_allagr_impute, Laborday_impute, Land_serv_q, capital_serv_q,
        Inter_all_real, Inter_all_nom)
Run:  python src/00e_apply_resolution.py
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
