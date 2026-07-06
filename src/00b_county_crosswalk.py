# -*- coding: utf-8 -*-
"""
Build a NAME-based crosswalk between the production panel and the authoritative
cropland ag-county list (data/ag_counties_crop15.csv), because production-data
countyids are unreliable (e.g. 421103 赤壁市 is a wrong code) while the ag list
has the correct code.  Matching is on (province 2-digit, county name); a county's
ALL historical names are tried, so a renamed unit still matches.  Renames where
the NAME itself changed (华县 -> 华州区) are covered by county_corrections.RENAME_COMBINE.

Output a single review sheet for manual checking of duplicates / mismatches:
    data/county_crosswalk_review.csv   (one row per production county)
Nothing is applied to the panel here.
Run:  python src/00b_county_crosswalk.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import _common as C
import county_corrections as CC

def norm(s):
    return str(s).strip() if pd.notna(s) else ""

# ---- authoritative ag list: correct code + name + province(2-digit) ----
ag = pd.read_csv(C.AG_LIST, encoding="utf-8-sig")
ag["code"] = pd.to_numeric(ag["code"], errors="coerce")
ag = ag.dropna(subset=["code"]).copy()
ag["code"] = ag["code"].astype(int)
ag["prov2"] = ag["code"] // 10000
ag["agname"] = ag["地名"].map(norm)
# (prov2, name) -> list of ag codes  (duplicates within a province are flagged)
ag_map = ag.groupby(["prov2", "agname"])["code"].apply(list).to_dict()
ag_codeset = set(ag["code"])

# ---- production counties: each countyid, ALL names it ever had, coverage ----
base = pd.read_stata(C.BASE_DTA, convert_categoricals=False)
g = base.groupby("countyid")
prod = pd.DataFrame({
    "prod_countyid": list(g.groups.keys()),
})
prod["prov2"] = prod["prod_countyid"] // 10000
names = g["county_name"].apply(lambda s: sorted({norm(x) for x in s if norm(x)}))
prod["names"] = prod["prod_countyid"].map(names)
prod["name_latest"] = prod["prod_countyid"].map(
    base.sort_values("year").groupby("countyid")["county_name"].last().map(norm))
prod["y0"] = prod["prod_countyid"].map(g["year"].min().astype(int))
prod["y1"] = prod["prod_countyid"].map(g["year"].max().astype(int))
prod["n_years"] = prod["prod_countyid"].map(g["year"].size())
prod["gvp_nonNA"] = prod["prod_countyid"].map(g["GVP_allagr_impute"].apply(lambda s: int(s.notna().sum())))

members = set(CC.MEMBER_TO_CANON) | set(CC.MEMBER_TO_CANON.values())

rows = []
for _, r in prod.iterrows():
    cid = int(r.prod_countyid); prov2 = int(r.prov2)
    cands = []        # (agname, agcode) for any of this county's names
    for nm in r.names:
        for code in ag_map.get((prov2, nm), []):
            cands.append((nm, code))
    uniq_codes = sorted({c for _, c in cands})
    if len(uniq_codes) == 1:
        mt = "matched"; agcode = uniq_codes[0]
    elif len(uniq_codes) > 1:
        mt = "matched_DUP"; agcode = uniq_codes[0]
    else:
        mt = "combine_table" if cid in members else "NO_MATCH"
        agcode = CC.to_canonical(cid) if cid in members else ""
    rows.append(dict(
        prod_countyid=cid, prov2=prov2, name_latest=r.name_latest,
        all_names="|".join(r.names), y0=r.y0, y1=r.y1, n_years=r.n_years,
        gvp_nonNA=r.gvp_nonNA, match_type=mt, ag_code=agcode,
        ag_name=(ag.set_index("code")["agname"].get(agcode, "") if agcode != "" else ""),
        n_ag_candidates=len(uniq_codes),
        code_differs=(agcode != "" and agcode != cid)))

cw = pd.DataFrame(rows).sort_values(
    ["match_type", "prov2", "prod_countyid"],
    key=lambda s: s.map({"NO_MATCH": 0, "matched_DUP": 1, "combine_table": 2, "matched": 3})
    if s.name == "match_type" else s)
out = os.path.join(C.DATA, "county_crosswalk_review.csv")
cw.to_csv(out, index=False, encoding="utf-8-sig")

# ag-list counties with NO production match (we have no data for them)
matched_ag = set(pd.to_numeric(cw["ag_code"], errors="coerce").dropna().astype(int))
unmatched_ag = ag[~ag["code"].isin(matched_ag)][["code", "prov2", "agname", "省级", "crop_pct"]]
unmatched_ag.to_csv(os.path.join(C.DATA, "ag_counties_no_production.csv"),
                    index=False, encoding="utf-8-sig")

print("Crosswalk (production -> ag by name):")
print(cw["match_type"].value_counts().to_string())
print(f"\n  production counties: {len(cw)}")
print(f"  code differs from ag (countyid was wrong): {int(cw['code_differs'].sum())}")
print(f"  ag-list counties with NO production data : {len(unmatched_ag)} / {len(ag)}")
print(f"\nSaved -> {os.path.relpath(out, C.ROOT)}  (review NO_MATCH + matched_DUP rows first)")
print(f"Saved -> data/ag_counties_no_production.csv")
