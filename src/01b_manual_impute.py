# -*- coding: utf-8 -*-
"""
STEP 1b — manual multi-year imputation (provincial-trend-anchored, MEDIAN reference).

Reads data/manual_impute_list.csv — the abnormal stretches YOU identified from the
county-review SVGs (columns: countyid, var, year_start, year_end[, note]) — and
imputes every year in [year_start, year_end] for that county-variable by riding the
provincial MEDIAN trend, pinned to the last-good anchor (year_start-1) and the
first-good anchor (year_end+1):

    ln C(t) = R(t) + [ d0 + (d3 - d0) * (t - t0)/(t3 - t0) ]

  R(t) = median of ln(value) over the OTHER counties of the same province (good
         shape reference); d0,d3 = the county's offset from R at the two anchors.
  If R(t) is missing for some year, that year falls back to plain log-linear
  interpolation between the anchors.

Updates src/clean/county_panel_clean.csv IN PLACE: originals are preserved in
<var>_raw, the cell's <var>_imp flag is set True, and every change is appended to
src/clean/manual_impute_log.csv (fully reversible).  Does NOT touch caliber years
implicitly — you control exactly which years are imputed.
Run:  python src/01b_manual_impute.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import _common as C

LIST_CSV = os.path.join(C.DATA, "manual_impute_list.csv")
LOG_CSV  = os.path.join(C.CLEAN_DIR, "manual_impute_log.csv")

spec = pd.read_csv(LIST_CSV, encoding="utf-8-sig")
spec = spec[spec["countyid"].astype(str).str.strip().str.isdigit()].copy()
spec["countyid"] = spec["countyid"].astype(int)
# case-insensitive var match (the list may use inter_all_real / Inter_all_real, etc.)
_lc2var = {v.lower(): v for v in C.IO_VARS}
spec["var"] = spec["var"].astype(str).str.strip().str.lower().map(_lc2var)
bad_var = spec["var"].isna().sum()
spec = spec[spec["var"].notna()]
for c in ("year_start", "year_end"):
    spec[c] = spec[c].astype(int)
if bad_var:
    print(f"  (dropped {bad_var} rows with an unrecognised var name)")

m = pd.read_csv(C.CLEAN_PANEL)
Rmed = {v: C.provincial_median_ln(m, v) for v in spec["var"].unique()}   # same ref as 01
log_rows, applied, skipped = [], 0, []

for _, r in spec.iterrows():
    cid, var, ys, ye = int(r.countyid), r["var"], int(r.year_start), int(r.year_end)
    if cid not in m["countyid"].values:
        skipped.append((cid, var, "county not in panel")); continue
    sid = int(m.loc[m.countyid == cid, "SID"].iloc[0])
    cser = m[m.countyid == cid].set_index("year")[var]
    Rfull = Rmed[var]
    Rsid = Rfull.loc[sid] if sid in Rfull.index.get_level_values(0) else None
    imp = C.impute_stretch(cser, Rsid, ys, ye)          # SHARED provincial-trend method
    if imp is None:
        skipped.append((cid, var, f"missing/invalid anchor {ys-1} or {ye+1}")); continue
    for t, (new, method) in imp.items():
        orig = cser.get(t)
        m.loc[(m.countyid == cid) & (m.year == t), var] = new
        m.loc[(m.countyid == cid) & (m.year == t), var + "_imp"] = True
        log_rows.append(dict(countyid=cid, SID=sid, var=var, year=t,
                             orig_value=None if (orig is None or orig != orig) else float(orig),
                             imputed_value=new, method=method, anchor_lo=ys - 1, anchor_hi=ye + 1))
        applied += 1

if log_rows:
    newlog = pd.DataFrame(log_rows)
    if os.path.exists(LOG_CSV):
        newlog = pd.concat([pd.read_csv(LOG_CSV), newlog], ignore_index=True)
    newlog.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
    m.to_csv(C.CLEAN_PANEL, index=False)

print(f"Manual imputation: {len(spec)} stretches, {applied} county-years imputed.")
if log_rows:
    print("  by method:", pd.DataFrame(log_rows)["method"].value_counts().to_dict())
    print(f"  panel updated -> {os.path.relpath(C.CLEAN_PANEL, C.ROOT)}; log -> {os.path.relpath(LOG_CSV, C.ROOT)}")
if skipped:
    print(f"  SKIPPED {len(skipped)} (fix the list and re-run):")
    for s in skipped[:20]:
        print("   ", s)
