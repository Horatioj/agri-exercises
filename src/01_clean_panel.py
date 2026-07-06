# -*- coding: utf-8 -*-
"""
STEP 1 — Clean the county-level annual agricultural input-output panel.

Pipeline (fixed order; every change is reversible — originals kept as <var>_raw):
  1. Load data/cty_prod_account_agg_forAXN.dta, drop duplicate (countyid, year).
  2. GEOGRAPHY FILTER — keep agricultural rural counties; drop urban districts
     (names ending 区/辖/郊) and prefecture/municipality aggregate rows
     (countyid % 100 == 0).  See data/农业生产数据集变量代码本 + 数据说明.md.
  3. ÷10,000 UNIT-ENTRY FIX — county-years where the nominal value variables
     (GVP, Inter_all_nom, Inter_all_real) were keyed in yuan instead of wan-yuan
     and revert the next year (co-moving GVP & Inter) -> multiply by 10,000.
  4. DEFLATE — real_gvp = GVP_allagr_impute / PPI_CCD_2005 (province PPI, 2005
     prices, Hebei 2005 = 1), merged on (SID, year) from priceindex_province.dta.
  5. CALIBER YEARS — detect province-wide synchronized breaks (statistical-caliber
     / definition changes) so they are NOT pointwise-imputed.
  6. SPIKE DETECTION + IMPUTATION (per county, per variable, on ln series):
       spike-and-revert ONLY: |Δln_t| > ln(5) AND |Δln_{t+1}| > ln(5) with opposite
       signs — i.e. the value jumps to >=5x (or <=1/5) and reverts within one year.
       Flagged points are imputed by log-linear interpolation between the nearest
       non-flagged neighbours.  (No Hampel / level filter.)

Outputs:
  src/clean/county_panel_clean.csv  — cleaned panel; <var> cleaned, <var>_raw
                                       original, <var>_imp flag.
  src/clean/anomaly_log.csv          — one row per (county, var, flagged year).
  src/dq/caliber_candidate_years.csv — province synchronized-break candidates.
  src/dq/dropped_units.csv           — units removed by the geography filter.
Run:  python src/01_clean_panel.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import _common as C
import county_corrections as CC

C.ensure_dirs()
# Asymmetric spike-and-revert thresholds: an UPWARD spike (value jumps up then back)
# is almost always a data error -> stricter 3x; a DOWNWARD spike (drop then recover)
# can be a real bad-harvest/weather year -> looser 5x.
TAU_UP   = np.log(3.0)
TAU_DOWN = np.log(5.0)
JUMP_THR = 0.30      # |Δln| threshold for the caliber synchronized-break scan
JUMP_SHARE = 0.35    # share of county jumps in a year to call it a caliber candidate
JUMP_MINN = 10       # require >=this many county jumps in a year before a caliber call

np.seterr(divide="ignore", invalid="ignore")   # ln(0)/ln(neg) are masked by hand below

# ============================================================================
# 1. LOAD + DEDUPE
# ============================================================================
agg = (pd.read_stata(C.IO_RAW, convert_categoricals=False)
         .drop_duplicates(["countyid", "year"])
         .sort_values(["countyid", "year"]).reset_index(drop=True))
n0, c0 = len(agg), agg["countyid"].nunique()
print(f"1. Loaded io_raw_corrected: {n0} rows, {c0} counties, "
      f"years {int(agg.year.min())}-{int(agg.year.max())}")
# County codes/names, administrative merges, ag-county selection and the typo fixes
# were already applied upstream in 00d/00e (data/roster_resolution.csv).

# Hainan (SID 46) founded 1988 -> drop pre-1988 rows (matters for provincial trends).
n_hn = int(((agg["SID"] == 46) & (agg["year"] < 1988)).sum())
agg = agg[~((agg["SID"] == 46) & (agg["year"] < 1988))].copy()
print(f"1c. Hainan: dropped {n_hn} pre-1988 rows (founded 1988)")

# ============================================================================
# (County selection / harmonisation already done in 00d/00e.)
# 3. ÷10,000 UNIT-ENTRY FIX (on nominal value vars, BEFORE deflation)
# ============================================================================
NOM = ["GVP_allagr_impute", "Inter_all_nom", "Inter_all_real"]

def neighbor_factor(frame, col):
    """geomean(prev, next) / value_t for interior consecutive years; else NaN."""
    prev = frame.groupby("countyid")[col].shift(1)
    nxt  = frame.groupby("countyid")[col].shift(-1)
    yp   = frame.groupby("countyid")["year"].shift(1)
    yn   = frame.groupby("countyid")["year"].shift(-1)
    ok = (yp == frame["year"]-1) & (yn == frame["year"]+1) & (prev > 0) & (nxt > 0) & (frame[col] > 0)
    return (np.sqrt(prev*nxt) / frame[col]).where(ok)

band = lambda f: (f > 2000) & (f < 50000)          # factor in a wide band around 10,000
unit_flag = band(neighbor_factor(agg, "GVP_allagr_impute")) & band(neighbor_factor(agg, "Inter_all_real"))
unit_flag = unit_flag.fillna(False)
unit_fixed = agg.loc[unit_flag, ["countyid", "county_name", "year"]].copy()
agg.loc[unit_flag, NOM] = agg.loc[unit_flag, NOM] * 10000
print(f"3. ÷10,000 unit-entry fix applied to {int(unit_flag.sum())} county-years: "
      f"{sorted(unit_fixed['county_name'].unique().tolist())}")

# ============================================================================
# 4. DEFLATE GVP with province PPI
# ============================================================================
pi = pd.read_stata(C.PRICE_DTA, convert_categoricals=False)[["SID", "year", C.PPI_COL]]
for d in (agg, pi):                                    # align merge-key dtypes
    d["SID"] = d["SID"].astype("int64"); d["year"] = d["year"].astype("int64")
agg = agg.merge(pi, on=["SID", "year"], how="left")
agg["real_gvp"] = agg["GVP_allagr_impute"] / agg[C.PPI_COL]
miss = agg["real_gvp"].isna().sum() - agg["GVP_allagr_impute"].isna().sum()
print(f"4. Deflated GVP -> real_gvp using province {C.PPI_COL} "
      f"(rows with GVP but no deflator: {max(miss,0)})")

# ============================================================================
# 4b. ERROR -> NA: zeros in inputs, and tiny "low-then-jump" values (>1000x below
#     the county's own median of positive values) are data errors, not real
#     production -> set to NA (e.g. 510184 崇州市 land 35.2 then 128513).
# ============================================================================
for v in C.IO_VARS:
    n_zero = int((agg[v] == 0).sum())
    agg.loc[agg[v] == 0, v] = np.nan
    # implausibly small for these large-magnitude vars (e.g. 久治县 land all <1)
    sub1 = (agg[v] > 0) & (agg[v] < 1)
    n_sub1 = int(sub1.sum())
    agg.loc[sub1, v] = np.nan
    med = agg.groupby("countyid")[v].transform(lambda s: s[s > 0].median())
    low = (agg[v] < med / 1000.0)
    n_low = int(low.sum())
    agg.loc[low, v] = np.nan
    print(f"4b. {v:18s}: {n_zero:4d} zeros, {n_sub1:4d} <1, {n_low:4d} low-then-jump -> NA")

# ============================================================================
# 4c. BEGINNING/END run -> NA: trim a RUN of leading (or trailing) years that sit
#     >3x off the local stable median -- catches multi-year boundary spikes with only
#     one neighbour, e.g. 654002 伊宁市 land 603 -> 4227 -> 70000, 玛多县 land, 临淄区.
# ============================================================================
TAU_END = np.log(3.0)
JUMP5 = np.log(5.0)
def trim_run_idx(g, v):
    """NA a SHORT leading/trailing run (<=5 yrs) that is separated from the stable core
    by a BIG (>5x) year-on-year jump AND sits >3x off the post-jump core median.
    Anchored to the discontinuity (not the median), so genuine gradual growth -- even
    100x over the panel like capital -- is NOT trimmed.  e.g. 伊宁市 land 603/712/672/
    4227 then ~70000, 临淄区 1981-83."""
    s = g.sort_values("year")[v]
    s = s.where(s > 0).dropna()
    bad = []
    n = len(s)
    if n >= 8:
        vals = s.to_numpy(); idx = s.index.to_numpy(); lv = np.log(vals)
        jstar = 0                                        # leading: LAST >5x jump in yrs 1..5
        for j in range(1, min(6, n - 3)):
            if abs(lv[j] - lv[j-1]) > JUMP5:
                jstar = j
        if jstar and all(abs(lv[k] - np.log(np.median(vals[jstar:jstar+5]))) > TAU_END
                         for k in range(jstar)):
            bad.extend(idx[:jstar].tolist())
        jstar = 0                                        # trailing: LAST >5x jump near end
        for j in range(1, min(6, n - 3)):
            if abs(lv[n-j] - lv[n-j-1]) > JUMP5:
                jstar = j
        if jstar and all(abs(lv[n-1-k] - np.log(np.median(vals[n-jstar-5:n-jstar]))) > TAU_END
                         for k in range(jstar)):
            bad.extend(idx[n-jstar:].tolist())
    return bad
for v in C.IO_VARS:
    flags = [i for _, g in agg.groupby("countyid") for i in trim_run_idx(g, v)]
    agg.loc[flags, v] = np.nan
    print(f"4c. {v:18s}: {len(flags):4d} boundary-run years -> NA")

# ============================================================================
# 4d. CROSS-COUNTY placeholder -> NA: a (province, year) where >50% of counties share
#     the EXACT same value is a filled-in placeholder, not real data (e.g. Qinghai
#     capital_serv_q identical across counties 1981-84 then jumps 1985).
# ============================================================================
for v in C.IO_VARS:
    flags = []
    for (sid, yr), s in agg.groupby(["SID", "year"])[v]:
        vv = s.dropna()
        if len(vv) >= 5:
            mode = vv.mode()
            if len(mode) and (vv == mode.iloc[0]).mean() > 0.5:
                flags.extend(s.index[s == mode.iloc[0]].tolist())
    agg.loc[flags, v] = np.nan
    print(f"4d. {v:18s}: {len(flags):4d} cross-county placeholder cells -> NA")

# keep a raw (pre-impute) copy of each I-O variable
for v in C.IO_VARS:
    agg[v + "_raw"] = agg[v].astype(float)
    agg[v + "_imp"] = False

# ============================================================================
# 5. CALIBER YEARS — province synchronized breaks (not pointwise-imputed)
# ============================================================================
def dln_consecutive(df, col):
    d = df[["countyid", "year", col]].copy()
    d = d[d[col] > 0]
    d["ldiff"] = d.groupby("countyid")[col].transform(lambda s: np.log(s).diff())
    d["dyr"]   = d.groupby("countyid")["year"].diff()
    d.loc[d["dyr"] != 1, "ldiff"] = np.nan
    return d.dropna(subset=["ldiff"])

cal_rows, cal_strong = [], set()
for sid, dp in agg.groupby("SID"):
    name = dp["state"].iloc[0]
    flagged = {}
    for v in C.IO_VARS:
        d = dln_consecutive(dp, v)
        grp = d.groupby("year")["ldiff"]
        share = grp.apply(lambda s: (s.abs() > JUMP_THR).mean())
        nobs  = grp.size()
        for y, sv in share.items():
            if sv > JUMP_SHARE and nobs.get(y, 0) >= JUMP_MINN:
                flagged.setdefault(int(y), []).append(v)
    for y, vs in flagged.items():
        strong = len(vs) >= 2
        cal_rows.append({"SID": sid, "state": name, "year": y, "n_vars_flagged": len(vs),
                         "vars": "|".join(C.VLAB[x] for x in vs), "strong": strong})
        if strong:
            cal_strong.add((sid, y))
pd.DataFrame(cal_rows).sort_values(["SID", "year"]).to_csv(C.CALIBER_CSV, index=False)
print(f"5. Caliber scan: {len(cal_strong)} strong (>=2 vars) province-year breaks "
      f"-> {os.path.relpath(C.CALIBER_CSV, C.ROOT)} (held, not imputed)")

# ============================================================================
# 6. SPIKE DETECTION (single 1-year + double 2-year plateau spike-and-revert)
#    + PROVINCIAL-TREND imputation (C.impute_stretch, median reference).  The
#    SAME imputation method is used by the manual multi-year step (01b).
# ============================================================================
PLATEAU_TOL = np.log(2.0)   # plateau years may differ up to 2x (e.g. 连南 2005/06)
def _revert(din, dout):
    """Spike-and-revert test with ASYMMETRIC thresholds: 'din' rises into the
    excursion, 'dout' exits it (opposite sign).  An up-excursion (din>0) uses the
    stricter TAU_UP; a down-excursion uses the looser TAU_DOWN."""
    if not (np.isfinite(din) and np.isfinite(dout)) or np.sign(din) == np.sign(dout) or din == 0:
        return False
    tau = TAU_UP if din > 0 else TAU_DOWN
    return abs(din) > tau and abs(dout) > tau

def detect_stretches(yrs, val):
    """Spike-and-revert stretches: 1-year single, 2-year double, 3-year triple plateau."""
    n = len(val)
    L = np.where(val > 0, np.log(val), np.nan)
    def cons(a, b):   # years a..b consecutive and all finite
        return all(yrs[k+1]-yrs[k] == 1 for k in range(a, b)) and np.isfinite(L[a:b+1]).all()
    out = []; i = 1
    while i < n - 1:
        if cons(i-1, i+1) and _revert(L[i]-L[i-1], L[i+1]-L[i]):
            out.append((int(yrs[i]), int(yrs[i]))); i += 1; continue                 # single
        if i < n-2 and cons(i-1, i+2) and abs(L[i]-L[i+1]) < PLATEAU_TOL \
                and _revert(L[i]-L[i-1], L[i+2]-L[i+1]):
            out.append((int(yrs[i]), int(yrs[i+1]))); i += 2; continue               # double
        if i < n-3 and cons(i-1, i+3) and abs(L[i]-L[i+1]) < PLATEAU_TOL \
                and abs(L[i+1]-L[i+2]) < PLATEAU_TOL and _revert(L[i]-L[i-1], L[i+3]-L[i+2]):
            out.append((int(yrs[i]), int(yrs[i+2]))); i += 3; continue               # triple
        i += 1
    return out

Rmed = {v: C.provincial_median_ln(agg, v) for v in C.IO_VARS}   # robust, pre-impute
log_rows = []
for cid, g in agg.groupby("countyid"):
    g = g.sort_values("year")
    idx = g.index.values
    yrs = g["year"].values.astype(int)
    sid = int(g["SID"].iloc[0])
    meta = dict(SID=sid, state=g["state"].iloc[0], countyid=int(cid),
                county_name=g["county_name"].iloc[0])
    for v in C.IO_VARS:
        val = g[v + "_raw"].to_numpy(float)
        stretches = detect_stretches(yrs, val)
        if not stretches:
            continue
        Rfull = Rmed[v]
        Rsid = Rfull.loc[sid] if sid in Rfull.index.get_level_values(0) else None
        cser = pd.Series(val, index=yrs)
        for t1, t2 in stretches:
            length = "double" if t2 > t1 else "single"
            # spike-and-reverts are idiosyncratic (they revert, so they are NOT caliber
            # level-shifts); impute via provincial trend (which preserves real province-
            # wide events automatically).  Only unanchored stretches are held.
            imp = C.impute_stretch(cser, Rsid, t1, t2)
            action, method = ("impute", "") if imp else ("hold", "endpoint")
            for t in range(t1, t2 + 1):
                rec = {**meta, "var": v, "year": t, "stretch": f"{t1}-{t2}", "len": length,
                       "action": action, "orig_value": float(cser.get(t)) if cser.get(t) == cser.get(t) else np.nan,
                       "imputed_value": np.nan, "method": method}
                if imp is not None and t in imp:
                    newv, m = imp[t]
                    pos = idx[yrs == t][0]
                    agg.at[pos, v] = newv
                    agg.at[pos, v + "_imp"] = True
                    rec["imputed_value"], rec["method"] = newv, m
                log_rows.append(rec)

alog = pd.DataFrame(log_rows)
alog.to_csv(C.ANOMALY_LOG, index=False)

# ============================================================================
# 6b. DROP counties fully-NA in ANY of the 5 I-O variables (cannot be used)
# ============================================================================
fully_na = set()
for v in C.IO_VARS:
    cnt = agg.groupby("countyid")[v].apply(lambda s: s.notna().sum() == 0)
    fully_na |= set(cnt[cnt].index)
agg = agg[~agg["countyid"].isin(fully_na)].copy()
print(f"\n6b. Dropped {len(fully_na)} counties fully-NA in >=1 I-O variable "
      f"-> {agg['countyid'].nunique()} counties remain")

# ============================================================================
# 6c. SAVE canonical county roster (id + name + coverage) for the next step
# ============================================================================
roster = (agg.groupby("countyid").agg(
    county_name=("county_name", lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
    SID=("SID", "first"), state=("state", "first"),
    y0=("year", "min"), y1=("year", "max"), n_years=("year", "size")).reset_index())
complete = agg.dropna(subset=C.IO_VARS).groupby("countyid").size().rename("n_complete")
roster = roster.merge(complete, on="countyid", how="left")
roster["n_complete"] = roster["n_complete"].fillna(0).astype(int)
roster = roster.sort_values(["SID", "countyid"])
roster.to_csv(os.path.join(C.CLEAN_DIR, "county_roster.csv"), index=False, encoding="utf-8-sig")
print(f"6c. Saved county roster -> src/clean/county_roster.csv ({len(roster)} counties)")

# ============================================================================
# SAVE cleaned panel
# ============================================================================
front = ["SID", "state", "countyid", "county_name", "year"]
order = front + [c for c in agg.columns if c not in front]
agg = agg[order].sort_values(["SID", "countyid", "year"]).reset_index(drop=True)
agg.to_csv(C.CLEAN_PANEL, index=False)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n6. Spike detection + imputation:")
if len(alog):
    print(f"   flagged county-years: {len(alog)}  (over {agg.countyid.nunique()} counties x {len(C.IO_VARS)} vars)")
    print("   by action:", alog["action"].value_counts().to_dict())
    print("   by class :", alog["cls"].value_counts().to_dict())
    print("   by rule  :", alog["rule"].value_counts().to_dict())
    print("   imputed cells per variable:")
    for v in C.IO_VARS:
        print(f"      {v:18s} flagged={int((alog['var']==v).sum()):4d}  imputed={int(agg[v+'_imp'].sum()):4d}")
else:
    print("   none flagged")
print(f"\nSaved cleaned panel  -> {os.path.relpath(C.CLEAN_PANEL, C.ROOT)}  "
      f"({len(agg)} rows, {agg.countyid.nunique()} counties, {agg.SID.nunique()} provinces)")
print(f"Saved anomaly log    -> {os.path.relpath(C.ANOMALY_LOG, C.ROOT)}")
