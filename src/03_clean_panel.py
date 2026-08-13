# -*- coding: utf-8 -*-
"""
STEP 3 — Clean the county-level annual agricultural input-output panel.

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
       non-flagged neighbours. (gap is 3.0 now, but 5.0 is more conservative; see C.impute_stretch).
  7. MANUAL MULTI-YEAR IMPUTATION — the abnormal STRETCHES identified by eye from
     the county-review figures (data/manual_impute_list.csv).  Step 6 only catches
     one-year spikes that revert; a variable that sits at the wrong level for five
     years never trips it, so those runs are listed by hand and imputed here by
     riding the provincial MEDIAN trend between the two good anchors
     (C.impute_stretch — the same method step 6 uses, so every imputed value in
     the panel comes from one implementation).

This runs as ONE pass: the manual step operates on the in-memory frame before the
single save, rather than re-reading and rewriting the panel it just wrote.

Outputs:
  src/clean/county_panel_clean.csv  — cleaned panel; <var> cleaned, <var>_raw
                                       original, <var>_imp flag.
  src/clean/anomaly_log.csv          — one row per (county, var, flagged year).
  src/clean/manual_impute_log.csv    — one row per manually imputed county-year.
  src/dq/caliber_candidate_years.csv — province synchronized-break candidates.
  src/dq/dropped_units.csv           — units removed by the geography filter.
Run:  python src/03_clean_panel.py            # both steps
      python src/03_clean_panel.py --no-manual
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import _common as C
import _county_corrections as CC

ARGS = argparse.ArgumentParser()
ARGS.add_argument("--no-manual", action="store_true",
                  help="stop after step 6 (auto spike imputation only)")
ARGS = ARGS.parse_args()

MANUAL_LIST = os.path.join(C.DATA, "manual_impute_list.csv")
MANUAL_LOG = os.path.join(C.CLEAN_DIR, "manual_impute_log.csv")

C.ensure_dirs()
# Asymmetric spike-and-revert thresholds: an UPWARD spike (value jumps up then back)
# is almost always a data error -> stricter 3x; a DOWNWARD spike (drop then recover)
# can be a real bad-harvest/weather year
TAU_UP   = np.log(3.0)
TAU_DOWN = np.log(3.0)
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
# County codes/names, administrative merges and the typo fixes were already applied
# upstream in 01/02 (data/roster_resolution.csv).  The panel here is ALL resolved
# rural county-level units; `ag_county` (cropland>=15%) is a carried FLAG, not a
# filter, so the cleaned panel stays reusable for non-agricultural work.

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
JUMP5 = np.log(3.0) # set 3.0 now instead of 5.0
def trim_runs(g, v):
    """Find a SHORT leading/trailing run (<=5 yrs) separated from the stable core by a
    BIG (>JUMP5) year-on-year jump AND sitting >TAU_END off the post-jump core median.
    Anchored on the DISCONTINUITY, not the median, so genuine gradual growth -- even
    100x over the panel, like capital -- is not caught.
    e.g. 伊宁市 land 603/712/672/4227 then ~70000; 临淄区 1981-83.

    Returns [(index_array, factor)] where `factor` splices the run onto the core:
        factor = median(core segment) / median(run)
    so run * factor is level-continuous with the core.  The CALLER decides whether
    to NA the run or rescale it (see BOUNDARY_RUN_MODE).
    """
    s = g.sort_values("year")[v]
    s = s.where(s > 0).dropna()
    out = []
    n = len(s)
    if n >= 8:
        vals = s.to_numpy(); idx = s.index.to_numpy(); lv = np.log(vals)
        jstar = 0                                        # leading: LAST big jump in yrs 1..5
        for j in range(1, min(6, n - 3)):
            if abs(lv[j] - lv[j-1]) > JUMP5:
                jstar = j
        if jstar and all(abs(lv[k] - np.log(np.median(vals[jstar:jstar+5]))) > TAU_END
                         for k in range(jstar)):
            out.append((idx[:jstar], float(np.median(vals[jstar:jstar+5]) /
                                            np.median(vals[:jstar]))))
        jstar = 0                                        # trailing: LAST big jump near end
        for j in range(1, min(6, n - 3)):
            if abs(lv[n-j] - lv[n-j-1]) > JUMP5:
                jstar = j
        if jstar and all(abs(lv[n-1-k] - np.log(np.median(vals[n-jstar-5:n-jstar]))) > TAU_END
                         for k in range(jstar)):
            out.append((idx[n-jstar:], float(np.median(vals[n-jstar-5:n-jstar]) /
                                             np.median(vals[n-jstar:]))))
    return out


# BOUNDARY_RUN_MODE: "rescale" (default) splices the run onto the core by a single
# multiplicative factor; "na" drops it (the previous behaviour); "keep" leaves it.
#
# Why rescale is the better default: these runs are a LEVEL break, not noise.  The
# pattern -- a few years at one level, one big jump, then a stable core -- is the
# signature of a units/caliber change in the digitised 农本/农业部 source (a switch
# of reporting unit, or a re-based series spliced at the wrong level).  The
# year-to-year MOVEMENT inside the run is real information; only its LEVEL is wrong.
# Setting it to NA throws away both and shortens the panel exactly at the ends,
# where the frontier and the growth endpoints are most sensitive.  Rescaling
# removes the break and keeps the shape.  Every factor is written to the anomaly
# log, and <var>_raw preserves the original, so the operation stays reversible.
BOUNDARY_RUN_MODE = os.environ.get("BOUNDARY_RUN_MODE", "rescale")
_run_log = []
for v in C.IO_VARS:
    runs = [(ix, f) for _, g in agg.groupby("countyid") for ix, f in trim_runs(g, v)]
    n_yr = sum(len(ix) for ix, _ in runs)
    if BOUNDARY_RUN_MODE == "na":
        for ix, _ in runs:
            agg.loc[ix, v] = np.nan
        act = "-> NA"
    elif BOUNDARY_RUN_MODE in ("rescale", "decimal"):
        # "decimal" is the conservative reading of the unit-error story: only splice
        # when the factor is close to a power of ten (a plausible units/decimal slip
        # in the digitised source), and NA the rest.  Measured on this panel only
        # 19.4% of factors sit within 0.10 dex of 10^k (37.3% within 0.20), so
        # "rescale" applies the unit interpretation well beyond where the evidence
        # supports it, while "na" discards ~8,300 county-years.  "decimal" splits them.
        DEX = float(os.environ.get("BOUNDARY_RUN_DEX", "0.10"))
        n_sp = n_na = 0
        for ix, f in runs:
            ok = np.isfinite(f) and f > 0
            if ok and BOUNDARY_RUN_MODE == "decimal":
                ok = abs(np.log10(f) - round(np.log10(f))) <= DEX
            if not ok:
                agg.loc[ix, v] = np.nan
                n_na += len(ix)
                continue
            agg.loc[ix, v] = agg.loc[ix, v] * f
            n_sp += len(ix)
            for i in ix:
                _run_log.append(dict(countyid=agg.at[i, "countyid"], SID=agg.at[i, "SID"],
                                     var=v, year=agg.at[i, "year"], action="rescale",
                                     factor=f))
        act = f"-> {n_sp} rescaled / {n_na} NA"
    else:
        act = "-> kept (no action)"
    fac = [f for _, f in runs]
    extra = (f"  factor median {np.median(fac):.2f}, range {min(fac):.2f}-{max(fac):.2f}"
             if fac else "")
    print(f"4c. {v:18s}: {n_yr:4d} boundary-run years in {len(runs):3d} runs {act}{extra}")
if _run_log:
    pd.DataFrame(_run_log).to_csv(os.path.join(C.DQ_DIR, "boundary_run_rescale.csv"),
                                  index=False)
    print(f"    boundary-run factors -> dq/boundary_run_rescale.csv")

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
# 5. CALIBER YEARS — province synchronized breaks; require mannual check, not yet implemented
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

# PROVINCIAL REFERENCE for imputation.  "chain" (default) cumulates the median
# year-on-year log CHANGE over counties observed in BOTH years; "level" is the old
# per-(SID,year) median of ln.  The level version moves whenever the SET of
# reporting counties changes and inherits province-wide breaks directly into the
# path the imputation rides -- measured here, the level path jumps by up to 5.5 ln
# units (~250x) between adjacent years, and that survives on a balanced core, so it
# is a real break in the reference rather than composition alone.
PROV_REF = os.environ.get("PROV_REF", "chain")
Rmed = {v: C.provincial_ref_ln(agg, v, method=PROV_REF) for v in C.IO_VARS}
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
# 7. MANUAL MULTI-YEAR IMPUTATION (provincial-trend-anchored, MEDIAN reference)
#
# data/manual_impute_list.csv holds the abnormal STRETCHES identified by eye from
# the county-review figures (countyid, var, year_start, year_end[, note]).  Every
# year in [year_start, year_end] is imputed by riding the provincial MEDIAN trend,
# pinned to the last-good anchor (year_start-1) and the first-good anchor
# (year_end+1):
#     ln C(t) = R(t) + [ d0 + (d3 - d0) * (t - t0)/(t3 - t0) ]
# R(t) = median ln(value) over the province's OTHER counties (shape reference);
# d0, d3 = the county's offset from R at the two anchors.  Where R(t) is missing
# the year falls back to plain log-linear interpolation.  Caliber years are never
# touched implicitly — the list controls exactly which years move.
# ============================================================================
mlog_rows, m_applied, m_skipped = [], 0, []
if ARGS.no_manual:
    print("\n7. Manual imputation: SKIPPED (--no-manual)")
elif not os.path.exists(MANUAL_LIST):
    print(f"\n7. Manual imputation: no list at {os.path.relpath(MANUAL_LIST, C.ROOT)}, skipped")
else:
    spec = pd.read_csv(MANUAL_LIST, encoding="utf-8-sig")
    spec = spec[spec["countyid"].astype(str).str.strip().str.isdigit()].copy()
    spec["countyid"] = spec["countyid"].astype(int)
    # case-insensitive var match (the list may use inter_all_real / Inter_all_real)
    _lc2var = {v.lower(): v for v in C.IO_VARS}
    spec["var"] = spec["var"].astype(str).str.strip().str.lower().map(_lc2var)
    bad_var = int(spec["var"].isna().sum())
    spec = spec[spec["var"].notna()]
    for _c in ("year_start", "year_end"):
        spec[_c] = spec[_c].astype(int)
    if bad_var:
        print(f"   (dropped {bad_var} rows with an unrecognised var name)")

    Rmed = {v: C.provincial_median_ln(agg, v) for v in spec["var"].unique()}
    for _, r in spec.iterrows():
        cid, var, ys, ye = int(r.countyid), r["var"], int(r.year_start), int(r.year_end)
        if cid not in agg["countyid"].values:
            m_skipped.append((cid, var, "county not in panel")); continue
        sid = int(agg.loc[agg.countyid == cid, "SID"].iloc[0])
        cser = agg[agg.countyid == cid].set_index("year")[var]
        Rfull = Rmed[var]
        Rsid = Rfull.loc[sid] if sid in Rfull.index.get_level_values(0) else None
        imp = C.impute_stretch(cser, Rsid, ys, ye)      # SHARED provincial-trend method
        if imp is None:
            m_skipped.append((cid, var, f"missing/invalid anchor {ys-1} or {ye+1}")); continue
        for t, (new, method) in imp.items():
            orig = cser.get(t)
            sel = (agg.countyid == cid) & (agg.year == t)
            agg.loc[sel, var] = new
            agg.loc[sel, var + "_imp"] = True
            mlog_rows.append(dict(countyid=cid, SID=sid, var=var, year=t,
                                  orig_value=None if (orig is None or orig != orig) else float(orig),
                                  imputed_value=new, method=method,
                                  anchor_lo=ys - 1, anchor_hi=ye + 1))
            m_applied += 1

    pd.DataFrame(mlog_rows).to_csv(MANUAL_LOG, index=False, encoding="utf-8-sig")
    print(f"\n7. Manual imputation: {len(spec)} stretches, {m_applied} county-years imputed")
    if mlog_rows:
        print("   by method:", pd.DataFrame(mlog_rows)["method"].value_counts().to_dict())
    if m_skipped:
        print(f"   SKIPPED {len(m_skipped)} (fix the list and re-run):")
        for s in m_skipped[:20]:
            print("     ", s)

# ============================================================================
# SAVE cleaned panel
# ============================================================================
front = ["SID", "state", "countyid", "county_name", "year", "ag_county"]
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
    for _c in ("cls", "rule"):                    # optional columns
        if _c in alog.columns:
            print(f"   by {_c:6s}:", alog[_c].value_counts().to_dict())
    print("   imputed cells per variable:")
    for v in C.IO_VARS:
        print(f"      {v:18s} flagged={int((alog['var']==v).sum()):4d}  imputed={int(agg[v+'_imp'].sum()):4d}")
else:
    print("   none flagged")
print(f"\nSaved cleaned panel  -> {os.path.relpath(C.CLEAN_PANEL, C.ROOT)}  "
      f"({len(agg)} rows, {agg.countyid.nunique()} counties, {agg.SID.nunique()} provinces)")
print(f"Saved anomaly log    -> {os.path.relpath(C.ANOMALY_LOG, C.ROOT)}")
if mlog_rows:
    print(f"Saved manual log     -> {os.path.relpath(MANUAL_LOG, C.ROOT)}")
