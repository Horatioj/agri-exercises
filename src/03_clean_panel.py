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
# Detector thresholds. Env-overridable so the sensitivity of the cleaning
# footprint to the cut-off can be measured (see 05_imputation_audit.py) rather than
# argued about: a detector that fires on a tenth of the panel is describing normal
# variation, not anomalies.
#
# Default 4.0.  Measured footprint on this panel (share of county-years with at
# least one variable modified / share of observed cells):
#     3x  11.88% / 3.72%   -- over the 10% budget
#     4x   9.15% / 2.74%   <- default
#     5x   7.73% / 2.21%
# 3x also looked least like a unit-error detector: only 19.4% of boundary-run
# splice factors sat within 0.10 dex of a power of ten, against 27.3% at 5x, i.e.
# loosening the cut mainly pulls in genuine variation.  4x is the setting that
# meets the budget while staying closest to the original detector.
_T = lambda k, d: np.log(float(os.environ.get(k, d)))
TAU_UP   = _T("TAU_UP", 4.0)      # upward spike (more likely an error)
TAU_DOWN = _T("TAU_DOWN", 4.0)    # downward spike (could be a real bad year)
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

# Hainan (SID 46, founded 1988) pre-1988 rows are KEPT.  They used to be dropped
# because the level break at provincehood distorted the provincial trend, but the
# boundary-run splice (4c) is the general treatment for exactly that pattern, and
# the reference is now a chained median of year-on-year changes, which a single
# province's level shift no longer moves.  Deleting real observations to protect a
# reference that is now robust to them is a cost with no remaining benefit.
n_hn = int(((agg["SID"] == 46) & (agg["year"] < 1988)).sum())
print(f"1c. Hainan: kept {n_hn} pre-1988 rows (handled by the 4c splice, not a drop)")

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

# ----------------------------------------------------------------------------
# Snapshot the TRUE original NOW, before any cleaning rule fires.  It used to be
# taken after 4b/4c/4d, which meant <var>_raw was not the original at all: every
# value those steps set to NA or rescaled was already gone from it, so the review
# SVGs drew a "raw" line that had been partly cleaned and the audit could not see
# those edits.  Taken here, <var>_raw is what arrived from the source, so
# 05_imputation_audit.py counts every modification and 04's grey line shows what
# was actually replaced.
# ----------------------------------------------------------------------------
for v in C.IO_VARS:
    agg[v + "_raw"] = agg[v].astype(float)
    agg[v + "_imp"] = False

# ============================================================================
# 4b. ERROR -> NA: zeros, and values <1 in these large-magnitude units.  Both are
#     unusable rather than merely mis-levelled -- ln(0) is undefined and a sub-1
#     value cannot be spliced onto anything -- so they stay a deletion.
#
#     The old "low-then-jump" rule (anything >1000x below the county median -> NA)
#     is GONE.  It described the same pattern the 4c splice now repairs -- an
#     opening block on a different basis -- but deleted it instead of correcting
#     the level, and because it ran first it pre-empted 4c entirely (it removed
#     102 of Hainan's 136 pre-1987 Inter_all_real observations before the splice
#     could see them).  Whatever tiny values survive 4c are isolated single years,
#     which is exactly the spike-and-revert case step 6 handles, so one mechanism
#     now covers what two did.
# ============================================================================
for v in C.IO_VARS:
    n_zero = int((agg[v] == 0).sum())
    agg.loc[agg[v] == 0, v] = np.nan
    # implausibly small for these large-magnitude vars (e.g. 久治县 land all <1)
    sub1 = (agg[v] > 0) & (agg[v] < 1)
    n_sub1 = int(sub1.sum())
    agg.loc[sub1, v] = np.nan
    print(f"4b. {v:18s}: {n_zero:4d} zeros, {n_sub1:4d} <1 -> NA")

# ============================================================================
# 4c. LEADING run -> rescale/NA: trim a RUN of opening years that sits far off the
#     local stable median -- e.g. 654002 伊宁市 land 603 -> 4227 -> 70000, 玛多县, 临淄区.
#
#     LEADING ONLY.  The rule exists to repair 农本数据电子化 -- the digitisation of
#     the early cost-of-production/农业部 ledgers -- which is an EARLY-YEAR problem by
#     construction: a unit or caliber convention carried over from the paper source
#     before the series settles onto its modern basis.  Nothing about that story
#     applies at the end of the panel, and the panel said so: the trailing scan was
#     producing 373 rescales concentrated in 2011-2016 (85 in 2016 alone), which are
#     late-panel reporting changes, boundary reorganisations or genuine collapses --
#     none of them a digitisation slip, and none of them legitimately "spliced" onto
#     an earlier core.  The trailing branch is therefore gone; end-of-panel anomalies
#     fall to the spike step, the caliber step and the manual list.
# ============================================================================
TAU_END = _T("TAU_END", 4.0)      # run sits this far off the post-jump core
# The run must lie entirely BEFORE this year.  A calendar window, not a run length.
#
# This replaces the old MAXRUN cap (a count of years) and, with it, the question of
# how many correctly-based years may sit INSIDE the block.  Under a length cap the
# run had to be a contiguous prefix, so an embedded good year -- 130705 宣化区 land
# reads 500 / 192 / 100003 / 369 / 31718, where 1983 is already on the modern basis
# and only 1981, 1982, 1984 are not -- either truncated the block or forced a
# tolerance knob to admit it.  A calendar window makes the question disappear: the
# off-basis years inside the window are simply SELECTED, contiguous or not, and
# on-basis years inside it are left alone because they are not off-core.
#
# 1990 is where the 农本数据电子化 story stops being plausible.  The early ledgers
# were digitised from paper cost-of-production/农业部 sources carrying their own
# unit conventions; by the 1990s the county returns are natively electronic and on
# a settled basis, so a large break after that is a real event (boundary change,
# reporting reform, collapse) and must not be spliced away.
BOUNDARY_RUN_YEAR = int(os.environ.get("BOUNDARY_RUN_YEAR", 1990))
JUMP5   = _T("JUMP5", 4.0)        # size of the boundary discontinuity
def trim_runs(g, v):
    """Find LEADING years, all before BOUNDARY_RUN_YEAR, that sit >TAU_END off the
    post-break core median, where the break itself is a BIG (>JUMP5) year-on-year
    jump.  Anchored on the DISCONTINUITY, not the median, so genuine gradual growth
    -- even 100x over the panel, like capital -- is not caught.
    e.g. 伊宁市 land 603/712/672/4227 then ~70000; 临淄区 1981-83.

    LEADING AND PRE-1990 ONLY.  The pattern this rule exists for is 农本数据电子化 --
    the early years were digitised from paper cost-of-production/农业部 ledgers kept
    on a different basis (unit, caliber, or a re-based splice), so the break is at
    the OPENING of the panel by construction.  A trailing scan was tried and
    removed: it has no such story behind it, the late 1990s-2010s county returns are
    natively electronic and internally consistent, and any late-panel break it
    "found" would be an ordinary structural change (撤县设区, land reallocation, a
    genuine collapse) that must NOT be spliced away.

    The run is a SET of off-core years inside the window, not a contiguous prefix.
    A correctly-based year can sit inside the off-basis stretch -- 130705 宣化区 land
    reads 500 / 192 / 100003 / 369 / 31718, where 1983 is already on the modern
    basis and only 1981, 1982, 1984 are not.  Selecting off-core years rather than
    requiring a block means such a year is simply not selected: it needs no
    tolerance knob, and it is left untouched instead of being dragged along by a
    factor that does not apply to it.

    Returns [(index_array, factor)] where `factor` splices the run onto the core:
        factor = median(core segment) / median(run)
    so run * factor is level-continuous with the core.  The CALLER decides whether
    to NA the run or rescale it (see BOUNDARY_RUN_MODE).
    """
    s = g.sort_values("year")[["year", v]]
    s = s[s[v] > 0].dropna()
    out = []
    n = len(s)
    if n >= 8:
        vals = s[v].to_numpy(); idx = s.index.to_numpy()
        yrs = s["year"].to_numpy(); lv = np.log(vals)
        # j indexes the FIRST core year, so the run is the off-core subset of [0, j).
        # Every run year must precede BOUNDARY_RUN_YEAR, which caps j at the number
        # of in-window observations.  Take the largest qualifying j: a single noisy
        # year later in the window -- a spike step 6 would have smoothed anyway --
        # must not be allowed to end the search early (620826 land jumps at j=1 with
        # dlog 9.6, then again at j=2 and j=3 on ordinary noise; stopping at the last
        # jump left its 1981 value sitting 9,919x below the county median).
        jmax = min(int((yrs < BOUNDARY_RUN_YEAR).sum()), n - 3)
        jstar, off_star = 0, None
        for j in range(1, jmax + 1):
            if abs(lv[j] - lv[j-1]) <= JUMP5:
                continue
            core = np.log(np.median(vals[j:j+5]))
            off = [k for k in range(j) if abs(lv[k] - core) > TAU_END]
            # the first year must be off-core, or this is not an opening block on a
            # different basis at all
            if off and off[0] == 0:
                jstar, off_star = j, np.array(off)
        if jstar:
            out.append((idx[off_star], float(np.median(vals[jstar:jstar+5]) /
                                             np.median(vals[off_star]))))
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
BOUNDARY_RUN_MODE = os.environ.get("BOUNDARY_RUN_MODE", "decimal")
_run_log = []
for v in C.IO_VARS:
    # A units slip happens in the SOURCE ledger, which is nominal.  real_gvp is the
    # only variable this pipeline deflates (Land_serv_q / capital_serv_q /
    # Inter_all_real already arrive at 2005 prices), so for GVP the run is detected
    # and the factor measured on the NOMINAL series and then applied to both.
    # Measured contamination if we did not: the province deflator moves a median of
    # 1.049 (p95 1.080) across a break year, so up to ~8% of a real_gvp factor would
    # be price change rather than units -- inside the 0.10 dex gate, but there is no
    # reason to carry it.
    det = "GVP_allagr_impute" if v == "real_gvp" else v
    runs = [(ix, f) for _, g in agg.groupby("countyid") for ix, f in trim_runs(g, det)]
    n_yr = sum(len(ix) for ix, _ in runs)
    if BOUNDARY_RUN_MODE == "na":
        for ix, _ in runs:
            agg.loc[ix, v] = np.nan
        act = "-> NA"
    elif BOUNDARY_RUN_MODE in ("rescale", "decimal"):
        # "decimal": a boundary run is RESCALED if and only if the factor is a power
        # of ten, and DELETED otherwise.  One rule, no escape hatch.
        #
        # The claim being made when we rescale is specific -- the source ledger was
        # kept in a different unit -- and that claim has a signature: the factor is
        # exactly 10^k.  So the factor is SNAPPED to that power of ten rather than
        # applied as the raw median ratio (which is why the logged factors used to
        # read 0.089838 instead of 0.1: two noisy medians, not a unit).  Anything
        # else is a level break we can detect but cannot interpret, and splicing it
        # by an empirical ratio would fabricate the level rather than repair it.
        #
        # A magnitude escape hatch (BIG_BREAK: "a break this large cannot be growth,
        # splice it whatever the factor looks like") used to admit those cases and
        # was removed.  Being unable to call a break growth is not the same as
        # knowing what it is.  It also decided neighbouring counties differently for
        # no reason: 130702 桥东区 and 130703 桥西区 are adjacent districts of 张家口市
        # with the SAME 1981-84 land basis error, but their empirical ratios came out
        # 479.8 and 67.6, so one cleared the threshold and was spliced by a
        # meaningless factor while the other was deleted.  Now both are deleted.
        DEX = float(os.environ.get("BOUNDARY_RUN_DEX", "0.10"))
        n_sp = n_na = 0
        for ix, f in runs:
            ok = np.isfinite(f) and f > 0
            f_emp = f                       # empirical ratio, kept for the log
            snapped = False
            if ok and BOUNDARY_RUN_MODE == "decimal":
                ok = abs(np.log10(f) - round(np.log10(f))) <= DEX
                if ok:
                    f = 10.0 ** round(np.log10(f))
                    snapped = True
            if not ok:
                agg.loc[ix, v] = np.nan
                n_na += len(ix)
                continue
            for i in ix:
                before = agg.at[i, v]
                after = before * f
                agg.at[i, v] = after
                if v == "real_gvp":              # keep the nominal series consistent
                    agg.at[i, "GVP_allagr_impute"] = agg.at[i, "GVP_allagr_impute"] * f
                _run_log.append(dict(
                    countyid=agg.at[i, "countyid"], SID=agg.at[i, "SID"], var=v,
                    year=agg.at[i, "year"], action="rescale",
                    value_before=before,          # the digitised source value
                    value_after=after,            # what the panel now carries
                    factor=f,                     # APPLIED factor (snapped to 10^k if decimal)
                    factor_empirical=f_emp,       # raw median(core)/median(run) ratio
                    snapped_to_power_of_ten=snapped,
                    log10_factor=np.log10(f),
                    decimal_dex=abs(np.log10(f_emp) - round(np.log10(f_emp)))))
            n_sp += len(ix)
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

# <var>_raw was snapshotted before 4b (see above); do NOT overwrite it here or the
# NA/rescale steps become invisible again.  Only ensure the _imp flags exist.
for v in C.IO_VARS:
    if v + "_imp" not in agg.columns:
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

# Half-width of the window used to decide WHICH side of a discontinuity is the
# anomaly (see _is_the_anomaly).  Wide enough that a couple of bad years cannot
# carry the median, narrow enough not to span a genuine trend.
LOCAL_W = int(os.environ.get("LOCAL_W", 5))


def _is_the_anomaly(L, i0, i1, loc):
    """Spike-and-revert is SYMMETRIC: a good plateau sitting between two bad years
    looks exactly like a bad spike sitting between two good ones.  The local test
    cannot tell them apart, so it needs an arbiter -- the county's own level.

    Returns True only if the candidate stretch sits FURTHER from the local level
    than its own anchors do.  If the anchors are the distant ones, they are the
    anomaly and the stretch is the normal state, so the flag is refused.

    532524 建水县 real_gvp is the case that forced this.  Raw 2000-2005 reads
    95,300 / 19,534 / 98,287 / 113,000 / 15,172 / 84,083: the bad years are 2001
    and 2004 alone.  The scan correctly flagged both as single down-spikes, then
    reached 2002 and found a textbook 2-year up-plateau -- +1.62 in, -2.01 out,
    the two years within 0.14 of each other -- because BOTH its neighbours were
    the outliers.  It imputed a perfectly good 98,287 / 113,000 down to
    17,510 / 15,644, i.e. it repaired the good years to match the broken ones.
    Against the 1997-2008 local median of 94,255 the arbiter separates them
    cleanly: the 2001 and 2004 stretches sit 1.57 and 1.83 in logs from the local
    level against anchors at 0.03, while the 2002-03 stretch sits 0.11 away
    against anchors at 1.70.

    LIMIT, stated rather than hidden: this is a majority rule.  Where the bad
    years OUTNUMBER the good ones inside the window, the median follows them and
    the arbiter inverts.  It cannot do better -- at that point nothing in the
    series identifies which level is correct, and the case belongs on the manual
    list.
    """
    lo = max(0, i0 - LOCAL_W)
    hi = min(len(L), i1 + LOCAL_W + 1)
    w = L[lo:hi]
    w = w[np.isfinite(w)]
    if len(w) < 5 or not np.isfinite(loc):
        return True                     # too little context to overrule the local test
    d_stretch = abs(np.nanmedian(L[i0:i1+1]) - loc)
    anchors = [L[k] for k in (i0 - 1, i1 + 1) if 0 <= k < len(L) and np.isfinite(L[k])]
    if not anchors:
        return True
    d_anchor = abs(np.median(anchors) - loc)
    return d_stretch > d_anchor


def detect_stretches(yrs, val):
    """Spike-and-revert stretches: 1-year single, 2-year double, 3-year triple plateau."""
    n = len(val)
    L = np.where(val > 0, np.log(val), np.nan)
    def cons(a, b):   # years a..b consecutive and all finite
        return all(yrs[k+1]-yrs[k] == 1 for k in range(a, b)) and np.isfinite(L[a:b+1]).all()

    def local(i0, i1):
        lo = max(0, i0 - LOCAL_W); hi = min(n, i1 + LOCAL_W + 1)
        w = L[lo:hi]; w = w[np.isfinite(w)]
        return np.median(w) if len(w) else np.nan

    out = []; i = 1
    while i < n - 1:
        if cons(i-1, i+1) and _revert(L[i]-L[i-1], L[i+1]-L[i]) \
                and _is_the_anomaly(L, i, i, local(i, i)):
            out.append((int(yrs[i]), int(yrs[i]))); i += 1; continue                 # single
        if i < n-2 and cons(i-1, i+2) and abs(L[i]-L[i+1]) < PLATEAU_TOL \
                and _revert(L[i]-L[i-1], L[i+2]-L[i+1]) \
                and _is_the_anomaly(L, i, i+1, local(i, i+1)):
            out.append((int(yrs[i]), int(yrs[i+1]))); i += 2; continue               # double
        if i < n-3 and cons(i-1, i+3) and abs(L[i]-L[i+1]) < PLATEAU_TOL \
                and abs(L[i+1]-L[i+2]) < PLATEAU_TOL and _revert(L[i]-L[i-1], L[i+3]-L[i+2]) \
                and _is_the_anomaly(L, i, i+2, local(i, i+2)):
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
        # Detect and impute on the CURRENT cleaned series, not on <var>_raw.
        #
        # This step used to read _raw, which is snapshotted before 4b and therefore
        # still holds the original digitised values -- so the spike test and, worse,
        # the imputation ANCHORS were blind to every repair steps 3-4d had already
        # made, and the result was written straight over them.  130705 宣化区 land
        # is the case that exposed it: 4c correctly rescaled 1981/82/84 by x100, but
        # this step still saw the raw 500 / 192 / 100003 / 369, called 1983 a spike
        # (up from 192, down to 369), and imputed it against the UN-rescaled 1982
        # anchor of 192 -- dragging a perfectly good 100,003 down to 235 and pushing
        # 1984 up to 61,565.  Both "spikes" were artefacts of comparing repaired
        # years against unrepaired neighbours.  On the cleaned series the 1983
        # excursion is +1.65 / -1.00 in logs, and -1.00 does not clear TAU_DOWN, so
        # it is correctly left alone.  _raw is still kept, untouched, as the audit
        # trail of what the source said.
        val = g[v].to_numpy(float)
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
# 8. BACKSTOP: any cell still sitting >1000x below its own county median -> NA.
#
#     Nothing upstream can reach these.  4c only scans the panel boundaries, and
#     what survives here is an INTERIOR lone year (typically 1982-83, one case in
#     2015) that the spike step also missed because its neighbours are themselves
#     part of the disturbed opening block, so the revert test never fires.  A
#     value three orders of magnitude below the county's own level is not a level
#     shift to be spliced -- there is no run to preserve the shape of -- so it is
#     deleted rather than corrected.  This is the deliberate floor of the cleaner:
#     if the earlier rules improve, this count goes to zero on its own.
# ============================================================================
LOW_FLOOR = float(os.environ.get("LOW_FLOOR", 1000.0))
alog_rows_floor, n_floor = [], 0
for v in C.IO_VARS:
    x = pd.to_numeric(agg[v], errors="coerce")
    med = x.where(x > 0).groupby(agg["countyid"]).transform("median")
    bad = (x > 0) & (med > 0) & (x < med / LOW_FLOOR)
    n_floor += int(bad.sum())
    if bad.any():
        for i in agg.index[bad]:
            alog_rows_floor.append(dict(countyid=agg.at[i, "countyid"], SID=agg.at[i, "SID"],
                                        var=v, year=agg.at[i, "year"], action="floor_NA",
                                        value_before=float(x[i]), ratio=float(med[i] / x[i])))
    agg.loc[bad, v] = np.nan
print(f"\n8. Extreme-low backstop (< county median / {LOW_FLOOR:g}): {n_floor} cells -> NA")
if alog_rows_floor:
    pd.DataFrame(alog_rows_floor).to_csv(os.path.join(C.DQ_DIR, "extreme_low_floor.csv"),
                                         index=False, encoding="utf-8-sig")
    print("   -> dq/extreme_low_floor.csv")

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
