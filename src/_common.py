# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA       = os.path.join(ROOT, "data")
SRC        = os.path.join(ROOT, "src")
CLEAN_DIR  = os.path.join(SRC, "clean")
DQ_DIR     = os.path.join(SRC, "dq")
FIG_DIR    = os.path.join(SRC, "figures")

AGG_DTA    = os.path.join(DATA, "cty_prod_account_agg_forAXN.dta")
GVP_DTA    = os.path.join(DATA, "agr_GVP.dta")        # GVP source
LABOR_DTA  = os.path.join(DATA, "agr_labor.dta")      # labour source (agr_labor1)
PRICE_DTA  = os.path.join(DATA, "priceindex_province.dta")
# Agricultural-county definition: cropland >= 15% in ANY year 1986-2015 (the
# UNION), built by 00_cropland_share.py from the annual CACD 30 m rasters.
# The older ag_counties_crop15.csv used 2010 ALONE, which is the LOWEST-count
# year of the whole series -- it dropped ~170 counties that farmed for much of
# the panel and had urbanised by 2010, conditioning the sample on the very
# transition the paper studies.  Set AG_LIST back to that file to reproduce
# the old sample.
# STRICT: cropland >= 15% in EVERY raster year (1986-2015), not merely in one.
# The UNION rule admitted 2,149 counties, the strict rule 1,855 -- it removes 294,
# of which 120 are urban districts.  The union was chosen to avoid conditioning
# the sample on a single cross-section, but it has no guard against a county that
# crossed the line briefly: 29 of the counties it admitted cleared 15% in only
# 1-5 of the 30 years and were then carried for all 36 panel years.  Requiring
# every year removes those and most of the peri-urban districts with them.
# NOTE it is STRICTLY stronger than "last_above >= 2015" (1,954 counties): 99
# counties are still above at the end but dipped below in between.
AG_LIST    = os.path.join(DATA, "ag_county", "ag_counties_strict_aglist.csv")
AG_LIST_UNION = os.path.join(DATA, "ag_county", "ag_counties_union_aglist.csv")
AG_LIST_2010 = os.path.join(DATA, "ag_county", "ag_counties_crop15.csv")  # superseded
BASE_DTA   = os.path.join(DATA, "base_panel.dta")     # assembled by 00_build_base_panel.do
IO_RAW     = os.path.join(DATA, "io_raw_corrected.dta")  # corrected raw I-O (00e), input to 01

CLEAN_PANEL = os.path.join(CLEAN_DIR, "county_panel_clean.csv")
ANOMALY_LOG = os.path.join(CLEAN_DIR, "anomaly_log.csv")
CALIBER_CSV = os.path.join(DQ_DIR, "caliber_candidate_years.csv")
DROPPED_CSV = os.path.join(DQ_DIR, "dropped_units.csv")

# ----------------------------------------------------------------------------
# The five input-output variables that are cleaned and reviewed
#   output : real_gvp           (GVP deflated by province PPI, 2005 prices, 10k RMB)
#   inputs : Laborday_impute    (labour, man-days)
#            Land_serv_q         (land service quantity, 2005 prices)
#            capital_serv_q      (capital service quantity, 2005 prices)
#            Inter_all_real      (real intermediate inputs, 2005 prices)
# ----------------------------------------------------------------------------
IO_VARS = ["real_gvp", "Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
VLAB = {
    "real_gvp":        "Real GVP (output)",
    "Laborday_impute": "Labour (man-days)",
    "Land_serv_q":     "Land service",
    "capital_serv_q":  "Capital service",
    "Inter_all_real":  "Intermediate (real)",
}

# Province deflator column used to deflate nominal GVP -> real_gvp.
# PPI_CCD_2005 is the province analogue of the county PPP_2005 (Fisher+CCD, Hebei 2005=1).
PPI_COL = "PPI_CCD_2005"

# ----------------------------------------------------------------------------
# County-type LABELS.  These are DESCRIPTIVE, not a filter.
#
# There is exactly ONE agricultural-county definition in this project: the
# cropland test in 02_apply_resolution.py (cropland >= 15% in any year 1986-2015),
# written onto the panel as the `ag_county` flag.  A name-based
# `apply_geography_filter` used to live here as a second, parallel rule that
# dropped 区/辖/郊 names and countyid%100==0 aggregate rows; it was never called
# by any script, so it was deleted rather than left looking authoritative.
#
# The label below is kept because the composition it measures is worth reporting:
# 616 of the 1,975 agricultural counties (31.2%) have 区 names.  They are NOT a
# filter failure -- they pass the cropland test on their own merits, being
# peri-urban districts that really do farm -- but they are atypical enough that
# 59_frontier_audit.py quantifies how far they move each frontier (~3% at the
# aggregate, ~19% on the K/L partial view).  Use it to SPLIT a sample for a
# robustness check, never to define one.
# ----------------------------------------------------------------------------
DISTRICT_SUFFIXES = ("区", "辖", "郊")   # 市辖区 / "XX市辖" aggregates / 郊区 suburbs


def is_urban_district(name):
    """True if the county NAME is that of an urban district.  Descriptive label
    only -- see the note above; this does not decide sample membership."""
    if not isinstance(name, str):
        return False
    return name.endswith(DISTRICT_SUFFIXES)


def is_aggregate_row(countyid):
    """countyid ending in 00 = prefecture / municipality total, not a real county unit."""
    return int(countyid) % 100 == 0


# ----------------------------------------------------------------------------
# Matplotlib CJK font (so Chinese county names render on Windows)
# ----------------------------------------------------------------------------
def set_cjk_font(plt):
    from matplotlib import font_manager
    for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
               r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs():
    for d in (CLEAN_DIR, DQ_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# Shared imputation: provincial-trend-anchored (MEDIAN reference). Used by BOTH
# the auto spike-and-revert step (01) and the manual multi-year step (01b), so
# every imputed value follows the same method.
# ----------------------------------------------------------------------------
def provincial_median_ln(df, var):
    """Per-(SID, year) MEDIAN of ln(value) — a robust provincial shape reference."""
    v = pd.to_numeric(df[var], errors="coerce")
    lnv = np.log(v.where(v > 0))
    return lnv.groupby([df["SID"], df["year"]]).median()


def provincial_ref_ln(df, var, method="chain"):
    """Provincial ln reference path per (SID, year).

    method="level"  per-(SID,year) MEDIAN of ln(value).  Simple, but it moves when
                    the SET of counties reporting changes, and it inherits any
                    province-wide level break straight into the reference the
                    imputation rides.
    method="chain"  median of year-on-year log CHANGES over the counties observed
                    in BOTH years, cumulated into a path.  Each link uses a fixed
                    set of counties, so entry/exit cannot move the reference, and
                    a level break in any single county shifts only its own link.
                    Anchored to the province's first-year median level so the
                    result is on the same scale as method="level".
    """
    v = pd.to_numeric(df[var], errors="coerce")
    lnv = np.log(v.where(v > 0))
    if method == "level":
        return lnv.groupby([df["SID"], df["year"]]).median()

    t = pd.DataFrame({"SID": df["SID"].to_numpy(), "cid": df["countyid"].to_numpy(),
                      "year": df["year"].to_numpy(), "ln": lnv.to_numpy()})
    t = t.dropna(subset=["ln"]).sort_values(["cid", "year"])
    t["d"] = t.groupby("cid")["ln"].diff()
    t["gap"] = t.groupby("cid")["year"].diff()
    step = t[(t["gap"] == 1) & t["d"].notna()].groupby(["SID", "year"])["d"].median()
    lvl = lnv.groupby([df["SID"], df["year"]]).median()

    keys, vals = [], []
    for sid, s in step.groupby(level=0):
        s = s.droplevel(0).sort_index()
        lv = lvl.loc[sid].dropna().sort_index()
        if lv.empty:
            continue
        y0 = int(s.index.min()) - 1
        acc = float(lv.get(y0, lv.iloc[0]))
        keys.append((sid, y0)); vals.append(acc)
        for y, dd in s.items():
            acc += float(dd)
            keys.append((sid, int(y))); vals.append(acc)
    if not keys:
        return lvl
    return pd.Series(vals, index=pd.MultiIndex.from_tuples(keys, names=["SID", "year"])
                     ).sort_index()


def impute_stretch(cser, Rsid, t1, t2):
    """Impute years [t1, t2] of one county-variable by riding the provincial median
    trend, pinned to the last-good (t1-1) and first-good (t2+1) anchors:
        ln C(t) = R(t) + [d0 + (d3-d0)*(t-t0)/(t3-t0)]
    cser : county's value Series indexed by year.  Rsid : provincial median ln by
    year (or None -> log-linear fallback).  Returns {year:(value, method)} or None
    if an anchor year is missing/invalid."""
    t0, t3 = t1 - 1, t2 + 1
    if t0 not in cser.index or t3 not in cser.index:
        return None
    c0, c3 = cser.get(t0), cser.get(t3)
    if not (c0 > 0) or not (c3 > 0):
        return None
    lnC0, lnC3 = np.log(c0), np.log(c3)
    R0 = Rsid.get(t0, np.nan) if Rsid is not None else np.nan
    R3 = Rsid.get(t3, np.nan) if Rsid is not None else np.nan
    use_prov = np.isfinite(R0) and np.isfinite(R3)
    d0, d3 = (lnC0 - R0, lnC3 - R3) if use_prov else (0.0, 0.0)
    out = {}
    for t in range(t1, t2 + 1):
        w = (t - t0) / (t3 - t0)
        Rt = Rsid.get(t, np.nan) if Rsid is not None else np.nan
        if use_prov and np.isfinite(Rt):
            out[t] = (float(np.exp(Rt + d0 + (d3 - d0) * w)), "prov_median")
        else:
            out[t] = (float(np.exp(lnC0 + (lnC3 - lnC0) * w)), "linear")
    return out
