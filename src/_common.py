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
AG_LIST    = os.path.join(DATA, "ag_counties_crop15.csv")  # cropland>=15% ag counties
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
# Geography filter: keep agricultural rural counties, drop urban districts and
# prefecture/municipality aggregate rows.
# ----------------------------------------------------------------------------
DISTRICT_SUFFIXES = ("区", "辖", "郊")   # 市辖区 / "XX市辖" aggregates / 郊区 suburbs


def is_urban_district(name):
    """True for urban municipal districts that should be excluded."""
    if not isinstance(name, str):
        return False
    return name.endswith(DISTRICT_SUFFIXES)


def is_aggregate_row(countyid):
    """countyid ending in 00 = prefecture / municipality total, not a real county unit."""
    return int(countyid) % 100 == 0


def apply_geography_filter(df, keep_codes=None):
    """Return (kept_df, dropped_df) after removing urban districts + aggregate rows.

    keep_codes: countyids exempt from the district/aggregate drop (e.g. 撤县设区
    agricultural counties harmonised in county_corrections.CANON_KEEP).
    """
    keep_codes = set(keep_codes or ())
    d = df.copy()
    d["_district"] = d["county_name"].map(is_urban_district)
    d["_aggregate"] = d["countyid"].map(is_aggregate_row)
    drop_mask = (d["_district"] | d["_aggregate"]) & ~d["countyid"].isin(keep_codes)
    kept = d[~drop_mask].drop(columns=["_district", "_aggregate"]).copy()
    dropped = d[drop_mask].copy()
    dropped["drop_reason"] = np.where(dropped["_aggregate"], "aggregate_row",
                                      "urban_district")
    dropped = dropped.drop(columns=["_district", "_aggregate"])
    return kept, dropped


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
