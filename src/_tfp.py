# -*- coding: utf-8 -*-
"""
Shared TFP analysis primitives.

Everything downstream of the frontier estimators (aggregation, winsorizing,
period cuts, the Solow cross-check) used to be copy-pasted into every analysis
script, which is how `aggregate()` ended up with three slightly different
implementations.  It lives here once.

  BETA_*        Cobb-Douglas input elasticities from the converged BC92 SFA
  STAGES        the five reform stages used throughout the paper
  PERIODS5/6    the descriptive period cuts used by the frontier figures
  load_panel    cleaned I-O panel, positivity-filtered, optionally balanced
  solow_ln_tfp  CRS Solow residual, county-level
  county_dln    per-county first difference with the year-gap guard
  winsorize_by_year / tornqvist_aggregate / stage_table
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

import _common as C

# ---------------------------------------------------------------------------
# Input elasticities from the BC92 tnormal SFA (21_sfa_bc92.do), land-normalised
# so they impose CRS; the land elasticity is the 1 - bL - bK - bM residual.
#
# BETA_AG is the headline set: the frontier CONVERGED on the full 1,975-county
# agricultural sample defined by the MULTI-YEAR cropland union (converged=1,
# 28 iterations, N=63,370; sigma_u2=0.2523, sigma_v2=0.1096, gamma=0.697,
# mu=4.335, eta=-0.0069, Wald chi2(38)=35,438).  This supersedes
# both earlier sets and removes the placeholder that 40_cps_prep.py was flagging:
# the weights are now estimated on the SAME sample the analysis runs on.
#
# The older sets are kept for reproducing earlier tables and for the subsample
# robustness check, not because anything should prefer them.
# ---------------------------------------------------------------------------
BETA_AG = {"Laborday_impute": 0.4094, "Land_serv_q": 0.0697,
           "capital_serv_q": 0.0544, "Inter_all_real": 0.4666}
BETA_SUB1000 = {"Laborday_impute": 0.448, "Land_serv_q": 0.081,
                "capital_serv_q": 0.038, "Inter_all_real": 0.433}
BETA_OLD2600 = {"Laborday_impute": 0.470, "Land_serv_q": 0.015,
                "capital_serv_q": 0.070, "Inter_all_real": 0.446}
BETA_FULL = BETA_AG            # back-compatible alias; prefer BETA_AG

# The five reform stages (Lin 1992; Huang & Rozelle; Gong 2018).  Data start in
# 1981, so stage 1 is truncated from its usual 1978 opening.
STAGES = [("1 Reform take-off", 1981, 1984),
          ("2 Stagnation",      1985, 1988),
          ("3 Recovery",        1989, 1996),
          ("4 Adjustment",      1997, 2003),
          ("5 Subsidy era",     2004, 2016)]

# Descriptive period cuts for the frontier figures: 5-period Kalirajan et al.
# (via Gong 2018 JDE), and a 6-period variant splitting the last at 2004
# (Zhang & Bruemmer 2011; national phase-out of the agricultural tax).
PERIODS5 = [(1981, 1984), (1985, 1989), (1990, 1993), (1994, 1997), (1998, 2016)]
PERIODS6 = [(1981, 1984), (1985, 1989), (1990, 1993), (1994, 1997),
            (1998, 2003), (2004, 2016)]


def stage_of(year, stages=STAGES):
    for name, a, b in stages:
        if a <= year <= b:
            return name
    return "NA"


def period_of(year, periods=PERIODS5):
    for a, b in periods:
        if a <= year <= b:
            return f"{a}-{b}"
    return "NA"


# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------
def load_panel(extra_cols=(), balanced=False, counties=None, path=None, ag_only=True):
    """Cleaned I-O panel restricted to rows where all five I-O variables are
    present and strictly positive (the analysis sample everywhere).

    ag_only    : keep only cropland>=15% agricultural counties (`ag_county`==1).
                 The cleaned panel deliberately holds ALL ~2,570 resolved rural
                 county units so it can be reused, and the agricultural
                 restriction is made here.  Default True because every analysis
                 in this project is on the agricultural sample; pass False to
                 work with the full county set.
    extra_cols : additional columns that must also be non-missing.  Requiring
                 them BEFORE the balance test matters: a county with a hole in,
                 say, nominal GVP must not be counted as balanced.
    balanced   : keep only counties observed in every year of the panel.
    counties   : optional iterable of countyids to restrict to.
    """
    d = pd.read_csv(path or C.CLEAN_PANEL)
    extra_cols = list(extra_cols)
    missing = [c for c in extra_cols if c not in d.columns]
    if missing:
        raise KeyError(f"{os.path.basename(path or C.CLEAN_PANEL)} lacks {missing}")
    if ag_only:
        if "ag_county" not in d.columns:
            raise KeyError("cleaned panel lacks `ag_county` — re-run 02 and 03")
        d = d[d["ag_county"] == 1]
    d = d.dropna(subset=C.IO_VARS + extra_cols)
    d = d[(d[C.IO_VARS] > 0).all(axis=1)].copy()
    if counties is not None:
        d = d[d.countyid.isin(set(counties))].copy()
    if balanced:
        nyr = d.year.nunique()
        cnt = d.groupby("countyid").year.nunique()
        d = d[d.countyid.isin(cnt[cnt == nyr].index)].copy()
    return d.sort_values(["countyid", "year"]).reset_index(drop=True)


def gvp_weights(panel):
    """{(countyid, year): real_gvp} lookup for the Tornqvist output shares."""
    p = panel.dropna(subset=["real_gvp"])
    p = p[p.real_gvp > 0]
    return {(int(r.countyid), int(r.year)): float(r.real_gvp) for r in p.itertuples()}


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
def solow_ln_tfp(df, beta=BETA_FULL):
    """County-level CRS Solow residual, ln TFP = ln y - SUM_k beta_k ln x_k."""
    return np.log(df["real_gvp"]) - sum(beta[k] * np.log(df[k]) for k in beta)


def county_dln(df, col, by="countyid", year="year"):
    """First difference of `col` within county, keeping only consecutive years.

    The gap guard matters: the panel is unbalanced, so a naive .diff() would
    silently treat a 1987->1993 jump as one year of growth.
    """
    s = df.sort_values([by, year]).copy()
    s["dln"] = s.groupby(by)[col].diff()
    s["_gap"] = s.groupby(by)[year].diff()
    out = s[(s["_gap"] == 1) & s["dln"].notna()]
    return out[[by, year, "dln"]].reset_index(drop=True)


def winsorize_by_year(df, col="dln", out_col="dln_w", lo=0.01, hi=0.99):
    """Trim the growth cross-section WITHIN each year (SFA.do's winsor2 1/99,
    applied by year so a heavy-tailed early year is not trimmed against a calm
    late one)."""
    d = df.copy()
    d[out_col] = d.groupby("year")[col].transform(
        lambda s: s.clip(s.quantile(lo), s.quantile(hi)))
    return d


def tornqvist_aggregate(dln_df, gvp, col="dln"):
    """County growth -> national annual series, two weightings.

    simple   : equal weight per county
    weighted : output-share Tornqvist, w_it = 0.5*(s_{i,t-1} + s_{i,t}),
               s = county share of national real GVP.  Averaging adjacent-year
               shares is what stops any single year dominating.

    `dln_df` columns: countyid, year (= t, growth t-1 -> t), <col>.
    Returns year, n, dln_simple, dln_weighted, dln_sd, cum_*, growth_* (%/yr).
    """
    rows = []
    for t, g in dln_df.groupby("year"):
        c = g.countyid.to_numpy()
        d = g[col].to_numpy(float)
        gt = np.array([gvp.get((int(x), int(t)), np.nan) for x in c])
        gtm = np.array([gvp.get((int(x), int(t) - 1), np.nan) for x in c])
        ok = np.isfinite(gt) & np.isfinite(gtm) & np.isfinite(d)
        d, gt, gtm = d[ok], gt[ok], gtm[ok]
        if len(d) == 0:
            continue
        w = 0.5 * (gt / gt.sum() + gtm / gtm.sum())
        w = w / w.sum()
        rows.append((int(t), len(d), float(d.mean()), float((w * d).sum()),
                     float(d.std(ddof=1)) if len(d) > 1 else np.nan))
    out = (pd.DataFrame(rows, columns=["year", "n", "dln_simple", "dln_weighted", "dln_sd"])
             .sort_values("year").reset_index(drop=True))
    for k in ("simple", "weighted"):
        out[f"cum_{k}"] = out[f"dln_{k}"].cumsum()
        out[f"growth_{k}"] = 100 * (np.exp(out[f"dln_{k}"]) - 1)
    return out


def cagr_pct(annual, kind="weighted"):
    """Mean %/yr over the whole span, from the cumulative log index."""
    if annual.empty:
        return np.nan
    span = annual.year.iloc[-1] - annual.year.iloc[0]
    return 100 * annual[f"cum_{kind}"].iloc[-1] / span if span else np.nan


def stage_table(annual, method, kind="weighted", stages=STAGES):
    """Per-stage mean growth and annual volatility of one national series."""
    a = annual.copy()
    a["stage"] = a.year.map(lambda y: stage_of(y, stages))
    rows = []
    for name, y0, y1 in stages:
        s = a[a.stage == name]
        if s.empty:
            continue
        rows.append(dict(method=method, stage=name,
                         yrs=f"{s.year.min()}-{s.year.max()}",
                         mean_grw_pct=round(100 * (np.exp(s[f"dln_{kind}"].mean()) - 1), 2),
                         annual_sd=round(s[f"growth_{kind}"].std(), 2)))
    return pd.DataFrame(rows)


def summarise(annual, tag, kinds=("simple", "weighted")):
    """One printed line per weighting: cumulative, %/yr, annual mean and sd."""
    y0, y1 = annual.year.iloc[0], annual.year.iloc[-1]
    for k in kinds:
        g = annual[f"growth_{k}"]
        print(f"  {tag:>10} {k:>8}: cum {y0}-{y1} {100*annual[f'cum_{k}'].iloc[-1]:+6.1f}% log"
              f"  |  {cagr_pct(annual, k):+.2f}%/yr  |  annual mean {g.mean():+.2f}%  sd {g.std():.2f}")
