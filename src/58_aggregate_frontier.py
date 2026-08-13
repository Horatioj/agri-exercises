# -*- coding: utf-8 -*-
"""
Figure 9 / Figure 10 with the four inputs AGGREGATED into a single index.

WHY AN AGGREGATE INDEX
  52_ and 53_ put ONE input on the x axis at a time, so each figure sees a
  two-dimensional slice of a four-input technology.  Aggregating x = f(L, Land,
  K, M) into a scalar puts the WHOLE input bundle on one axis, which is the
  textbook one-input production function and is directly comparable to the
  aggregate TFP indices computed elsewhere in the project.

WHAT THE FOUR INPUTS ACTUALLY ARE (codebook, 农业生产数据集变量代码本)
      Laborday_impute  labour, WORKED DAYS
      Land_serv_q      land service flow.  Hedonic (quality-adjusted) nominal
                       farmland RENT in yuan, divided by Fixed_PI_2005.
      capital_serv_q   capital service flow over non-residential buildings,
                       transport equipment and farm machinery.  PIM stock with
                       geometric depreciation, aggregated by user cost
                       (r + delta - pi), in yuan, divided by Fixed_PI_2005.
      Inter_all_real   intermediates, yuan at 2005 prices (nominal twin
                       Inter_all_nom is also supplied).
  So three of the four are VALUES at 2005 prices and the nominal counterpart of
  land and capital is recoverable as <quantity> * Fixed_PI_2005.  That is what
  makes cost-share weighting worth attempting at all.

WHY AGGREGATION IS POSSIBLE ANYWAY (the quantity side)
  A Tornqvist quantity index uses only the GROWTH RATES of each input,
  d ln X = SUM_j wbar_j d ln X_j, and the level form below divides every input
  by its own sample geometric mean.  Either way an arbitrary per-input scale
  factor CANCELS EXACTLY.  A Tornqvist index needs each input's unit to be the
  same in every county and every year -- which the codebook guarantees -- and
  needs nothing at all about the unit's size.  So the index below is valid
  regardless of how the level question in the next block is resolved.

WHY COST SHARES ARE STILL ONLY HALF AVAILABLE -- A LEVEL PROBLEM, NOT A PRICE ONE
  Rebuilding all four nominal costs in yuan gives medians (n = 89,434
  county-years with a wage match):

      share of nominal GVP    labour 12.4%     intermediates 19.6%
                              land    0.022%   capital        0.099%

  Land and capital land three orders of magnitude below the other two.  The
  price data are NOT the missing piece: Fixed_PI_2005 is supplied, and applying
  it changes nothing about the order of magnitude.  Two checks locate it.

  LAND -- consistent with a factor of about 1000.  Dividing total rebuilt
  nominal rent by total CACD cropland area over the 1,975 agricultural counties
  gives an implied rent of 0.08 yuan/mu in 1990 and 0.41 in 2015, ~1000x below
  the farmland rental market.  At x1000 the same series reads

      1990   83     1995  201     2000  222     2005  173     2010  358
      2015  413     yuan/mu, nominal

  every one of which is inside the observed range for its year.  One constant
  reconciles the level and the whole time profile, so land looks like a unit
  issue (yuan vs thousand yuan) and nothing worse.

  CAPITAL -- NO constant works.  On the cleaned agricultural sample real capital
  service per county grows 68.8x between 1981 and 2015, against 10.9x for real
  intermediates and 4.6x for machinery horsepower.  The median county ratio K/M
  therefore DRIFTS -- 0.0028 in 1990 to 0.0320 in 2015, a factor of 11.6 -- so a
  rescaling that fixes the early years breaks the late ones.  A PIM started from
  a near-zero initial stock produces exactly this pattern.  This is the SECOND
  independent complaint against capital_serv_q: 50_io_surface finds its fitted
  output elasticity negative for most observations in most years under both
  Cobb-Douglas and translog.

  BOTH SERIES STEP AT 2005, the base year of Fixed_PI_2005.  Between 2000 and
  2005 K/M jumps 0.0037 -> 0.0083 while Land/M halves 0.0015 -> 0.0006, neither
  with any counterpart in the intermediate or machinery series.  Whatever the
  cause, it is in the construction of these two variables and not in the panel.

  In the index capital carries beta_K = 0.054, so its 68.8x growth contributes
  0.054 * ln(68.8) = 0.23 to ln x -- about a quarter of aggregate input growth
  over the panel from this one series alone.  That is not fatal but it is not
  small, and it belongs in any reading of the time ordering in Figure 9.

  What the two SOUND shares deliver is a check on the weights actually used.
  Between labour and intermediates alone:

      observed cost shares   L 0.387 : M 0.613     (wage x days, nominal inter.)
      BC92 SFA elasticities  L 0.467 : M 0.533     (estimated on this sample)

  Same order and same direction.  wage_CropIndustry is a CROP-INDUSTRY wage
  while GVP_allagr covers farming, forestry, livestock and fishery, so the
  observed labour share is a LOWER bound -- which is the direction of the gap.

  Default weights are therefore the ELASTICITIES from the converged BC92 SFA
  (_tfp.BETA_AG): under constant returns with factors paid their marginal
  product the output elasticity EQUALS the cost share.  Sum(beta) = 1, so the
  index is homogeneous of degree one and x/L, x/Land are well defined.
  --weights offers two alternatives, both computed from the data at run time:
      sfa        BC92 elasticities                                    (default)
      hybrid     L:M from OBSERVED cost shares, land and K keep their
                 SFA elasticities -- uses the real prices where they exist
      costshare  observed shares of total input cost, all four.  Land and
                 capital collapse to ~0.4% combined, so this is a BOUND showing
                 what the price data literally say, not a recommended index.
  A fourth scheme is deliberately NOT offered.  The scale constants for land and
  capital could be calibrated by forcing total input cost to equal nominal GVP
  each year and solving for them, which is the residual-imputation convention in
  the USDA-ERS accounts.  It is not used here because the residual it would
  split is 1 - 0.124 - 0.196 = 0.68, far too large, and it is too large precisely
  because the crop-industry wage understates the all-agriculture labour bill.
  Calibrating on a residual that is known to be biased would move that bias
  wholesale into the land and capital weights and hide it there.

  Each input is first divided by its own sample GEOMETRIC MEAN, so the index is
  unit-free and equals 1 at the average bundle.  Without that step the geometric
  weighting mixes incompatible units into an uninterpretable scale.

FIGURE 9 (three normalisations, as requested)
  (A) Y      against x          levels
  (B) Y/L    against x/L        per labour day
  (C) Y/Land against x/Land     per unit of land service
  Note that in (B) and (C) the denominator is also INSIDE x.  That is deliberate
  and standard (Kumar & Russell 2002 divide both axes by labour), and it is
  legitimate because Sum(beta)=1 makes x homogeneous of degree one; but the
  reader should know the axes are not independent.

FIGURE 10 (isoquants -- one dimension must be held out)
  A scalar input index has NO isoquant: with one input there is nothing to trade
  off.  So the aggregate isoquant holds ONE input out of the index and plots it
  against the aggregate of the remaining three:
  (A) Labour  against the aggregate of Land, K, M
  (B) Land    against the aggregate of L, K, M
  Boundary = lower-left envelope of the output-normalised cloud (Pareto-minimal
  chain then lower convex hull), the same construction as Figure B of
  52_frontier_isoquant_figs.py.

Input : src/clean/county_panel_clean.csv (agricultural sample)
Output: src/figures/fig9agg_frontier.png, fig10agg_isoquant.png
        src/clean/descriptive/fig9agg_*.csv, fig10agg_*.csv,
        captions_fig9agg_10agg.md
Run:  python src/58_aggregate_frontier.py
      python src/58_aggregate_frontier.py --weights hybrid
      python src/58_aggregate_frontier.py --weights costshare
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C
import _tfp as T

OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)
CAPTIONS = {}

DEFAULT_YEARS = [1986, 1995, 2000, 2005, 2010, 2015]
INK, INK2 = "#1a1a19", "#5c5b55"
XCOLS = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
SHORT = {"Laborday_impute": "L", "Land_serv_q": "Land",
         "capital_serv_q": "K", "Inter_all_real": "M"}


def aggregate_index(d, cols, beta):
    """CRS geometric (Tornqvist with fixed weights) index over `cols`.

    Weights are renormalised to sum to 1 over the SUBSET, so holding an input
    out of the index still yields a homogeneous-of-degree-one aggregate.
    Each input is divided by its sample geometric mean first, so the index is
    unit-free and equals 1 at the average bundle -- without that, geometric
    weighting of variables on different unit bases has no interpretable scale.
    """
    w = {c: beta[c] for c in cols}
    s = sum(w.values())
    w = {c: v / s for c, v in w.items()}
    lx = 0.0
    for c in cols:
        gm = np.exp(np.log(d[c]).mean())          # sample geometric mean
        lx = lx + w[c] * (np.log(d[c]) - np.log(gm))
    return np.exp(lx), w


def observed_cost_shares(d):
    """Median share of TOTAL INPUT COST for each of the four inputs, in yuan.

    Costs are rebuilt from the codebook definitions:
        labour        Laborday_impute * wage_avg        (wage_CropIndustry.dta)
        land          Land_serv_q    * Fixed_PI_2005    (priceindex_province)
        capital       capital_serv_q * Fixed_PI_2005
        intermediates Inter_all_nom                     (already nominal)

    Shares are normalised by TOTAL INPUT COST, not by GVP: a Tornqvist weight
    only has to sum to one across inputs, so this needs no zero-profit closure.
    Land and capital come out near zero -- see the module docstring; the return
    value is reported honestly and the caller decides what to do with it.

    The trailing block prints the two UNIT-FREE ratios that diagnose why.  Both
    land and capital are 2005-price yuan by the codebook, exactly as
    Inter_all_real is, so K/M and Land/M would be O(0.1-1) if the three shared a
    scale.  Land sits ~1000x low but at a CONSTANT factor across the panel;
    capital sits low AND drifts by a factor of 7, which no rescaling repairs.
    """
    pi = pd.read_stata(os.path.join(C.DATA, "priceindex_province.dta"),
                       convert_categoricals=False)[["SID", "year", "Fixed_PI_2005"]]
    wg = pd.read_stata(os.path.join(C.DATA, "wage_CropIndustry.dta"),
                       convert_categoricals=False)[["countyid", "year", "wage_avg"]]
    s = (d.merge(pi.drop_duplicates(["SID", "year"]), on=["SID", "year"], how="left")
          .merge(wg.drop_duplicates(["countyid", "year"]),
                 on=["countyid", "year"], how="left"))
    cost = {"Laborday_impute": s.Laborday_impute * s.wage_avg,
            "Land_serv_q": s.Land_serv_q * s.Fixed_PI_2005,
            "capital_serv_q": s.capital_serv_q * s.Fixed_PI_2005,
            "Inter_all_real": s.Inter_all_nom}                 # nominal twin
    cost = pd.DataFrame(cost)
    ok = cost.notna().all(axis=1) & (cost > 0).all(axis=1)
    cost, sub = cost[ok], s[ok]
    tot = cost.sum(axis=1)
    sh = {c: float((cost[c] / tot).median()) for c in cost.columns}
    gvp = sub.GVP_allagr_impute * 1e4                          # wan yuan -> yuan
    print(f"observed cost shares from {ok.sum():,d} of {len(s):,d} county-years "
          f"({sub.countyid.nunique():,d} counties, {sub.year.min()}-{sub.year.max()})")
    print(f"  {'input':6s} {'share of input cost':>20s} {'share of nominal GVP':>21s}")
    for c in XCOLS:
        print(f"  {SHORT[c]:6s} {sh[c]:20.4f} {float((cost[c]/gvp).median()):21.4f}")

    # Unit-free scale diagnostic: all three of land, capital and intermediates
    # are 2005-price yuan, so these ratios would be O(0.1-1) on a common scale.
    # NOTE: computed on the FULL agricultural sample `s`, not on the wage-matched
    # subset `sub` -- these ratios need no wage, and the figures use the full sample.
    r = pd.DataFrame({"year": s.year,
                      "K/M": s.capital_serv_q / s.Inter_all_real,
                      "Land/M": s.Land_serv_q / s.Inter_all_real})
    med = r.groupby("year").median()
    yrs = [y for y in (1981, 1990, 2000, 2005, 2010, 2015) if y in med.index]
    print("  scale check, ratios to real intermediates (both 2005-price yuan):")
    print("    " + "".join(f"{y:>10d}" for y in yrs))
    for k in ("K/M", "Land/M"):
        print(f"    {k:<6s}" + "".join(f"{med.loc[y, k]:10.5f}" for y in yrs))
    if len(yrs) > 1:
        d_k = med.loc[yrs[-1], "K/M"] / med.loc[yrs[0], "K/M"]
        d_l = med.loc[yrs[-1], "Land/M"] / med.loc[yrs[0], "Land/M"]
        print(f"    drift {yrs[0]}->{yrs[-1]}: K/M x{d_k:.1f}, Land/M x{d_l:.1f}"
              "  -- a constant rescaling can only fix a ratio that does NOT drift")
    return sh


def index_weights(d, scheme):
    """Weights for the aggregate input index under the chosen scheme."""
    beta = dict(T.BETA_AG)
    if scheme == "sfa":
        print("weights: BC92 SFA output elasticities (cost shares under CRS)")
        return beta
    sh = observed_cost_shares(d)
    if scheme == "costshare":
        print("weights: OBSERVED shares of total input cost -- a BOUND, not the "
              "recommended index (the land and capital LEVELS do not reconcile; "
              "see the scale check above)")
        return sh
    # hybrid: keep the SFA weights for the two inputs that have no price level,
    # and let the two that DO have prices split the remainder in their observed
    # ratio.  This uses the real price information exactly where it exists.
    keep = beta["Land_serv_q"] + beta["capital_serv_q"]
    r = sh["Laborday_impute"] / (sh["Laborday_impute"] + sh["Inter_all_real"])
    out = {"Land_serv_q": beta["Land_serv_q"], "capital_serv_q": beta["capital_serv_q"],
           "Laborday_impute": (1 - keep) * r, "Inter_all_real": (1 - keep) * (1 - r)}
    print(f"weights: HYBRID -- observed L:M = {r:.3f}:{1-r:.3f} "
          f"(SFA says {beta['Laborday_impute']/(beta['Laborday_impute']+beta['Inter_all_real']):.3f}"
          f":{beta['Inter_all_real']/(beta['Laborday_impute']+beta['Inter_all_real']):.3f}), "
          f"land and K at their SFA elasticities")
    return out


def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def quantile_frontier(x, y, alpha=0.95, nbins=20, minobs=30, concave=False,
                      tag=""):
    """Order-alpha frontier: the alpha-quantile of y within x bins, made
    non-decreasing in x.  Same estimator as Figure 9 in 53_, for the same
    reason: the raw full envelope on ~1,850 observations is set by 2-5 extreme
    counties (verified), so it reports an outlier rather than the technology.

    CONCAVITY IS NOT IMPOSED (`concave=False`).  It used to be, via an upper
    concave hull over the bin points, and on two of the three panels the data
    reject it: in the per-labour and per-land normalisations the binned quantiles
    are CONVEX in logs, so the concave hull discards every interior point and
    collapses to the chord between the two endpoints -- a straight line floating
    above the cloud it is supposed to summarise.  The function still REPORTS how
    many points a hull would discard, so the reader learns that concavity fails
    instead of seeing it silently enforced.  This follows the same rule as
    57_weather_frontier.py: the shape is the finding, not something to correct.
    Pass `concave=True` to restore the hull.

    EVERYTHING HAPPENS IN LOGS, for two reasons that are not cosmetic.

    (1) EQUAL-WIDTH BINS IN log x, NOT EQUAL-COUNT BINS IN x.  With equal-count
        bins the top bin spans most of a decade -- x is heavily right-skewed --
        and the estimator then pairs that bin's alpha-quantile of y, which is
        set by the observations at the bin's RIGHT edge, with the bin's MEDIAN
        x, which sits at its left edge.  Because y rises steeply with x that
        misplaces the point far up and to the left of any real observation.  On
        the per-labour panel it put the top vertex at y = 1.44 where the local
        95th percentile is 0.37, i.e. the frontier floated clear above its own
        cloud.  Equal-width log bins bound the within-bin x range instead, and
        bins that fall below `minobs` are dropped rather than widened.

    (2) THE HULL IS TAKEN IN (log x, log y).  Taking it in levels and drawing
        the result on log axes bulges every segment upward -- a straight line in
        levels is convex on a log-log plot -- which was the second half of the
        same visual failure.  Concavity in logs is also the more defensible
        restriction here: it says the output elasticity of the aggregate input
        does not increase, which is the diminishing-returns statement the figure
        is meant to display.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    if len(lx) < 3 * minobs:
        return np.array([]), np.array([])
    # trim the extreme 0.5% of x so a lone county cannot set the bin grid
    lo, hi = np.quantile(lx, [0.005, 0.995])
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.searchsorted(edges, lx, side="right") - 1, 0, nbins - 1)
    bx, by = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < minobs:
            continue
        bx.append(np.median(lx[m])); by.append(np.quantile(ly[m], alpha))
    if len(bx) < 3:
        return np.exp(np.array(bx)), np.exp(np.array(by))
    bx, by = np.array(bx), np.array(by)

    H = []                                       # upper concave hull, in logs
    for p in zip(bx, by):
        while len(H) >= 2 and _cross(H[-2], H[-1], p) >= 0:
            H.pop()
        H.append(p)
    if tag:
        print(f"    {tag}: {len(bx)} bins, concave hull would keep {len(H)}"
              + ("  [CONCAVITY REJECTED -- hull is the chord]"
                 if len(H) <= 2 else ""))
    if concave:
        hx = np.array([h[0] for h in H]); hy = np.array([h[1] for h in H])
    else:
        hx, hy = bx, by
    hy = np.maximum.accumulate(hy)              # free disposal: cannot fall
    return np.exp(hx), np.exp(hy)


def dea_frontier(x, y, densify=40, tag="", rts="nirs"):
    """TEXTBOOK DEA frontier: the envelope of the raw observations.

    With one input and one output the DEA frontier IS the upper concave hull of
    the observed points, closed under free disposal -- every county is compared
    with the convex combination of peers that produces the most output at its
    input level, which is exactly what solving the LP for each county returns.
    No binning, no quantile, nothing smoothed away.  This is the object
    `quantile_frontier` deliberately replaces, drawn here so the two can be
    compared directly.

    NON-INCREASING RETURNS (rts="nirs", the default) matches the DEA-Malmquist
    technology used in 20_dea_vs_fe_tfp.py, so the two parts of the project
    describe the same frontier.  The NIRS technology is
        T = {(x, y): y <= SUM_i lambda_i y_i,  x >= SUM_i lambda_i x_i,
                     SUM_i lambda_i <= 1}
    and `SUM lambda <= 1` -- scaling a peer DOWN is allowed, scaling it UP is not
    -- is imposed simply by adding the ORIGIN to the point set before taking the
    hull.  The effect is visible and substantive: the frontier's first segment
    becomes the ray from the origin through the highest-productivity county, so
    no part of it can sit above that county's output-input ratio.  VRS would let
    the boundary rise above that ray at low x, asserting increasing returns that
    the NIRS model rules out.  Pass rts="vrs" for the unrestricted hull.

    The hull is taken in LEVELS, because convexity of the technology set is a
    level-space assumption -- that is the DEA model, and taking it in logs would
    be a different (and weaker) restriction.  Each segment is then DENSIFIED
    before it is returned: matplotlib joins vertices with a straight line in
    DISPLAY space, so on log axes a two-point segment would be drawn as a curve
    that the DEA model never asserts.  The densified points trace the actual
    straight line in levels.

    Returns (x, y) of the drawn curve plus the number of COUNTIES that define it
    (the origin is not a county and is not counted).  That second number is the
    whole argument for the order-alpha estimator, since on this panel a handful
    of observations out of ~1,850 set the entire envelope.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    o = np.argsort(x); x, y = x[o], y[o]
    pts = list(zip(x, y))
    if rts == "nirs":
        pts = [(0.0, 0.0)] + pts             # SUM lambda <= 1
    H = []
    for p in pts:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) >= 0:
            H.pop()
        H.append(p)
    # free disposal in x: output on the frontier can never fall as input rises
    vx = np.array([h[0] for h in H]); vy = np.maximum.accumulate([h[1] for h in H])
    ncty = len(vx) - 1 if rts == "nirs" else len(vx)
    if rts == "nirs" and len(vx) > 1:
        # The origin cannot be drawn on a log axis.  Keep the RAY it defines by
        # walking the first segment back to the smallest observed input instead.
        s = vy[1] / vx[1]                    # max output-input ratio in this year
        vx, vy = vx.copy(), vy.copy()
        vx[0], vy[0] = x.min(), s * x.min()
    if tag:
        print(f"    {tag}: DEA-{rts.upper()} envelope on {ncty} of {len(x):,d} "
              f"counties ({100*ncty/len(x):.2f}%)")
    if len(vx) < 2:
        return vx, vy, ncty
    px, py = [], []
    for i in range(len(vx) - 1):                 # straight in LEVELS, densified
        t = np.linspace(0, 1, densify, endpoint=(i == len(vx) - 2))
        px.append(vx[i] + t * (vx[i+1] - vx[i]))
        py.append(vy[i] + t * (vy[i+1] - vy[i]))
    return np.concatenate(px), np.concatenate(py), ncty


def lower_envelope(a, b):
    """Pareto-minimal chain then lower convex hull -- the Figure B construction."""
    o = np.argsort(a); a, b = a[o], b[o]
    chain, best = [], np.inf
    for x, y in zip(a, b):
        if y < best - 1e-15:
            chain.append((x, y)); best = y
    if len(chain) < 3:
        return np.array([c[0] for c in chain]), np.array([c[1] for c in chain])
    H = []
    for p in chain:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) <= 0:
            H.pop()
        H.append(p)
    return np.array([h[0] for h in H]), np.array([h[1] for h in H])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--weights", choices=["sfa", "hybrid", "costshare"],
                    default="sfa")
    ap.add_argument("--frontier", choices=["alpha", "dea", "both"], default="both",
                    help="alpha = order-alpha quantile frontier (used for the "
                         "substantive reading); dea = textbook DEA envelope of the "
                         "raw observations; both = one figure each, same axes")
    ap.add_argument("--rts", choices=["nirs", "vrs"], default="nirs",
                    help="returns-to-scale assumption for the DEA figure. nirs "
                         "matches the DEA-Malmquist technology in 20_dea_vs_fe_tfp.py")
    a = ap.parse_args()

    d = T.load_panel()
    YEARS = [int(y) for y in a.years.split(",")]
    miss = [y for y in YEARS if y not in set(d.year)]
    if miss:
        raise SystemExit(f"years not in the panel: {miss}")
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} agricultural counties")
    beta = index_weights(d, a.weights)
    d["x"], w = aggregate_index(d, XCOLS, beta)
    SFX = "" if a.weights == "sfa" else "_" + a.weights   # keep the default names
    print("aggregate input index weights (renormalised to 1):")
    for c in XCOLS:
        print(f"  {SHORT[c]:5s} {w[c]:.4f}")
    print(f"  x is unit-free and = 1 at the sample geometric-mean bundle "
          f"(median x = {d.x.median():.3f})\n")
    cols = plt.cm.viridis(np.linspace(0, .88, len(YEARS)))

    # ------------------------------------------------------------- FIGURE 9
    PANELS = [("Y vs x  (levels)", None, "Real GVP", "Aggregate input index x"),
              ("Y/L vs x/L  (per labour day)", "Laborday_impute",
               "Real GVP per labour day", "Aggregate input per labour day"),
              ("Y/Land vs x/Land  (per unit land service)", "Land_serv_q",
               "Real GVP per unit land service", "Aggregate input per unit land service")]
    # The SAME data and the SAME axes are drawn twice, once per estimator, so the
    # only difference between the two files is how the boundary is computed.
    KINDS = ["alpha", "dea"] if a.frontier == "both" else [a.frontier]
    for kind in KINDS:
        KLAB = ("order-alpha quantile" if kind == "alpha"
                else f"textbook DEA {a.rts.upper()}")
        print(f"\nFigure 9 -- {KLAB} frontier")
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
        rows, nvert = [], []
        for ax, (ttl, den, ylab, xlab) in zip(axes, PANELS):
            yv = d.real_gvp if den is None else d.real_gvp / d[den]
            xv = d.x if den is None else d.x / d[den]
            for yr, col in zip(YEARS, cols):
                m = d.year == yr
                ax.scatter(xv[m], yv[m], s=8, alpha=.20, color=col, linewidths=0,
                           rasterized=True)
            ftop = 0.0                    # highest point any frontier reaches here
            for yr, col in zip(YEARS, cols):
                m = d.year == yr
                tag = f"[{ttl.split()[0]}] {yr}"
                if kind == "alpha":
                    hx, hy = quantile_frontier(xv[m].values, yv[m].values,
                                               alpha=a.alpha, tag=tag)
                else:
                    hx, hy, nv = dea_frontier(xv[m].values, yv[m].values, tag=tag,
                                              rts=a.rts)
                    nvert.append(nv)
                if len(hx) < 2:
                    continue
                ftop = max(ftop, float(np.nanmax(hy)))
                ax.plot(hx, hy, "-", color=col, lw=2.3,
                        label=f"{yr}  (n={int(m.sum()):,d})")
                rows.append(pd.DataFrame({"panel": ttl, "year": yr,
                                          "x": hx, "f_y": hy}))
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(xv.quantile(.005), xv.quantile(.995))
            # The DEA envelope routinely sits far above the cloud it bounds, so the
            # y axis is opened to contain it.  Clipping the boundary would hide
            # exactly the fact this figure exists to show, and would also make the
            # gap between the two estimators look smaller than it is.
            ax.set_ylim(yv.quantile(.005), max(yv.quantile(.995), 1.06 * ftop))
            ax.set_xlabel(xlab + "  (log)", fontsize=9)
            ax.set_ylabel(ylab + "  (log)", fontsize=9)
            ax.set_title(ttl, fontsize=11, loc="left", fontweight="bold", color=INK)
            ax.grid(alpha=.25, which="both"); ax.legend(fontsize=7.5, title="year")
        stub = f"fig9agg_frontier{SFX}" + ("" if kind == "alpha" else "_dea")
        pd.concat(rows).to_csv(os.path.join(OUT, f"{stub}.csv"), index=False)
        sub = ("order-alpha quantile frontier, alpha = %g" % a.alpha if kind == "alpha"
               else f"textbook DEA {a.rts.upper()} envelope of the raw observations")
        fig.suptitle("Figure 9 (aggregated inputs): output against the aggregate "
                     f"input index\n{sub}", fontsize=13.5, fontweight="bold", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        p = os.path.join(C.FIG_DIR, f"{stub}.png")
        fig.savefig(p, dpi=170, facecolor="white"); plt.close(fig)
        print("saved:", p)
        if kind == "dea":
            CAPTIONS[stub] = (
                f"Figure 9 (aggregated inputs, TEXTBOOK DEA {a.rts.upper()}). Identical "
                f"data, axes and input index to {'fig9agg_frontier' + SFX}; only the "
                "boundary differs. Here it is the DEA envelope of the raw observations, "
                "which with one input and one output is exactly the upper concave hull "
                "of the cloud closed under free disposal -- the same object the DEA LP "
                "returns for every county. Nothing is binned or smoothed. "
                + ("Non-increasing returns are imposed as sum(lambda) <= 1, i.e. by "
                   "adding the origin to the point set, so the first segment is the ray "
                   "through the year's highest output-input ratio and no part of the "
                   "frontier may sit above it; this is the same technology as the "
                   "sequential-NIRS DEA-Malmquist in 20_dea_vs_fe_tfp.py. "
                   if a.rts == "nirs" else
                   "Variable returns are imposed, so the boundary may rise above the "
                   "best observed output-input ratio at low input levels. ")
                + "The hull is "
                "taken in levels, because convexity of the technology set is a "
                "level-space assumption, and each segment is densified before drawing "
                "so the log axes show the true straight line rather than a curve the "
                "model does not assert. Read it against the order-alpha version: the "
                f"envelope here is defined by {np.mean(nvert):.1f} counties per "
                f"year-panel on average out of roughly {int(d.groupby('year').size().median()):,d}, "
                "i.e. well under 1%, so it reports the most extreme observations rather "
                "than the technology, and a single mismeasured county moves a whole "
                "year's frontier. That fragility is why the order-alpha estimator is the "
                "one used for the substantive reading. Log scales; points are county-years.")

    # ------------------------------------------------------------ FIGURE 10
    # A scalar index has no isoquant, so hold ONE input out and plot it against
    # the aggregate of the rest, output-normalised.
    HOLD = [("Laborday_impute", "Labour days"), ("Land_serv_q", "Land service")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    irows = []
    for ax, (hold, hlab) in zip(axes, HOLD):
        rest = [c for c in XCOLS if c != hold]
        xr, wr = aggregate_index(d, rest, beta)
        an = (d[hold] / d.real_gvp).values
        bn = (xr / d.real_gvp).values
        for yr, col in zip(YEARS, cols):
            m = (d.year == yr).values
            ax.scatter(an[m], bn[m], s=8, alpha=.20, color=col, linewidths=0,
                       rasterized=True)
        for yr, col in zip(YEARS, cols):
            m = (d.year == yr).values
            ok = np.isfinite(an) & np.isfinite(bn) & (an > 0) & (bn > 0) & m
            hx, hy = lower_envelope(an[ok], bn[ok])
            if len(hx) < 2:
                print(f"  [{SHORT[hold]}] {yr}: envelope <2 vertices, skipped")
                continue
            print(f"  [hold {SHORT[hold]}] {yr}: envelope from {len(hx)} vertices "
                  f"(n={int(ok.sum()):,d})")
            ax.plot(hx, hy, "-", color=col, lw=2.3, label=str(yr))
            irows.append(pd.DataFrame({"held_out": SHORT[hold], "year": yr,
                                       "held_per_y": hx, "rest_per_y": hy}))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(np.nanquantile(an, .01), np.nanquantile(an, .99))
        ax.set_ylim(np.nanquantile(bn, .005), np.nanquantile(bn, .995))
        ax.set_xlabel(f"{hlab} per unit of real GVP  (log)", fontsize=9)
        ax.set_ylabel("Aggregate of the other three inputs per unit of real GVP  (log)",
                      fontsize=9)
        ax.set_title(f"Hold out {SHORT[hold]}: {hlab} vs the rest", fontsize=11,
                     loc="left", fontweight="bold", color=INK)
        ax.grid(alpha=.25, which="both"); ax.legend(fontsize=7.5, title="year")
    pd.concat(irows).to_csv(os.path.join(OUT, f"fig10agg_isoquant{SFX}.csv"), index=False)
    fig.suptitle("Figure 10 (aggregated inputs): unit isoquants, one input against "
                 "the aggregate of the rest", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(C.FIG_DIR, f"fig10agg_isoquant{SFX}.png")
    fig.savefig(p, dpi=170, facecolor="white"); plt.close(fig)
    print("saved:", p)

    # ------------------------------------------------------------- captions
    wtxt = ", ".join(f"{SHORT[c]}={w[c]:.3f}" for c in XCOLS)
    # The caption must state the weights ACTUALLY used, not the default.
    WEIGHT_NOTE = {
        "sfa": "The weights are the output ELASTICITIES from the converged BC92 SFA.",
        "hybrid": "Labour and intermediates are weighted by their OBSERVED cost "
                  "shares (county wage x worked days, and nominal intermediate "
                  "spending); land and capital keep their BC92 SFA elasticities "
                  "because their cost level is not identified.",
        "costshare": "The weights are the OBSERVED shares of total input cost for "
                     "all four inputs. This is a bound showing what the price data "
                     "literally say, not the recommended index: land and capital "
                     "together take under 0.5%, which is not credible.",
    }[a.weights]
    CAPTIONS[f"fig9agg_frontier{SFX}"] = (
        f"Figure 9 (aggregated inputs). Output against a single aggregate input "
        f"index, {', '.join(map(str, YEARS))}; {d.countyid.nunique():,d} agricultural "
        "counties, 2005 prices. The index is a constant-returns geometric "
        f"(fixed-weight Tornqvist) aggregate of labour, land service, capital service "
        f"and intermediate inputs with weights {wtxt}, each input first divided by its "
        "sample geometric mean so the index is unit-free and equals 1 at the average "
        f"bundle. {WEIGHT_NOTE} Cost shares are only half available. Labour can be "
        "priced with the county wage in data/wage_CropIndustry.dta and intermediates "
        "are supplied in nominal yuan, but land and capital, although the codebook "
        "defines both as nominal yuan deflated by Fixed_PI_2005, come out at 0.02% and "
        "0.10% of nominal output -- three orders of magnitude too small, and with a "
        "level that no single constant reconciles in the capital case. Under constant "
        "returns with factors paid their marginal product the output elasticity equals "
        "the cost share, which is the standard substitute where the price level is "
        "missing. Panels: (A) levels, (B) per labour day, (C) per unit of land service. "
        "In (B) and (C) the denominator also enters the index; this is the Kumar & "
        "Russell (2002) convention and is legitimate because the weights sum to one, "
        "but the two axes are not independent. Each year's boundary is an order-alpha "
        f"quantile frontier (alpha={a.alpha:g}) computed on that year's observations only. "
        "Log scales; points are county-years.")
    CAPTIONS[f"fig10agg_isoquant{SFX}"] = (
        f"Figure 10 (aggregated inputs). Unit isoquants, {', '.join(map(str, YEARS))}. "
        "A scalar input index has no isoquant -- with one input there is nothing to "
        "trade off -- so one input is HELD OUT and plotted against the "
        "constant-returns aggregate of the remaining three, both divided by the "
        "county's own real GVP. Panel (A) holds out labour, panel (B) land service. "
        "The boundary is the lower-left envelope of the output-normalised cloud "
        "(Pareto-minimal chain, then lower convex hull), the Farrell / Kumar & Russell "
        "(2002) convention and the same construction as Figure B of "
        "52_frontier_isoquant_figs.py. An isoquant closer to the origin is the more "
        "productive technology. The printed diagnostic reports how many observations "
        "each envelope rests on: on this panel it is typically a handful out of ~1,850, "
        "because at a given output the cross-county variation is an efficiency gradient "
        "rather than a factor-substitution margin. Log scales; points are county-years.")
    cap = os.path.join(OUT, f"captions_fig9agg_10agg{SFX}.md")
    with open(cap, "w", encoding="utf-8") as fh:
        fh.write("# Figure captions - aggregated-input Fig 9 / Fig 10\n\n")
        for k in sorted(CAPTIONS):
            fh.write(f"## {k}\n\n{CAPTIONS[k]}\n\n")
    print("\n" + "=" * 78)
    for k in sorted(CAPTIONS):
        print(f"\n[{k}]\n{CAPTIONS[k]}")
    print(f"\ncaptions -> {cap}")


if __name__ == "__main__":
    main()
