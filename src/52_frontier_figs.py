# -*- coding: utf-8 -*-
"""
ALL production-frontier and isoquant figures, from one frontier engine.
a best-practice boundary over the county input-output cloud. Every boundary now comes from _frontier.py, so the
technology is defined once.

FAMILIES  (--family, default all)
  ratio      Y/L against X/L, one X at a time (K, M, Land), per-year boundary.
             The two-dimensional view: a scalar index would collapse exactly the
             substitution structure this shows.                 
  isoquant   Unit isoquants: L/Y against X/Y, per-year lower-left boundary.
                                                    
  aggregate  Y against the fixed-weight geometric input index, and against each
             raw input on its own axis.  The one-input production function,
             directly comparable to the aggregate TFP indices.  The hold-one-out
             isoquants that used to live here are gone: they kept the CRS radial
             normalisation that fam_isoquant has since replaced with an LP, so
             the same family contradicted itself, and one raw input against a
             three-input geometric index is a hybrid axis nobody could read.
  Weather figures live in 56_weather_figs.py, NOT here.  The input-weather
  isoquants this file used to draw divided WEATHER by output, which has no
  interpretation: weather is neither chosen nor scalable, so "GDD per unit of
  real GVP" is not a quantity.  56_ solves the weather frontier properly, as a
  surface f(x, w) with w entering the LP as an equality (weak disposability).

BY REGION  (--by-region)
  Every family above can be faceted by Nine-Agri-Regions, assigned at province level
  (_frontier.REGION9).  Each panel gets its OWN boundary computed on its own
  counties, so a region's frontier is not contaminated by another's.

RETURNS TO SCALE
  Two different things are both called CRS and they must not be conflated:
  the aggregate INDEX is always homogeneous of degree one (weights renormalised
  to sum to 1 -- that is what makes x/L meaningful), while the BOUNDARY carries
  its own assumption, --rts, default NIRS to match 20_dea_tfp.py.  See the
  _frontier module docstring for why CRS is wrong for the boundary specifically:
  it makes the one-input frontier a RAY, assuming away the diminishing returns
  the figure exists to display.  --rts crs draws it so the collapse is visible.

Run:  python src/52_frontier_figs.py
      python src/52_frontier_figs.py --family aggregate --by-region
      python src/52_frontier_figs.py --rts crs --family ratio
"""
from __future__ import annotations
import os, re, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C
import _tfp as T
import _frontier as F

C.set_cjk_font(plt)
OUT_FIG = os.path.join(C.FIG_DIR, "frontier")
OUT_CSV = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_CSV, exist_ok=True)

INK, INK2, GRID = "#1a1a19", "#5c5b55", "#e6e5e0"
# Sample starts in 1985.  capital_serv_q behaves differently before it: the
# median K/M runs 0.008-0.012 over 1981-84 against 0.0023-0.0039 from 1985 on
# (1981 alone is 3.9x the 1985-90 mean), and county coverage jumps from 1,082 in
# 1984 to 1,819 in 1985 -- so the early block is both a different series and a
# different sample.  This filters the SAMPLE, not just the drawn years, because
# the sample is also the DEA reference set.
FROM_YEAR = 1985
DEFAULT_YEARS = [1985, 1989, 1993, 1998, 2002, 2007, 2011, 2016]
XCOLS = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
SHORT = {"Laborday_impute": "L", "Land_serv_q": "Land",
         "capital_serv_q": "K", "Inter_all_real": "M"}
DESC = {"Land_serv_q": "land service", "capital_serv_q": "capital service",
        "Inter_all_real": "intermediate inputs (real)",
        "Laborday_impute": "labour days"}
# GVP is 万元, every input VALUE series is 元 -- see _frontier docstring.
WAN = 1e4
CAPTIONS = {}

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
def observed_cost_shares(d, quiet=False):
    """Median share of TOTAL INPUT COST, in yuan, rebuilt from the codebook:
        labour        Laborday_impute * wage_avg     (wage_CropIndustry.dta)
        land          Land_serv_q    * Fixed_PI_2005
        capital       capital_serv_q * Fixed_PI_2005
        intermediates Inter_all_nom                  (already nominal)

    Land and capital come out three orders of magnitude below labour and
    intermediates.  Prices are NOT the missing piece -- Fixed_PI_2005 is
    supplied and applying it changes nothing about the order of magnitude.
    Land is consistent with a single constant (~1000x) that reconciles both its
    level and its whole time profile.  Capital is not: real capital service per
    county grows 68.8x from 1981 to 2015 against 10.9x for intermediates, so
    K/M DRIFTS by a factor of ~11 and no rescaling fixes early and late years
    together.  Both series also STEP at 2005, the base year of Fixed_PI_2005.
    """
    pi = pd.read_stata(os.path.join(C.DATA, "priceindex_province.dta"),
                       convert_categoricals=False)[["SID", "year", "Fixed_PI_2005"]]
    wg = pd.read_stata(os.path.join(C.DATA, "wage_CropIndustry.dta"),
                       convert_categoricals=False)[["countyid", "year", "wage_avg"]]
    s = (d.merge(pi.drop_duplicates(["SID", "year"]), on=["SID", "year"], how="left")
          .merge(wg.drop_duplicates(["countyid", "year"]), on=["countyid", "year"],
                 how="left"))
    cost = pd.DataFrame({
        "Laborday_impute": s.Laborday_impute * s.wage_avg,
        "Land_serv_q": s.Land_serv_q * s.Fixed_PI_2005,
        "capital_serv_q": s.capital_serv_q * s.Fixed_PI_2005,
        "Inter_all_real": s.Inter_all_nom})
    ok = cost.notna().all(axis=1) & (cost > 0).all(axis=1)
    cost, sub = cost[ok], s[ok]
    tot = cost.sum(axis=1)
    sh = {c: float((cost[c] / tot).median()) for c in cost.columns}
    if not quiet:
        gvp = sub.GVP_allagr_impute * WAN            # 万元 -> 元
        print(f"observed cost shares from {int(ok.sum()):,d} of {len(s):,d} county-years")
        print(f"  {'input':6s} {'share of input cost':>20s} {'share of nominal GVP':>21s}")
        for c in XCOLS:
            print(f"  {SHORT[c]:6s} {sh[c]:20.4f} {float((cost[c]/gvp).median()):21.4f}")
    return sh


def index_weights(d, scheme):
    beta = dict(T.BETA_AG)
    if scheme == "sfa":
        print(f"weights: BC92 SFA output elasticities [source: {T.BETA_SOURCE}], "
              f"renormalised to sum to 1  (raw RTS = {sum(beta.values()):.4f})")
        print("  (only the RELATIVE elasticities survive renormalisation, so the "
              "level of the\n   SFA's RTS estimate does not enter the index)")
        return beta
    sh = observed_cost_shares(d)
    if scheme == "costshare":
        print("weights: OBSERVED cost shares -- a BOUND on what the price data "
              "literally say, not the recommended index")
        return sh
    keep = beta["Land_serv_q"] + beta["capital_serv_q"]
    r = sh["Laborday_impute"] / (sh["Laborday_impute"] + sh["Inter_all_real"])
    print(f"weights: HYBRID -- observed L:M = {r:.3f}:{1-r:.3f}, land and K at "
          f"their SFA elasticities")
    return {"Land_serv_q": beta["Land_serv_q"], "capital_serv_q": beta["capital_serv_q"],
            "Laborday_impute": (1 - keep) * r, "Inter_all_real": (1 - keep) * (1 - r)}


# ---------------------------------------------------------------------------
KLAB_TXT = {
    "dea": lambda a: f"Boundary: DEA {a.rts.upper()} envelope "
                     "observations.",
    "alpha": lambda a: f"Boundary: order-alpha quantile (alpha={a.alpha:g}).",
}


def boundary(xv, yv, kind, alpha, rts, tag):
    """One call site for every output-side boundary, so `ratio` and `aggregate`
    cannot drift apart again.

    Returns (x, y, n_defining, vx, vy) -- the last two are the KINKS of the
    piecewise-linear DEA envelope, which the caller marks.  Without them a DEA
    curve on log axes is genuinely ambiguous to a reader: each segment is
    straight in LEVELS with a falling slope (diminishing marginal returns), but
    a log-log plot draws the ELASTICITY, which rises along a segment, so the
    curve bows upward and looks convex.  Marking the vertices shows that the
    object is a chain of straight segments and that the bowing happens BETWEEN
    kinks, not at them.  (None, None) for the quantile boundary, whose bin
    points carry no comparable interpretation.
    """
    if kind == "dea":
        hx, hy, n, vx, vy = F.dea_frontier(xv, yv, rts=rts, tag=tag,
                                           return_vertices=True)
        return hx, hy, n, vx, vy
    hx, hy = F.quantile_frontier(xv, yv, alpha=alpha, tag=tag)
    return hx, hy, np.nan, None, None


def _xtrim(kinks, xv, scale, q, kq=0.5, margin=1.4, cap=0.90):
    """Right-hand x limit, chosen from where the frontiers still DIFFER.

    Past its last KINK a year's envelope is the flat free-disposal
    extrapolation: a dashed horizontal line carrying no information, held up by
    one county.  The years separate from each other only on the RISING part, so
    that is what the axis should show.

    Using the FURTHEST kink was the first attempt and it failed for a measurable
    reason: across the five panels max(kink)/median(kink) runs 1.8x to 8.6x
    (intermediate inputs is the worst), so a single late-kinking year set the
    scale and squeezed every rising segment into the left margin.  The limit is
    now the MEDIAN kink with a margin -- half the years still show where their
    flat tail begins, the other half run off the right edge, and the region
    where the curves actually diverge fills the plot.  It is additionally capped
    at the 90th percentile of the data so the axis never runs past the counties.
    Falls back to the quantile when there are no kinks (order-alpha has none).
    """
    v = pd.to_numeric(xv, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    qhi = v.quantile(q if q is not None else .95)
    k = np.asarray([x for x in np.atleast_1d(kinks)
                    if np.isfinite(x) and x > 0], float)
    if not len(k):
        return qhi
    return float(min(qhi, v.quantile(cap), np.quantile(k, kq) * margin))


def _lims(v, scale, top=None, q=None):
    """Axis limits.

    Log axes can show the 0.5-99.5 percentile span.  Linear axes cannot on the
    OUTPUT axis -- the top 0.5% is 10-100x the median and would squash the rest
    into the corner -- but on the INPUT axis they can and should: the frontier
    is flat out there, so extending x costs no vertical range and cutting it
    hides observations that the envelope is supposed to cover.  Hence separate
    quantiles, `q`, per axis.  Linear starts at ZERO, which is the only origin a
    level-space concave hull can be read against (the NIRS first segment is the
    ray from it).
    """
    v = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    hi = v.quantile(q if q is not None else .95)
    return 0.0, max(hi, 1.06 * top) if top else hi


def grid_for(n):
    """Facet layout for up to nine regions."""
    ncol = 3 if n > 4 else (2 if n > 1 else 1)
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


def draw_dea(ax, hx, hy, vx, vy, col, label, lw=2.0):
    """Draw a DEA envelope with its free-disposal tail DASHED, and mark the kinks.

    The abrupt turn to horizontal is not a drawing artefact and not NIRS: it is
    free disposability of x.  The hull's last vertex is the highest-output county
    in that year, and beyond it `maximum.accumulate` holds f flat, because more
    input can never REDUCE maximum output.  NIRS and VRS give an identical tail
    (measured, 2012: both 32.8% of the curve, same f at x_max); only CRS has none.

    But the two halves are not the same kind of statement.  Up to the peak the
    boundary is DEFINED BY OBSERVATIONS.  Past it, nothing was observed producing
    more, so the flat segment is pure extrapolation from the disposability axiom
    -- and on this panel it covers 61-99% of the input range, carried by ONE
    county (2012: 330903 普陀区; 1989/1999/2003: 371082 荣成市).  Drawing both in
    the same solid line invites the reader to treat the second as evidence.
    Dashing it says which part the data support.
    """
    if vx is None or len(vx) < 2:
        ax.plot(hx, hy, "-", color=col, lw=lw, zorder=4, label=label)
        return
    k = int(np.argmax(vy))                       # peak = last data-defined vertex
    xk = vx[k]
    # The two pieces must OVERLAP BY ONE POINT.  Splitting on disjoint masks
    # (hx <= xk and hx > xk) leaves the solid end and the dashed start one
    # densification step apart -- measured 1.65% of the x range on the 2005
    # capital panel -- which reads as a broken line, and makes the kink marker
    # at the peak look detached from both halves.
    i = int(np.searchsorted(hx, xk * (1 + 1e-12), side="right"))
    i = max(i, 1)
    ax.plot(hx[:i], hy[:i], "-", color=col, lw=lw, zorder=4, label=label)
    if i < len(hx):                              # free-disposal extrapolation
        ax.plot(hx[i - 1:], hy[i - 1:], "--", color=col, lw=lw * .8, zorder=4,
                dashes=(4, 3), alpha=.85)
    ax.plot(vx[:k + 1], vy[:k + 1], "o", ms=4.5, color=col, mec="white",
            mew=.7, zorder=5)


def _legend(ax):
    """Legend OUTSIDE the axes, to the right.  Inside, it sat on top of the
    frontier's flat tail -- exactly the region the reader needs to see, because
    that is where the envelope is carried by a single county.  Only drawn if
    something was actually plotted, so a region too small for any boundary does
    not emit a matplotlib warning and an empty box."""
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7, title="year", loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=False)


def finish(fig, stub, caption=None):
    """Write the figure and register its caption.  Captions live OUTSIDE the
    image -- they eat plot area, cannot be copy-edited, and are unreadable at
    publication size -- so they are collected here and written to a markdown
    file at the end of the run.  Nothing is drawn on the axes by this."""
    fig.tight_layout()
    p = os.path.join(OUT_FIG, f"{stub}.png")
    fig.savefig(p, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    if caption:
        CAPTIONS[stub] = caption
    print("  saved", os.path.basename(p))


def draw_isoquant(ax, hx, hy, col, yr, xhi, yhi, i, n):
    """Draw one unit isoquant WITH its free-disposability extensions.

    Both conventional inputs are freely disposable -- using more of either is
    always feasible -- so the technology set does not stop at the last efficient
    vertex: it continues flat to the right of the boundary's end and vertically
    above its start.  Drawing only the efficient chain shows a curve floating in
    the middle of the plot and hides the fact that the region above and right of
    it is entirely feasible.  The extensions are what make the picture an
    input-requirement SET rather than a line.  (Weather is the exception -- it is
    weakly disposable, so figCD_* deliberately gets no extension.)
    """
    ex = np.concatenate([[hx[0]], hx, [xhi]])
    ey = np.concatenate([[yhi], hy, [hy[-1]]])
    ax.plot(ex, ey, "-", color=col, lw=2.0, zorder=4, label=str(yr),
            solid_capstyle="round")


def _axfmt(ax, xlab, ylab, title, scale="linear"):
    """LOG is the default because every plotted quantity spans 1.5-3.5 orders of
    magnitude (see 24_io_index_descriptives.py, Table 3), and on linear axes the
    cloud collapses into the bottom-left corner.

    LINEAR is offered because it is the only scale on which the DEA envelope
    LOOKS like what it IS.  The hull is piecewise linear and concave in LEVELS;
    on log axes each straight segment draws as an upward-bowing arc, because a
    log-log plot shows the ELASTICITY b*x/(a+b*x), which rises along a segment
    even though the marginal product b is constant on it and falls at every
    kink.  On linear axes that artefact is gone: straight segments are straight,
    and the falling slope IS the diminishing marginal return.  The price is the
    scatter, which becomes unreadable -- so the two scales answer different
    questions and neither replaces the other.
    """
    # LINEAR ONLY.  Log axes were removed: a level-space concave hull draws as an
    # upward-bowing arc on log-log, because the plot then shows the ELASTICITY
    # b*x/(a+b*x), which rises along a segment even though the marginal product
    # b is constant on it and falls at every kink.  Readers consistently took the
    # bow for a claim about the technology.  On linear axes a straight segment is
    # straight and the falling slope IS the diminishing marginal return.
    ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold", color=INK)
    ax.grid(alpha=.25, which="major")


# ============================================================ FAMILY: ratio
def fam_ratio(d, YEARS, cols, a, groups):
    """Y/L against X/L, per-year boundary, one X at a time."""
    for X in ["capital_serv_q", "Inter_all_real", "Land_serv_q"]:
        nrow, ncol = grid_for(len(groups))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, 5.0 * nrow),
                                 squeeze=False)
        rows = []
        for ax, (gname, g) in zip(axes.ravel(), groups):
            xv_all = (g[X] / g.Laborday_impute).replace([np.inf, -np.inf], np.nan)
            yv_all = (g.real_gvp / g.Laborday_impute).replace([np.inf, -np.inf], np.nan)
            ftop, xkink = 0.0, []
            for yr, col in zip(YEARS, cols):
                m = (g.year == yr).values
                ax.scatter(xv_all[m], yv_all[m], s=8, alpha=.20, color=col,
                           linewidths=0, rasterized=True)
            for yr, col in zip(YEARS, cols):
                m = (g.year == yr).values
                if m.sum() < 40:
                    continue
                hx, hy, n, vx, vy = boundary(xv_all[m].values, yv_all[m].values,
                                     a.solo, a.alpha, a.rts,
                                     f"[{SHORT[X]}/L {gname[:14]}] {yr}")
                if len(hx) < 2:
                    continue
                ftop = max(ftop, float(np.nanmax(hy)))
                draw_dea(ax, hx, hy, vx, vy, col, f"{yr} (n={int(m.sum()):,d})", 2.2)
                if vx is not None and len(vx):
                    xkink.append(float(vx[int(np.argmax(vy))]))
                rows.append(pd.DataFrame({"group": gname, "input": SHORT[X],
                                          "year": yr, "x_XL": hx, "f_YL": hy}))
            _axfmt(ax, f"{SHORT[X]}/L — {DESC[X]} per labour day{AXTAG}",
                   f"Y/L — real GVP per labour day{AXTAG}",
                   f"{gname}  (n={g.countyid.nunique():,d})", a.scale)
            ax.set_xlim(0, _xtrim(xkink, xv_all, a.scale, a.xq, a.xkink_q))
            ax.set_ylim(*_lims(yv_all, a.scale, ftop))
            _legend(ax)
        for ax in axes.ravel()[len(groups):]:
            ax.axis("off")
        stub = f"fig9_frontier_{SHORT[X]}L{a.sfx}"
        if rows:
            pd.concat(rows).to_csv(os.path.join(OUT_CSV, f"{stub}.csv"), index=False)
        fig.suptitle(f"Per-worker production frontier: output per labour day against "
                     f"{DESC[X]} per labour day\n"
                     f"{a.blab}, separate cross-section per year",
                     fontsize=12.5, fontweight="bold", color=INK)
        finish(fig, stub)

# ========================================================= FAMILY: isoquant
def fam_isoquant(d, YEARS, cols, a, groups):
    """Unit isoquants solved by LINEAR PROGRAMMING, no radial normalisation.

    The previous construction divided each county's inputs by its own output and
    took the lower-left envelope of the scaled cloud.  That is exact only under
    CRS: lambda = 1/y gives f(x/y) = 1 under constant returns, but under NIRS
    only f(x/y) >= 1, so the scaled points are feasible for unit output yet
    generally interior, and their envelope is an OUTER bound.  The BC92 estimate
    here is RTS = 0.455, so the bound was not tight.  It also produced a
    downward slope BY CONSTRUCTION -- the Pareto chain keeps a point only when
    its ordinate is a new running minimum -- so the slope could not be read as a
    substitution rate.

    Now each point is one LP on the same NIRS technology the rest of the project
    uses (see _frontier.dea_isoquant), at a COMMON output level for every year
    and every region so the curves are comparable, with the two inputs not on
    the axes held at the pooled median.  The slope is now a genuine marginal
    rate of substitution and the left edge is a genuine infeasibility.
    """
    ALLC = XCOLS
    ystar = float(d.real_gvp.median())
    ref = d[ALLC].median().values
    i1 = ALLC.index("Laborday_impute")
    for X in ["capital_serv_q", "Inter_all_real", "Land_serv_q"]:
        i2 = ALLC.index(X)
        nrow, ncol = grid_for(len(groups))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, 5.0 * nrow),
                                 squeeze=False)
        rows = []
        for ax, (gname, g) in zip(axes.ravel(), groups):
            # Does a substitution margin exist at all?  If corr(ln L, ln X) at a
            # given output is POSITIVE, counties using more labour use more of
            # everything and the trade-off the isoquant traces is thin.
            band = g[(g.real_gvp > .85 * ystar) & (g.real_gvp < 1.15 * ystar)]
            rho = (np.corrcoef(np.log(band.Laborday_impute), np.log(band[X]))[0, 1]
                   if len(band) > 30 else np.nan)
            # ONE grid for every year in the panel, taken from the union of the
            # per-year informative ranges, so a vertical gap between curves is a
            # change in the technology and not a change of grid
            los, his, per = [], [], {}
            for yr in YEARS:
                s = g[g.year == yr]
                if len(s) < 60:
                    continue
                Xm, yv = s[ALLC].values, s.real_gvp.values
                lo, hi = F.isoquant_range(Xm, yv, ystar, i1, i2, ref=ref)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    per[yr] = (Xm, yv); los.append(lo); his.append(hi)
            if not per:
                ax.axis("off"); continue
            # A single log-spaced grid over the UNION of the per-year ranges
            # starves the narrow years: 2015's informative band is 420k-450k
            # while 2001's is 3.5m-6.5m, so a 34-point union grid put ZERO
            # points inside 2015's own range.  The grid is instead the union of
            # per-year sub-grids, so every year is resolved on its own band and
            # all years still share one evaluation set.
            lo, hi = min(los), max(his)
            grid = np.unique(np.concatenate(
                [np.exp(np.linspace(np.log(l), np.log(h * 1.06), 18))
                 for l, h in zip(los, his)]))
            nfe = 0
            for yr, col in zip(YEARS, cols):
                if yr not in per:
                    continue
                Xm, yv = per[yr]
                z = F.dea_isoquant(Xm, yv, ystar, i1, i2, grid, ref=ref)
                ok = np.isfinite(z)
                nfe += int((~ok).sum())
                if ok.sum() < 2:
                    continue
                gx, gz = grid[ok], z[ok]
                ax.plot(gx, gz, "-", lw=1.8, color=col, label=str(yr), zorder=4)
                # Mark the KINKS, not the grid.  A DEA isoquant is the boundary
                # of a polyhedron, so it is piecewise linear and its economically
                # meaningful points are the vertices, where the marginal rate of
                # substitution changes.  Drawing a marker at every grid node
                # instead advertised the resolution of the sweep as if it were
                # structure, and it was inconsistent with the frontier figures,
                # where a marker means a hull vertex.
                if len(gx) >= 3:
                    sl = np.diff(gz) / np.diff(gx)
                    j = np.where(np.abs(np.diff(sl)) >
                                 0.02 * np.maximum(np.abs(sl[:-1]), 1e-12))[0] + 1
                    if len(j):
                        ax.plot(gx[j], gz[j], "o", ms=4, color=col, mec="white",
                                mew=.6, zorder=5)
                rows.append(pd.DataFrame({"group": gname, "input": SHORT[X],
                                          "year": yr, "L": grid[ok], "X_min": z[ok]}))
            print(f"    [iso {SHORT[X]} {gname[:14]}] y*={ystar:,.0f}, "
                  f"L grid {lo:,.0f}-{hi:,.0f} ({hi/lo:.1f}x), "
                  f"{nfe} infeasible cells, corr(lnL,ln{SHORT[X]})={rho:+.2f}")
            _axfmt(ax, "Labour (worked days)", f"{DESC[X]} — minimum to reach y*",
                   f"{gname}  (corr={rho:+.2f})" if np.isfinite(rho) else gname)
            _legend(ax)
        for ax in axes.ravel()[len(groups):]:
            ax.axis("off")
        stub = f"fig10_isoquant_L{SHORT[X]}{a.sfx}"
        if rows:
            pd.concat(rows).to_csv(os.path.join(OUT_CSV, f"{stub}.csv"), index=False)
        fig.suptitle(f"Unit isoquant: labour vs {DESC[X]} at y* = {ystar:,.0f} "
                     f"— DEA {a.rts.upper()}, solved by LP",
                     fontsize=10.5, fontweight="bold", color=INK)
        finish(fig, stub)


# ======================================================== FAMILY: aggregate
def fam_aggregate(d, YEARS, cols, a, groups, beta):
    """Y against the scalar fixed-weight geometric input index, three normalisations."""
    PANELS = [("x", "Aggregate input index x", "aggregate"),
              ("Laborday_impute", "Labour (worked days)", "labour"),
              ("Land_serv_q", "Land service (2005-price yuan)", "land"),
              ("capital_serv_q", "Capital service (2005-price yuan)", "capital"),
              ("Inter_all_real", "Intermediate inputs (2005-price yuan)", "intermediate")]
    # --frontier both puts the two estimators SIDE BY SIDE on identical axes
    KINDS = ["dea", "alpha"] if a.frontier == "both" else [a.frontier]
    KLAB = {"dea": f" DEA {a.rts.upper()} envelope",
            "alpha": f"order-alpha quantile (alpha={a.alpha:g})"}
    for xcol, xlab, pname in PANELS:
        for kind in KINDS:
            nrow, ncol = grid_for(len(groups))
            fig, axes = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, 5.0 * nrow),
                                     squeeze=False)
            rows, nvert = [], []
            for ax, (gname, g) in zip(axes.ravel(), groups):
                yv = g.real_gvp
                xv = g[xcol]
                ftop, xkink = 0.0, []
                for yr, col in zip(YEARS, cols):
                    m = (g.year == yr).values
                    ax.scatter(xv[m], yv[m], s=8, alpha=.20, color=col, linewidths=0,
                               zorder=1, rasterized=True)
                for i, (yr, col) in enumerate(zip(YEARS, cols)):
                    m = (g.year == yr).values
                    if m.sum() < 40:
                        continue
                    hx, hy, n, vx, vy = boundary(xv[m].values, yv[m].values, kind,
                                         a.alpha, a.rts,
                                         f"[{pname} {kind} {gname[:14]}] {yr}")
                    if len(hx) < 2:
                        continue
                    if np.isfinite(n):
                        nvert.append(n)
                    ftop = max(ftop, float(np.nanmax(hy)))
                    draw_dea(ax, hx, hy, vx, vy, col, str(yr))
                    if vx is not None and len(vx):
                        xkink.append(float(vx[int(np.argmax(vy))]))
                    rows.append(pd.DataFrame({"group": gname, "panel": pname,
                                              "estimator": kind, "year": yr,
                                              "x": hx, "f_y": hy}))
                _axfmt(ax, xlab + AXTAG, "Real GVP" + AXTAG,
                       f"{gname}  (n={g.countyid.nunique():,d})", a.scale)
                ax.set_xlim(0, _xtrim(xkink, xv, a.scale, a.xq, a.xkink_q))
                ax.set_ylim(*_lims(yv, a.scale, ftop))
                _legend(ax)
            for ax in axes.ravel()[len(groups):]:
                ax.axis("off")
            # the estimator is always in the filename, so two runs never collide
            stub = f"fig9_{pname}_{kind}{a.sfx}"
            if rows:
                pd.concat(rows).to_csv(os.path.join(OUT_CSV, f"{stub}.csv"), index=False)
            fig.suptitle(f"Real GVP vs {xlab.split(' (')[0].lower()} — {KLAB[kind]}",
                         fontsize=10.5, fontweight="bold", color=INK)
            finish(fig, stub)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all",
                    choices=["all", "ratio", "isoquant", "aggregate"])
    ap.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    ap.add_argument("--from-year", type=int, default=FROM_YEAR,
                    help="drop everything before this year from the SAMPLE, not "
                         "just from the drawn years -- it is also the DEA "
                         "reference set. Default 1985; see the note by "
                         "DEFAULT_YEARS for why")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--exclude-counties", default=None,
                    help="CSV with a countyid column; those counties are dropped "
                         "from the SAMPLE, hence from the DEA reference set, so "
                         "they can no longer define the frontier. Use with "
                         "--outdir so the excluded run never overwrites the "
                         "headline figures")
    ap.add_argument("--outdir", default=None,
                    help="subdirectory of src/figures/frontier to write into")
    ap.add_argument("--frontier", choices=["alpha", "dea", "both"], default="both",
                    help="which boundary the aggregate family draws. both writes TWO "
                         "SEPARATE files per panel, one per estimator, on identical "
                         "axes -- never side by side in one figure")
    ap.add_argument("--rts", choices=["nirs", "vrs", "crs"], default="nirs",
                    help="returns to scale of the BOUNDARY (the index is always "
                         "homogeneous of degree one). nirs matches 20_dea_tfp.py; "
                         "crs collapses the boundary to a ray -- see _frontier docstring")
    ap.add_argument("--weights", choices=["sfa", "hybrid", "costshare"], default="sfa")
    ap.add_argument("--xkink-q", type=float, default=0.5,
                    help="quantile of the per-year frontier kinks that sets the "
                         "right-hand x limit (0.5 = median). Lower zooms further "
                         "into the region where the years differ")
    ap.add_argument("--xq", type=float, default=0.99,
                    help="linear scale only: quantile of the INPUT axis to show. "
                         "The frontier is flat at the top end, so extending x is "
                         "nearly free and cutting it hides covered observations")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="isoquant sensitivity only: drop this share of the most "
                         "extreme input ratios BEFORE building the envelope. "
                         "Default 0 -- an envelope is defined by its extremes, so "
                         "trimming one is a different estimator, not a robustness "
                         "tweak")
    ap.add_argument("--min-counties", type=int, default=25,
                    help="drop a region facet with fewer counties than this")
    ap.add_argument("--split", action="store_true",
                    help="one FILE per group instead of one figure with facets. "
                         "With --by-region that is one file per 九大农业区, which "
                         "is the readable option when each panel carries a scatter "
                         "of ~1,900 counties plus eight year-frontiers")
    ap.add_argument("--by-region", action="store_true",
                    help="facet by Nine-Agri-Regions")
    a = ap.parse_args()
    a.scale = "linear"                       # log axes removed, see _axfmt
    a.sfx = ("_byregion" if a.by_region else "") + \
            ("" if a.weights == "sfa" else "_" + a.weights) + \
            ("" if a.rts == "nirs" else "_" + a.rts) +             ("" if abs(a.alpha - 0.95) < 1e-9 else "_a%g" % (100 * a.alpha)) +             ("_excl" if a.exclude_counties else "")
    # families other than `aggregate` draw ONE boundary, so `both` picks the
    # order-alpha estimator there -- it is the one used for every substantive
    # reading, the DEA envelope being too fragile on ~1,850 counties.
    a.solo = "alpha" if a.frontier in ("alpha", "both") else "dea"
    globals()["AXTAG"] = ""
    a.blab = ("DEA envelope and order-alpha quantile, as separate figures"
              if a.frontier == "both" else
              f"order-alpha quantile boundary (alpha={a.alpha:g})" if a.frontier == "alpha"
              else f" DEA {a.rts.upper()} envelope")

    if a.outdir:
        globals()["OUT_FIG"] = os.path.join(C.FIG_DIR, "frontier", a.outdir)
        os.makedirs(OUT_FIG, exist_ok=True)
        print(f"figures -> {OUT_FIG}")

    d = T.load_panel()
    n0 = len(d)
    d = d[d.year >= a.from_year]
    if a.exclude_counties:
        ex = pd.read_csv(a.exclude_counties)
        ex = set(pd.to_numeric(ex["countyid"], errors="coerce").dropna().astype(int))
        # Dropped from the SAMPLE, which is also the reference set, so these
        # counties can no longer define any boundary.  They are removed entirely
        # rather than merely unscored: leaving them in the reference set and
        # hiding their marker would keep exactly the leverage the exclusion is
        # meant to remove (60_leave_one_out.py measures how large that is --
        # in Yangtze 2007/2011 dropping two counties moved EVERY other county's
        # efficiency score by more than 5%).
        nb = d.countyid.nunique()
        d = d[~d.countyid.isin(ex)]
        print(f"excluded {len(ex)} counties listed in "
              f"{os.path.basename(a.exclude_counties)}: "
              f"{nb:,d} -> {d.countyid.nunique():,d} counties, {len(d):,d} county-years")
    print(f"sample from {a.from_year}: {len(d):,d} county-years "
          f"({n0 - len(d):,d} dropped), {d.countyid.nunique():,d} counties")
    YEARS = [int(y) for y in a.years.split(",") if int(y) >= a.from_year]
    missing = [y for y in YEARS if y not in set(d.year)]
    if missing:
        raise SystemExit(f"years not in the panel: {missing}")
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} agricultural counties, "
          f"{d.year.min()}-{d.year.max()}")

    d = F.add_region(d)
    print("\nNine-Agri-Regions, counties in the agricultural sample:")
    rc = F.region_counts(d)
    for r, n in rc:
        print(f"  {r:42s} {n:5,d}")
    absent = [r for r in F.REGION_ORDER if r not in [x[0] for x in rc]]
    if absent:
        print(f"  absent from the agricultural sample: {', '.join(absent)}")

    beta = index_weights(d, a.weights)
    d["x"], w = F.aggregate_index(d, XCOLS, beta)
    print("aggregate input index weights (renormalised to 1): "
          + ", ".join(f"{SHORT[c]}={w[c]:.4f}" for c in XCOLS))

    # A region needs enough counties for a per-year boundary to mean anything.
    # Qinghai-Tibet contributes 4 counties to the cropland>=15% sample, so its
    # panel would be an empty frame; it is dropped and the drop is reported
    # rather than silently producing a blank facet.
    if a.by_region:
        small = [(r, n) for r, n in rc if n < a.min_counties]
        for r, n in small:
            print(f"  [region] {r} has {n} counties (< {a.min_counties}) -- panel dropped")
        rc = [(r, n) for r, n in rc if n >= a.min_counties]
    groups = ([(F.REGION_SHORT[r], d[d.region == r]) for r, _ in rc] if a.by_region
              else [("All agricultural counties", d)])
    cols = plt.cm.viridis(np.linspace(0, .88, len(YEARS)))
    fams = ["ratio", "isoquant", "aggregate"] if a.family == "all" else [a.family]

    def _slug(t):
        return re.sub(r"[^A-Za-z0-9]+", "", t)[:18]

    # --split turns the facet grid into one file per group.  Faceting nine
    # regions puts nine scatters of ~1,900 points and eight year-frontiers each
    # into one image; at publication size nothing in it is legible.
    batches = [[g] for g in groups] if a.split else [groups]
    base_sfx = a.sfx

    for fam in fams:
        print(f"\n--- family: {fam} ---")
        for batch in batches:
            a.sfx = base_sfx + (f"_{_slug(batch[0][0])}" if a.split else "")
            if fam == "ratio":
                fam_ratio(d, YEARS, cols, a, batch)
            elif fam == "isoquant":
                fam_isoquant(d, YEARS, cols, a, batch)
            elif fam == "aggregate":
                fam_aggregate(d, YEARS, cols, a, batch, beta)

    a.sfx = base_sfx
    print(f"\nfigures -> {OUT_FIG}\nCSVs")


if __name__ == "__main__":
    main()
