# -*- coding: utf-8 -*-
"""
Shared production-frontier machinery.

52_, 53_ and 58_ each grew their own copy of the same four estimators, and the
copies had drifted: 53's isoquant binned by labour and took a conditional
quantile, 52's and 58's took a Pareto chain then a lower convex hull, and the
two disagreed on the same data until 53 was rewritten to match.  Everything that
defines a boundary now lives here once, so a change to the technology is a change
in one place and the figures cannot silently diverge again.

WHAT THE VARIABLES ARE (农业生产数据集变量代码本, confirmed 2026-08-14)

  OUTPUT
    GVP_allagr_impute  农业总产值, NOMINAL, in 万元 (10k yuan).  Crops +
                       livestock + inseparable agricultural services.
                       real_gvp = GVP_allagr_impute / PPP_CCD_2005, where
                       PPP_2005 is the CCD farm-output price index, Hebei 2005=1.

  INPUTS
    Laborday_impute    农业劳动投入, WORKED DAYS.  A quantity, not a value.
                       1993-1996 smoothed at source.
    Land_serv_q        土地服务量.  Quality-adjusted farmland RENT: a hedonic
                       regression splits nominal rent onto soil fertility,
                       terrain and irrigation, and the fitted nominal rent is
                       divided by Fixed_PI_2005 (Hebei 2005 = 1).  Yuan.
    capital_serv_q     资本服务量.  Non-residential buildings, transport
                       equipment and farm machinery.  PIM stock with geometric
                       depreciation, aggregated by user cost (r + delta - pi),
                       divided by Fixed_PI_2005.  Yuan.
    Inter_all_real     中间投入, already deflated to 2005 prices.  Yuan.
                       Built from per-mu cost lines (seed, fertiliser,
                       pesticide, ...) weighted up by each crop's county sown
                       area.  Nominal twin Inter_all_nom is supplied.

  THE UNIT TRAP.  Output is in 万元; all three input VALUE series are in 元.
  Anything comparing an input value with output must multiply GVP by 1e4 first
  (see observed_cost_shares).  Getting this wrong looks exactly like a factor
  of 10,000 in a cost share.

RETURNS TO SCALE -- two DIFFERENT uses of "CRS", do not conflate them.

  (1) The AGGREGATE INPUT INDEX is a fixed-weight Tornqvist, and its weights are
      renormalised to sum to one.  That makes x homogeneous of degree one, which
      is what makes x/L and x/Land meaningful at all.  This is CRS in the
      AGGREGATOR and it is required whatever the technology's returns to scale
      are.  It is also why the level of the SFA's RTS estimate does not matter
      here -- only the RELATIVE elasticities survive renormalisation.

  (2) The FRONTIER BOUNDARY has its own returns-to-scale assumption, imposed on
      the hull.  NIRS (sum lambda <= 1) is the default because it matches
      20_dea_tfp.py, so every frontier in the project describes one technology.

  Why NOT CRS for the boundary, even though Fuglie (2010) works under constant
  returns: under CRS the one-input frontier IS A RAY through the origin, so
  Figure 9 collapses to a straight line with slope 1 on log axes and the
  diminishing returns the figure exists to show are assumed away by
  construction.  Fuglie's CRS is a restriction on the INDEX (cost shares sum to
  one) -- point (1) above, which this module already imposes -- not on the shape
  of an estimated boundary.  NIRS keeps the concave hull while still ruling out
  increasing returns, so it is the weaker assumption and the one that lets the
  data speak.  Pass rts="crs" to see the ray for yourself.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# 九大农业区 -- assigned at PROVINCE level, as the user's classification does.
#
# Provincial assignment is the convention in this literature and keeps the
# regions reproducible from SID alone, but it is genuinely coarse at three
# margins: northern Anhui and Jiangsu are Huang-Huai-Hai rather than Yangtze,
# western Gansu is arid rather than Loess Plateau, and western Sichuan is
# plateau rather than basin.  Those counties are carried by their province here.
# ---------------------------------------------------------------------------
REGION9 = {
    "Northeast China Plain":                    [21, 22, 23],
    "Huang-Huai-Hai Plain":                     [11, 12, 13, 37, 41],
    "Middle-lower Yangtze Plain":               [31, 32, 33, 34, 36, 42, 43],
    "Southern China":                           [35, 44, 46],
    "Sichuan Basin and surrounding regions":    [50, 51],
    "Yunnan-Guizhou Plateau":                   [52, 53, 45],
    "Loess Plateau":                            [14, 61],
    "Northern arid and semiarid region":        [15, 65, 62, 64],
    "Qinghai Tibet Plateau":                    [54, 63],
}
# Plot order: north to south, then the two high/arid regions last.  Qinghai-
# Tibet is kept in the map so a county is never silently unlabelled, but it
# almost entirely fails the cropland>=15% agricultural filter, so it normally
# drops out on its own -- reported, not hard-coded away.
REGION_ORDER = ["Northeast China Plain", "Huang-Huai-Hai Plain",
                "Loess Plateau", "Middle-lower Yangtze Plain",
                "Sichuan Basin and surrounding regions",
                "Yunnan-Guizhou Plateau", "Southern China",
                "Northern arid and semiarid region", "Qinghai Tibet Plateau"]
REGION_SHORT = {
    "Northeast China Plain": "Northeast",
    "Huang-Huai-Hai Plain": "Huang-Huai-Hai",
    "Middle-lower Yangtze Plain": "Yangtze",
    "Southern China": "South China",
    "Sichuan Basin and surrounding regions": "Sichuan Basin",
    "Yunnan-Guizhou Plateau": "Yunnan-Guizhou",
    "Loess Plateau": "Loess Plateau",
    "Northern arid and semiarid region": "N. arid/semiarid",
    "Qinghai Tibet Plateau": "Qinghai-Tibet",
}
_SID2REG = {s: r for r, ss in REGION9.items() for s in ss}


def add_region(d, col="region"):
    """Attach the 九大农业区 label from SID.  Unmapped provinces are reported
    rather than dropped, so a new SID cannot vanish from the figures unnoticed."""
    d = d.copy()
    d[col] = d["SID"].map(_SID2REG)
    miss = d[d[col].isna()]
    if len(miss):
        print(f"  [region] {miss.SID.nunique()} unmapped SID(s): "
              f"{sorted(miss.SID.unique())} -- {len(miss):,d} county-years")
    return d


def region_counts(d, col="region"):
    """Counties per region, in plot order, skipping regions with no counties."""
    g = d.dropna(subset=[col]).groupby(col).countyid.nunique()
    return [(r, int(g[r])) for r in REGION_ORDER if r in g.index and g[r] > 0]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _upper_hull(pts):
    """Upper concave hull of points sorted by x."""
    H = []
    for p in pts:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) >= 0:
            H.pop()
        H.append(p)
    return H


def _lower_hull(pts):
    """Lower convex hull of points sorted by x."""
    H = []
    for p in pts:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) <= 0:
            H.pop()
        H.append(p)
    return H


def _densify(vx, vy, n=40):
    """Interpolate along each segment so a straight line in LEVELS is drawn as a
    straight line on LOG axes.  matplotlib joins vertices in display space, so a
    two-point segment would otherwise be drawn as a curve the model never
    asserts."""
    if len(vx) < 2:
        return np.asarray(vx, float), np.asarray(vy, float)
    px, py = [], []
    for i in range(len(vx) - 1):
        t = np.linspace(0, 1, n, endpoint=(i == len(vx) - 2))
        px.append(vx[i] + t * (vx[i + 1] - vx[i]))
        py.append(vy[i] + t * (vy[i + 1] - vy[i]))
    return np.concatenate(px), np.concatenate(py)


# ---------------------------------------------------------------------------
# Output-side boundary: f(x) = max output at input x
# ---------------------------------------------------------------------------
def dea_frontier(x, y, rts="nirs", densify=40, tag="", return_vertices=False):
    """Textbook DEA boundary in (x, y).

    With one input and one output the DEA frontier IS the upper concave hull of
    the observations closed under free disposal -- exactly what solving the LP
    for every county returns, with nothing binned or smoothed.

      rts="nirs"  sum(lambda) <= 1.  Imposed by adding the ORIGIN to the point
                  set: a peer may be scaled DOWN but not UP.  The first segment
                  becomes the ray through the year's best output-input ratio and
                  no part of the boundary may sit above it.  Matches
                  20_dea_tfp.py.
      rts="vrs"   sum(lambda) = 1.  Unrestricted hull; the boundary may rise
                  above the best observed ratio at low x, asserting increasing
                  returns.
      rts="crs"   sum(lambda) free.  The boundary is the RAY through the best
                  ratio and nothing else -- a straight line, no curvature.
                  Provided so the collapse can be seen rather than argued about.

    The hull is taken in LEVELS, because convexity of the technology set is a
    level-space assumption.

    WHY THE DRAWN CURVE CAN LOOK CONVEX ON LOG AXES, and why `return_vertices`
    exists.  Each segment is a straight line in LEVELS, y = a + b*x with b
    falling at every kink -- that IS diminishing marginal returns.  But a log-log
    plot shows the ELASTICITY, b*x/(a+b*x), which RISES along a segment whenever
    a > 0 (i.e. on every segment except the first, which is a ray).  A rising
    log-log slope draws as a convex arc.  Measured on 2012: marginal product
    falls 15.7m -> 7.2m -> 0 across the three segments, while the elasticity
    inside the second segment climbs 0.458 -> 0.817.  Both facts are true; the
    figure only ever shows the second.  Densifying is still correct -- joining
    two vertices in DISPLAY space would draw a line the model never asserts --
    so the fix is to MARK the vertices, which makes the piecewise-linear
    structure visible and the bowing between them legible as a rendering of a
    straight level-space segment.

    Returns (x, y, n_defining_counties), or with `return_vertices` also the
    (vx, vy) kinks so the caller can mark them.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        e = np.array([])
        return (e, e, 0, e, e) if return_vertices else (e, e, 0)
    o = np.argsort(x); x, y = x[o], y[o]

    if rts == "crs":
        s = float(np.max(y / x))                 # best output-input ratio
        vx = np.array([x.min(), x.max()]); vy = s * vx
        if tag:
            print(f"    {tag}: DEA-CRS ray, slope {s:.4g} (1 county defines it)")
        dx, dy = _densify(vx, vy, densify)
        return (dx, dy, 1, vx, vy) if return_vertices else (dx, dy, 1)

    pts = list(zip(x, y))
    if rts == "nirs":
        pts = [(0.0, 0.0)] + pts
    H = _upper_hull(pts)
    vx = np.array([h[0] for h in H])
    vy = np.maximum.accumulate([h[1] for h in H])   # free disposal in x
    ncty = len(vx) - 1 if rts == "nirs" else len(vx)
    if rts == "nirs" and len(vx) > 1:
        # the origin cannot be drawn on a log axis; keep the RAY it defines by
        # walking the first segment back to the smallest observed input
        s = vy[1] / vx[1]
        vx, vy = vx.copy(), vy.copy()
        vx[0], vy[0] = x.min(), s * x.min()
    if tag:
        print(f"    {tag}: DEA-{rts.upper()} envelope on {ncty} of {len(x):,d} "
              f"counties ({100*ncty/len(x):.2f}%)")
    px, py = _densify(vx, vy, densify)
    return (px, py, ncty, vx, vy) if return_vertices else (px, py, ncty)


def quantile_frontier(x, y, alpha=0.95, nbins=20, minobs=30, concave=True, tag=""):
    """Order-alpha frontier: the alpha-quantile of y within equal-width bins of
    log x, made non-decreasing (Aragon, Daouia & Thomas-Agnan 2005; Daouia &
    Simar 2007).

    WHY NOT THE RAW ENVELOPE.  On ~1,850 counties the DEA envelope is set by 2-5
    observations, so it reports the most extreme county rather than the
    technology, and one mismeasured county moves a whole year's frontier.  The
    alpha-quantile is consistent, robust to outliers, and converges to the true
    frontier as alpha -> 1.

    EQUAL-WIDTH BINS IN log x, NOT EQUAL-COUNT BINS IN x.  x is heavily
    right-skewed, so an equal-count top bin spans most of a decade; the
    estimator then pairs that bin's alpha-quantile of y -- set by observations at
    the bin's RIGHT edge -- with the bin's MEDIAN x, at its left edge, and
    because y rises steeply with x that places the point far above any real
    observation.  Bins below `minobs` are dropped rather than widened.

    CONCAVITY IS IMPOSED, and in LEVELS.  Convexity of the technology set is a
    maintained axiom of this whole framework -- it is what the DEA programme
    assumes and what makes f(x) a concave frontier -- so it belongs in the
    estimator, not in a test.  It was previously left off because the concave
    hull discarded most interior bin points and, on some panels, kept only the
    two endpoints; that was recorded as "the data reject concavity".  That
    reading was wrong.  Under the maintained axiom the correct inference is the
    opposite: a bin whose alpha-quantile sags below the hull is a bin whose
    counties are INEFFICIENT, which is precisely what a frontier is supposed to
    lift them off.  Discarding the sag is the estimate, not a distortion of it.

    The hull is taken in LEVELS because concavity of f is a level-space property
    (T convex => f concave in x).  Taking it in logs imposes non-increasing
    ELASTICITY instead, which is a different and stronger restriction; measured
    on this panel the two differ little (a levels hull keeps 0-2 more vertices),
    but only the levels version is the assumption the model actually makes.

    ONE MEASURED SIDE EFFECT, reported rather than hidden.  Imposing concavity on
    top of a fixed alpha raises the EFFECTIVE alpha, because the hull lies weakly
    above the bin quantiles.  On this panel the share of counties left above the
    boundary falls from 4.1-5.3% (monotone only, i.e. the nominal 5%) to 0.8-3.8%
    (concave), so the concave boundary behaves like an alpha of roughly 0.97-0.99
    and the shift is not uniform across panels or years.  The realized share is
    printed with `tag` so the effective alpha is visible; if a target of exactly
    5% matters, lower `alpha` until the realized share lands there.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    if len(lx) < 3 * minobs:
        return np.array([]), np.array([])
    lo, hi = np.quantile(lx, [0.005, 0.995])     # a lone county cannot set the grid
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
    if concave:
        # hull in LEVELS -- see the docstring; convexity of T is a level-space
        # assumption, so the vertices are found on (x, y), not on (ln x, ln y)
        H = _upper_hull(list(zip(np.exp(bx), np.exp(by))))
        hx = np.log(np.array([h[0] for h in H]))
        hy = np.log(np.array([h[1] for h in H]))
    else:
        hx, hy = bx, by
    hy = np.maximum.accumulate(hy)            # free disposal in x
    if tag:
        # realized share above the boundary = the EFFECTIVE alpha
        fit = np.interp(lx, hx, hy, left=np.nan, right=np.nan)
        m = np.isfinite(fit)
        above = 100 * np.mean(ly[m] > fit[m]) if m.any() else np.nan
        print(f"    {tag}: {len(bx)} bins -> {len(hx)} vertices"
              + (" (concave, levels)" if concave else " (monotone only)")
              + f", {above:.1f}% of counties above the boundary"
              + f"  [nominal {100*(1-alpha):.0f}%]")
    return np.exp(hx), np.exp(hy)


# ---------------------------------------------------------------------------
# Input-side boundary: the isoquant
# ---------------------------------------------------------------------------
def lower_envelope(a, b, trim=0.0):
    """Unit isoquant: Pareto-minimal chain, then its lower convex hull.

    TRIM DEFAULTS TO ZERO, and should normally stay there.  It drops the most
    extreme b/a ratios from both tails before the chain is built -- that is, it
    removes precisely the observations an envelope is DEFINED BY.  Trimming a
    DEA-type boundary is not a robustness adjustment to the estimator, it is a
    different estimator, and one whose result cannot be described as "the
    minimum input requirement observed".  It is kept only as an explicit
    sensitivity switch; callers that use it must say so.

    The caller passes OUTPUT-NORMALISED inputs (a = A/y, b = B/y), or inputs
    radially scaled to a reference output y*; the two clouds differ only by the
    constant y*, so both give the same boundary shape.  This is the Farrell /
    Kumar & Russell (2002) convention.

    Step 1 (keep a point only if its b is a new running minimum as a increases)
    makes the boundary downward-sloping BY CONSTRUCTION, which is what an
    isoquant must be, and needs no substitution assumption to exist.  That
    matters on this panel: among counties at the same output corr(ln L, ln X) is
    POSITIVE, so counties using more labour use more of everything and there is
    no broad substitution margin.  The envelope is therefore set by a handful of
    extreme lower-left counties -- the figure is correct in form, but that count
    is what says how much weight it can carry, so callers should report it.

    THE RADIAL NORMALISATION ASSUMES CRS, and that assumption is NOT innocuous
    on this panel.  Dividing a county's inputs by its own output applies
    lambda = 1/y.  Under CRS, f(x/y) = f(x)/y = 1 exactly, so the scaled point
    sits ON the unit-output surface and the envelope of the scaled cloud IS the
    unit isoquant.  Under NIRS only the inequality f(theta x) >= theta f(x)
    holds for theta <= 1, so f(x/y) >= 1: the scaled point is FEASIBLE for unit
    output but generally INTERIOR to the unit isoquant, and the envelope is an
    OUTER bound on it rather than the thing itself.  The BC92 estimate on this
    sample is RTS = 0.455, a long way from 1, so the gap is not a technicality.
    Read these figures as "an outer bound on the unit input-requirement set,
    exact only under constant returns".

    A SECOND CAVEAT ABOUT WHAT THE SLOPE MEANS.  Step 1 keeps a point only when
    its b is a new running minimum as a increases, so the boundary slopes DOWN
    BY CONSTRUCTION whether or not the inputs substitute.  A downward slope here
    is therefore not evidence of substitution.  On this panel corr(ln L, ln X)
    among counties at the same output is POSITIVE, i.e. there is no broad
    substitution margin, and the envelope rests on a handful of extreme
    lower-left counties.  Its VERTICAL ORDERING across years is informative; its
    slope is an artefact of the construction.

    `trim` drops the most extreme b/a ratios first.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a, b = a[ok], b[ok]
    if trim > 0 and len(a) > 50:
        r = b / a
        keep = (r >= np.quantile(r, trim)) & (r <= np.quantile(r, 1 - trim))
        a, b = a[keep], b[keep]
    o = np.argsort(a); a, b = a[o], b[o]
    chain, best = [], np.inf
    for u, v in zip(a, b):
        if v < best - 1e-15:
            chain.append((u, v)); best = v
    if len(chain) < 3:
        return (np.array([c[0] for c in chain]), np.array([c[1] for c in chain]))
    H = _lower_hull(chain)
    return np.array([h[0] for h in H]), np.array([h[1] for h in H])


def dea_isoquant(X, y, ystar, i1, i2, grid, ref=None, nirs=True):
    """TRUE unit isoquant by linear programming -- no radial normalisation.

    Replaces the divide-by-own-output construction, which was exact only under
    CRS.  Dividing a county's inputs by its own output applies lambda = 1/y;
    under CRS f(x/y) = 1 exactly, but under NIRS only f(x/y) >= 1, so the scaled
    point is feasible for unit output yet generally INTERIOR to the isoquant and
    the envelope of the scaled cloud is an OUTER bound.  The BC92 estimate on
    this sample is RTS = 0.455, far from 1, so that bound is not tight.

    Here the isoquant is solved directly, at the SAME output level for every
    year, from the SAME technology the rest of the project uses:

        for each x1 on `grid`:
            min   SUM_i lam_i X[i, i2]
            s.t.  SUM_i lam_i y_i     >= ystar          the output being held
                  SUM_i lam_i X[i,i1] <= x1             the input on the x axis
                  SUM_i lam_i X[i,k]  <= ref_k          every OTHER input, held
                  SUM_i lam_i         <= 1              NIRS
                  lam >= 0

    Holding the other inputs at `ref` is what makes this a two-input SLICE of a
    four-input technology rather than a projection: leaving them free would let
    the solver buy the output with unlimited fertiliser and report a spuriously
    low labour-capital requirement.  Infeasible grid points come back NaN and
    are drawn as breaks -- they are real (that (x1, ref) combination cannot make
    ystar), not missing data.

    The slope here IS a substitution rate, unlike the Pareto-chain construction
    in `lower_envelope`, whose downward slope is imposed by the algorithm.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    if ref is None:
        ref = np.median(X, axis=0)
    others = [j for j in range(k) if j not in (i1, i2)]
    # exact pruning: the optimum of a linear programme sits at a vertex of
    # conv({0} u points), so non-vertices can be dropped without changing it
    P = np.column_stack([X, y])
    if n > 2 * (k + 1):
        sc = np.abs(P).max(axis=0); sc[sc == 0] = 1.0
        try:
            v = np.unique(ConvexHull(np.vstack([P / sc, np.zeros(k + 1)]),
                                     qhull_options="Qx").vertices)
            v = v[v < n]
            if len(v) >= k + 2:
                X, y, n = X[v], y[v], len(v)
        except Exception:
            pass
    rows = [-y, X[:, i1]] + [X[:, j] for j in others]
    if nirs:
        rows.append(np.ones(n))
    A = np.vstack(rows)
    out = np.full(len(grid), np.nan)
    for g, x1 in enumerate(grid):
        b = [-ystar, x1] + [ref[j] for j in others] + ([1.0] if nirs else [])
        r = linprog(X[:, i2], A_ub=A, b_ub=b, bounds=[(0, None)] * n,
                    method="highs")
        if r.success and r.fun is not None and np.isfinite(r.fun):
            out[g] = r.fun
    return out


def isoquant_range(X, y, ystar, i1, i2, ref=None, nirs=True, tol=1e-3):
    """Where is the isoquant actually INFORMATIVE?  Returns (lo, hi) on axis i1.

    Two things bound it, and neither is a quantile of the data:

      lo  the smallest x1 for which the LP is feasible at all.  Below it that
          much of input i1 cannot produce ystar however much of i2 is used, so
          the isoquant has a hard left edge -- an asymptote, not a missing value.
      hi  the smallest x1 at which the i2 requirement has flattened to its
          minimum.  Past it the i1 constraint no longer binds and the curve is a
          horizontal line carrying no trade-off.

    A grid spanning the data instead of [lo, hi] puts the whole substitution
    band in a couple of pixels.  Measured on 2005 labour-capital at the median
    output, the band runs from the 0.67th to the 1.03rd percentile of labour --
    0.36 percentage points of the distribution.  That narrowness is itself the
    finding: it is the LP saying what corr(ln L, ln K) > 0 already suggested,
    that this panel has almost no substitution margin.
    """
    v = X[:, i1]
    lo_try, hi_try = v.min() * 0.5, v.max()
    # `ref` MUST be the same vector the caller will plot with.  It was omitted
    # here originally, so the range finder silently fell back to the group's OWN
    # median of the held-out inputs while the figure used a pooled one.  For a
    # small region that own-median is tighter, the probe came back infeasible,
    # and the whole year was dropped from the panel even though it plots fine --
    # Loess Plateau lost 1985-1998 that way.
    f = lambda t: np.isfinite(dea_isoquant(X, y, ystar, i1, i2,
                                           np.array([t]), ref=ref, nirs=nirs)[0])
    if not f(hi_try):
        return np.nan, np.nan
    for _ in range(40):                       # bisect to the feasibility edge
        mid = np.sqrt(lo_try * hi_try) if lo_try > 0 else hi_try / 2
        if f(mid):
            hi_try = mid
        else:
            lo_try = mid
        if hi_try / max(lo_try, 1e-12) < 1.001:
            break
    lo = hi_try
    probe = np.exp(np.linspace(np.log(lo), np.log(v.max()), 60))
    z = dea_isoquant(X, y, ystar, i1, i2, probe, ref=ref, nirs=nirs)
    ok = np.isfinite(z)
    if ok.sum() < 2:
        return lo, v.max()
    zmin = np.nanmin(z)
    flat = np.where(ok & (z <= zmin * (1 + tol) + 1e-12))[0]
    hi = probe[flat[0]] if len(flat) else v.max()
    return lo, max(hi, lo * 1.05)


# ---------------------------------------------------------------------------
# Aggregate input index
# ---------------------------------------------------------------------------
def aggregate_index(d, cols, beta):
    """FIXED-WEIGHT GEOMETRIC (Cobb-Douglas) input index over `cols`.

    NOT a Tornqvist index, despite what this function and its callers used to
    say.  A Tornqvist is a CHAINED BILATERAL index between two adjacent periods,
        ln(X_t / X_{t-1}) = SUM_j 0.5 (s_jt + s_j,t-1) ln(x_jt / x_j,t-1),
    with PERIOD-SPECIFIC cost shares averaged across the two periods.  What is
    computed here is
        x_i = exp( SUM_j w_j [ ln x_ij - ln GM_j ] ),   w fixed for every i,
    a cross-sectional level index with ONE constant weight vector (a single BC92
    estimate).  The two coincide only when the shares do not move over time.
    Calling it Tornqvist implied a flexibility the object does not have -- the
    weights cannot respond to the very substitution the figures are about.

    Weights are renormalised to sum to 1 over the SUBSET, so holding an input
    out still yields a homogeneous-of-degree-one aggregate.  Each input is first
    divided by its own sample GEOMETRIC MEAN, so the index is unit-free and
    equals 1 at the average bundle -- without that step, geometric weighting of
    variables on different unit bases has no interpretable scale.

    Only ratios to the sample geometric mean enter, so an arbitrary per-input
    scale factor cancels exactly.  The index needs each input's unit to be the same in every
    county-year -- which the codebook guarantees -- and nothing about its size.
    That is why it is valid even though the land and capital LEVELS do not
    reconcile against output (see observed_cost_shares).
    """
    w = {c: beta[c] for c in cols}
    s = sum(w.values())
    w = {c: v / s for c, v in w.items()}
    lx = 0.0
    for c in cols:
        gm = np.exp(np.log(d[c]).mean())
        lx = lx + w[c] * (np.log(d[c]) - np.log(gm))
    return np.exp(lx), w
