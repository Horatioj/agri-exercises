# -*- coding: utf-8 -*-
"""
DESCRIPTIVE period figures (Sheng et al. 2025 APEP Fig 9 / Fig 10 analogs).

Output Y = real GVP, labour L = Laborday_impute, and ONE second input X at a
time -- capital K (capital_serv_q) or intermediate inputs M (Inter_all_real).
Both are produced by default, because labour-capital alone tells only half the
story in Chinese agriculture: intermediates (fertiliser, seed, fuel, feed) are
the input that actually exploded after 1978, and the K and M panels shift in
visibly different ways.

*** STANDALONE FRONTIER CONSTRUCTION ***
Not connected to the CPS f_t(x,w) machinery in 41_cps_decomposition.py.  No
weather, no Tornqvist scalar, and the reference set is FRESH PER PERIOD (not
sequential/cumulative), because the point of these figures is to show the
inter-period shift rather than accumulate it away.

HOW TO READ THEM
  Fig 9 is a per-worker production function.  Each point is a county-year:
  horizontal = input per labour day (X/L), vertical = output per labour day
  (Y/L).  The curve is the best-practice boundary for that period -- the most
  output any county achieved at that input intensity.  Two things to look for:
    * CURVATURE.  The boundary rises and flattens, i.e. diminishing returns to
      the second input.  This is the production-theory check: a well-measured
      panel should trace a concave frontier, not a straight line or a cloud.
    * SHIFT.  A later period's curve sitting ABOVE an earlier one means more
      output from the same input per worker -- technical progress.  Vertical
      distance between curves at a fixed X/L is the size of that progress; the
      gap between a county's point and its own period's curve is its
      inefficiency.
  Fig 10 is the same technology seen from the input side.  It fixes output at
  y* (sample median) and asks: what is the least capital (or intermediate)
  needed at each level of labour?  Read it as:
    * SHIFT.  An isoquant closer to the ORIGIN is the better technology -- the
      same output from less of both inputs.  So Fig 10 moving IN is the mirror
      image of Fig 9 moving UP; they are two views of one frontier shift.
    * COUNTY PATHS.  The marked lines are three counties' actual (L, X) mixes
      over time, radially scaled to y*.  Moving down-and-left = getting more
      efficient.
    * SHAPE -- AND ITS LIMIT ON THIS DATA.  A textbook isoquant slopes DOWN,
      because the inputs substitute.  That is NOT what this panel shows: among
      counties producing within +-15% of y*, corr(ln L, ln M) = +0.57 and
      corr(ln L, ln K) is likewise positive, and the low-quantile input
      requirement RISES with labour.  At a given output, counties using more
      labour use more of everything -- the dominant cross-county variation is an
      overall efficiency/intensity gradient, not factor-mix substitution.  The
      lower-left boundary is therefore near-degenerate and comes out flat.  This
      is a property of the data, not of the construction (the exact LP minimum
      is worse: it is set by one freak observation per period and orders the
      periods BACKWARDS against Fig 9).  Sheng et al.'s Figure 10 works because
      18 OECD countries genuinely differ in factor mix; 1,832 Chinese counties
      in the same year mostly do not.  Treat Fig 9 as the reliable descriptive
      frontier and Fig 10's vertical ordering, not its slope, as informative.

Fig 9 analog  scatter of (X/L, Y/L) by year, with a per-year order-alpha
              quantile frontier (see quantile_frontier)
              f(x) = max SUM lam_i y_i  s.t.  x >= SUM lam_i x_i, SUM lam_i <= 1
              evaluated on a grid of x and joined piecewise-linearly.
Fig 10 analog per-period isoquant in (L, X) INPUT LEVELS at a reference output
              y* = sample median real GVP, i.e.
                 min SUM lam_i X_i  s.t. SUM lam_i Y_i >= y*,
                                         SUM lam_i L_i <= L, SUM lam_i <= 1
              plus the OBSERVED input paths of candidate counties.

Periods: 5-period Kalirajan et al. (as cited in Gong 2018 JDE) and a 6-period
variant splitting the last at 2004 (Zhang & Bruemmer 2011; national phase-out
of agricultural taxes).  Data start in 1981, so period 1 is 1981-84.

SAMPLE: the cropland>=15% agricultural counties of the cleaned panel.  (This
read the weather-joined panel until it was noticed that no weather variable is
used here at all -- which silently dropped every county lacking a weather join.)

ISOQUANT vs EXPANSION PATH -- two different objects, do not conflate them:
  * An ISOQUANT (Fig 10) fixes OUTPUT and varies the mix.  It slopes DOWN and
    exists only if the inputs substitute.  On this panel they do not (see the
    Fig 10 note), so it is near-degenerate.
  * An EXPANSION PATH (Fig 11) varies OUTPUT and reads off the bundle.  It
    slopes UP and is the input-space analogue of an Engel curve.  It needs no
    substitution, so it is well identified here and is the more informative
    descriptive figure for this data.

OUTPUTS  fig9_frontier_Y_{KL,ML}.png     per-worker production frontier
         fig10_isoquant_Y_{KL,ML}.png    unit isoquant (weak -- see the note)
         fig11_expansion_Y_{KL,ML}.png   input expansion path
         plus CSVs of every plotted series, so the plots can be rebuilt in the
         paper's own style.
Run:  python src/53_period_frontier_figs.py
      python src/53_period_frontier_figs.py --input K --years 1986,2000,2015
      python src/53_period_frontier_figs.py --alpha 1.0    # raw DEA envelope
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C
import _tfp as T

C.set_cjk_font(plt)          # Chinese county names in labels

OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)

# Captions live OUTSIDE the figures.  A caption block pasted into the axes eats
# plot area, cannot be copy-edited, and is unreadable at publication size; the
# text belongs in the manuscript.  Each figure registers its caption here and
# they are printed and written to captions_fig9_10_11.md at the end.
CAPTIONS = {}

# Fig 9 and Fig 10 are drawn for SINGLE YEARS, not multi-year periods, so the
# two figures describe exactly the same technology and the same counties.
# Averaging over a period mixes technologies and inflates n unevenly across
# periods (1981-84 has 494-824 counties, 1998-2016 has ~1,800), which made the
# earlier period version compare frontiers built on very different samples.
#
# Default years: 1986 / 1995 / 2000 / 2005 / 2010 / 2015 -- six cross-sections
# five years apart, so the frontier shift is read as a SEQUENCE rather than
# inferred from three widely spaced snapshots.  County coverage is within ~6%
# across all six (1,836-1,874 on the current sample), so differences between the
# curves are technology, not coverage.  1981-84 are excluded: only 494-824
# counties report all five I-O variables, less than half the later years.
# (Sheng et al. use three years; more are shown here because the panel supports
# it and the intermediate years are where the shift actually turns.)
DEFAULT_YEARS = [1986, 1995, 2000, 2005, 2010, 2015]
# GVP county cross-section is NOT independently measured in these years (see the
# data-quality note printed at run time): 1982 repeats the 1981 shares and
# 2013-2016 repeat one frozen cross-section, each rescaled by a provincial index.
LOCKSTEP_YEARS = [1982, 2013, 2014, 2015, 2016]

# The second input, one figure at a time.  key -> (column, short, axis label)
INPUTS = {
    "K": ("capital_serv_q", "K", "capital service"),
    "M": ("Inter_all_real", "M", "intermediate inputs (real)"),
}


def trim_ref(df, col, q):
    """Drop the top q of the reference set on `col` before building a frontier.

    The DEA frontier here is set by a SINGLE extreme observation: max Y/K per
    period is 28-52 against a median of 0.05-0.17, so the raw unit isoquant sits
    ~100x below the observed cloud and its period ordering is dictated by which
    period happened to contain the most extreme point.  Trimming makes the
    descriptive figure readable; it is a presentation choice, reported, not a
    change to the estimator used anywhere else in the project."""
    if q <= 0 or len(df) < 50:
        return df
    return df[df[col] <= df[col].quantile(1 - q)]


def load():
    """Agricultural sample of the cleaned panel, with both second inputs."""
    d = T.load_panel()                      # ag_only, all five I-O vars > 0
    d = d.rename(columns={"real_gvp": "Y", "Laborday_impute": "L",
                          "capital_serv_q": "K", "Inter_all_real": "M"})
    d["YL"] = d.Y / d.L
    d["KL"] = d.K / d.L
    d["ML"] = d.M / d.L
    return d


def label(p):
    return str(p[0]) if p[0] == p[1] else f"{p[0]}–{p[1]}"


# --------------------------------------------------------------- Fig 9 frontier
def frontier_ratio(k, y, grid):
    """f(k)=max SUM lam y s.t. SUM lam k <= k, SUM lam <=1, lam>=0.
    Solved as the 2-variable DUAL: min k*u + v s.t. k_i u + v >= y_i, u,v>=0.
    Reference points are first pruned to the vertices of conv({0} u {(k,y)}),
    which is exact (the optimum is attained at a vertex)."""
    P = np.column_stack([k, y])
    if len(P) > 4:
        s = np.abs(P).max(axis=0); s[s == 0] = 1
        try:
            v = np.unique(ConvexHull(np.vstack([P / s, np.zeros(2)]),
                                     qhull_options="Qx").vertices)
            v = v[v < len(P)]
            k, y = k[v], y[v]
        except Exception:
            pass
    A = -np.column_stack([k, np.ones_like(k)])
    b = -y
    out = []
    for kk in grid:
        r = linprog([kk, 1.0], A_ub=A, b_ub=b,
                    bounds=[(0, None), (0, None)], method="highs")
        out.append(r.fun if r.success else np.nan)
    return np.array(out)


def quantile_frontier(x, y, grid, alpha=0.95, nbins=24, minobs=25):
    """Order-alpha (partial) frontier: the ALPHA-quantile of y within bins of x,
    then the concave free-disposal envelope of those bin points.

    WHY NOT THE RAW DEA ENVELOPE.  With ~59,000 county-years spanning four orders
    of magnitude in x, the full-envelope frontier is set by a handful of extreme
    observations and comes out essentially FLAT -- it reports the single most
    extreme county, not the technology.  On this panel the per-decile MAX of Y/L
    jumps around by a factor of 50 (0.13 to 6.2 within one period) while the
    per-decile 95th PERCENTILE rises smoothly and monotonically (0.024 to 0.31).
    The curvature is real; the max envelope simply cannot see it.

    The alpha-quantile frontier is the standard fix (Aragon, Daouia &
    Thomas-Agnan 2005; Daouia & Simar 2007): instead of the absolute maximum it
    takes the level exceeded by only (1-alpha) of units at that input intensity,
    which is consistent, robust to outliers, and converges to the true frontier
    as alpha -> 1.  Pass alpha=1.0 for the raw full-envelope DEA frontier.

    Bins are equal-count (quantile) bins, so every bin carries the same weight
    regardless of how the observations pile up on the log axis.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if alpha >= 1.0 or len(x) < nbins * minobs:
        return frontier_ratio(x, y, grid)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    bx, by = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < minobs:
            continue
        bx.append(np.median(x[m]))
        by.append(np.quantile(y[m], alpha))
    if len(bx) < 3:
        return frontier_ratio(x, y, grid)
    # concave, non-decreasing envelope of the bin quantiles (free disposal in x)
    return frontier_ratio(np.array(bx), np.array(by), grid)


# -------------------------------------------------------------- Fig 10 isoquant
def isoquant_LX(L, X, Y, ystar, Lgrid):
    """min SUM lam X s.t. SUM lam Y >= ystar, SUM lam L <= L, SUM lam <= 1."""
    n = len(L)
    A_ub = np.vstack([-Y, L, np.ones(n)])           # -SUM lam Y <= -ystar ; SUM lam L <= L ; SUM lam <= 1
    out = []
    for LL in Lgrid:
        r = linprog(X, A_ub=A_ub, b_ub=[-ystar, LL, 1.0],
                    bounds=[(0, None)] * n, method="highs")
        out.append(r.fun if r.success else np.nan)
    return np.array(out)


def robust_isoquant(L, X, Y, ystar, Lgrid, alpha=0.05, nbins=20, minobs=25):
    """Farrell isoquant at output y*, built from a LOW quantile rather than the
    minimum.

    CONSTRUCTION
      1. Radially scale every county-year to the reference output:
             (L_i, X_i) * (y* / Y_i)
         Under the constant returns already imposed by the land-normalised
         frontier this puts each observation on the y* input-requirement
         surface, so all periods are directly comparable.
      2. Equal-count bins of scaled L; within each bin take the ALPHA-quantile
         (default 5th percentile) of scaled X -- the robust analogue of "least
         X needed at this L".
      3. Enforce free disposability of labour (the requirement cannot rise as L
         rises: running minimum left to right), then take the lower CONVEX hull
         so the boundary is convex, as an isoquant must be.

    WHY NOT THE EXACT LP MINIMUM.  `min SUM lam X s.t. SUM lam Y >= y*` is the
    textbook isoquant, but on this panel it is decided by ONE county-year with a
    freak X/Y ratio, which makes every period's isoquant a flat line at that
    single county's level -- and, worse, ORDERS THE PERIODS BACKWARDS relative to
    the Fig 9 frontier (1981-84 appeared to need less input than 1998-2016)
    purely because the early cross-section happened to contain a more extreme
    outlier.  The quantile boundary restores the ordering and matches the
    order-alpha frontier used for Fig 9, so the two figures are two views of the
    SAME robust technology rather than two different objects.
    """
    L = np.asarray(L, float); X = np.asarray(X, float); Y = np.asarray(Y, float)
    ok = np.isfinite(L) & np.isfinite(X) & np.isfinite(Y) & (L > 0) & (X > 0) & (Y > 0)
    L, X, Y = L[ok], X[ok], Y[ok]
    f = ystar / Y                                   # radial scaling to y*
    Ls, Xs = L * f, X * f
    if len(Ls) < nbins * minobs:
        return np.full(len(Lgrid), np.nan)

    edges = np.unique(np.quantile(Ls, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(edges, Ls, side="right") - 1, 0, len(edges) - 2)
    bl, bx = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < minobs:
            continue
        bl.append(np.median(Ls[m]))
        bx.append(np.quantile(Xs[m], alpha))
    if len(bl) < 3:
        return np.full(len(Lgrid), np.nan)
    bl, bx = np.array(bl), np.array(bx)
    o = np.argsort(bl); bl, bx = bl[o], bx[o]
    bx = np.minimum.accumulate(bx)                  # free disposability of labour

    # lower convex hull of (ln L, ln X): keep points where the slope increases
    ll, lx = np.log(bl), np.log(bx)
    keep = [0]
    for i in range(1, len(ll)):
        while len(keep) >= 2:
            a_, b_ = keep[-2], keep[-1]
            s1 = (lx[b_] - lx[a_]) / (ll[b_] - ll[a_])
            s2 = (lx[i] - lx[b_]) / (ll[i] - ll[b_])
            if s2 <= s1:            # b_ is above the chord -> drop it
                keep.pop()
            else:
                break
        keep.append(i)
    hl, hx = ll[keep], lx[keep]
    out = np.interp(np.log(Lgrid), hl, hx, left=np.nan, right=np.nan)
    return np.exp(out)


def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def pareto_isoquant(L, X, Y, ystar, trim=0.01):
    """Unit isoquant as the LOWER-LEFT ENVELOPE of the y*-scaled cloud.

    THIS IS THE SAME OBJECT 52_frontier_isoquant_figs.py FIGURE B DRAWS, and the
    two now agree by construction.  Figure B works in output-normalised ratios
    (L/Y, X/Y); this works in (L, X) radially scaled to y*, i.e.
        (L*y*/Y, X*y*/Y) = y* * (L/Y, X/Y)
    -- the SAME cloud up to the constant factor y*.  Only the construction had
    differed, and that is what made the two figures disagree.

    WHY THE PREVIOUS CONSTRUCTION FAILED.  It binned by labour, took a 5%
    quantile of X within each bin, then applied `np.minimum.accumulate` to
    impose free disposability of labour.  On this panel the within-bin quantile
    RISES with labour (no factor substitution), and a running minimum over a
    rising sequence returns its first element forever -- so every isoquant
    collapsed to a horizontal line at the leftmost bin's value.  A conditional
    quantile is not a lower-left envelope; the two answer different questions.

    CONSTRUCTION (Farrell; Kumar & Russell 2002 convention)
      1. radially scale every observation to y*
      2. keep the Pareto-minimal set: scanning by increasing L, keep a point
         only if its X is a new running minimum -- no point lies below-left
      3. take the lower CONVEX hull of that chain
    Step 2 makes the boundary downward-sloping BY CONSTRUCTION, which is what an
    isoquant must be, and it needs no substitution assumption to exist.

    `trim` drops the most extreme X/Y ratios first.  The envelope is otherwise
    decided by a handful of observations (2-5 per year on the output frontier),
    so this is the same robustness concern the order-alpha frontier addresses --
    stated, not hidden.
    """
    L = np.asarray(L, float); X = np.asarray(X, float); Y = np.asarray(Y, float)
    ok = np.isfinite(L) & np.isfinite(X) & np.isfinite(Y) & (L > 0) & (X > 0) & (Y > 0)
    L, X, Y = L[ok], X[ok], Y[ok]
    f = ystar / Y
    Ls, Xs = L * f, X * f
    if trim > 0 and len(Ls) > 50:
        r = Xs / Ls
        keep = (r >= np.quantile(r, trim)) & (r <= np.quantile(r, 1 - trim))
        Ls, Xs = Ls[keep], Xs[keep]
    o = np.argsort(Ls); Ls, Xs = Ls[o], Xs[o]

    chain, best = [], np.inf
    for a_, b_ in zip(Ls, Xs):                 # Pareto-minimal: nothing below-left
        if b_ < best - 1e-15:
            chain.append((a_, b_)); best = b_
    if len(chain) < 3:
        return np.array([c[0] for c in chain]), np.array([c[1] for c in chain])
    H = []                                      # lower convex hull, convex to origin
    for pt in chain:
        while len(H) >= 2 and _cross(H[-2], H[-1], pt) <= 0:
            H.pop()
        H.append(pt)
    return np.array([h[0] for h in H]), np.array([h[1] for h in H])


def expansion_path(s, X, nq=12):
    """EMPIRICAL INPUT EXPANSION PATH: median (L, X) within output deciles.

    This is the input-space analogue of an Engel curve.  An Engel curve traces
    how spending on a good moves with INCOME; an expansion path traces how the
    input bundle moves with OUTPUT.  Both are conditioned on the level of the
    scale variable, and both slope upward -- so, unlike an isoquant, an
    expansion path does NOT require the two inputs to substitute.  That is why
    it works on this panel and the isoquant does not.

    Constructed non-parametrically: equal-count bins of output, then the median
    L and median X within each bin, joined in increasing output order.  Medians
    rather than means because both inputs are strongly right-skewed in levels.

    NOT a cost-minimising expansion path.  The textbook object is the locus of
    cost-minimising bundles at FIXED relative factor prices; we have no county
    factor prices, so this is the observed central tendency, which also embeds
    whatever price variation and inefficiency exist across counties.
    """
    q = pd.qcut(s.Y, nq, labels=False, duplicates="drop")
    g = (s.assign(_q=q).groupby("_q")
           .agg(Y=("Y", "median"), L=("L", "median"), X=(X, "median"),
                n=("Y", "size")).reset_index())
    return g


def pick_candidates(d):
    """Three candidate 'representative' counties under different criteria."""
    n = d.groupby("countyid").year.nunique()
    full = set(n[n == n.max()].index)
    f = d[d.countyid.isin(full)]
    nm = f.groupby("countyid").county_name.last()
    sid = f.groupby("countyid").SID.last()
    g = f.pivot_table(index="countyid", columns="year", values="Y")
    # growth measured to 2012: 2013-2016 carry no genuine county cross-section
    gr = np.log(g[2012] / g[1985]) / (2012 - 1985)
    med = gr.median()
    cand = []
    a = (gr - med).abs().idxmin()
    cand.append(("a_median_growth", a, f"complete panel, 1985-2012 output growth "
                                       f"closest to sample median ({gr[a]:.4f} vs {med:.4f}/yr)"))
    gb = f[f.SID.isin([23, 41])].groupby("countyid").Y.mean()
    b = gb.idxmax()
    cand.append(("b_grain_province", b, f"largest mean output among complete "
                                        f"Heilongjiang/Henan counties ({gb[b]:,.0f})"))
    mk = f.groupby("year").KL.median(); my = f.groupby("year").YL.median()
    t = f.assign(dk=np.log(f.KL) - f.year.map(np.log(mk)),
                 dy=np.log(f.YL) - f.year.map(np.log(my)))
    dist = t.groupby("countyid").apply(lambda s: np.sqrt((s.dk ** 2 + s.dy ** 2).mean()))
    c = dist.idxmin()
    cand.append(("c_typical_path", c, f"minimum RMS log-deviation from the annual "
                                      f"median (K/L, Y/L) path (d={dist[c]:.3f})"))
    return [(tag, int(cid), str(nm[cid]), int(sid[cid]), why) for tag, cid, why in cand]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)),
                    help="comma-separated years for Fig 9/10/11")
    ap.add_argument("--input", choices=["K", "M", "both"], default="both",
                    help="second input on the X axis: capital, intermediates, or both")
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="order-alpha quantile frontier for Fig 9 (1.0 = raw full-"
                         "envelope DEA, which this panel is too heavy-tailed for)")
    ap.add_argument("--iso-alpha", type=float, default=0.05,
                    help="lower quantile defining the Fig 10 isoquant boundary")
    ap.add_argument("--paths", choices=["none", "period", "annual"], default="period",
                    help="county input paths on Fig 10: period means (readable), "
                         "every year (noisy), or omit")
    ap.add_argument("--trim", type=float, default=0.05,
                    help="drop this top share of the reference set (Y/L for fig9, "
                         "Y/X for fig10) before building each period frontier; "
                         "0 = raw DEA")
    a = ap.parse_args()
    d = load()
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} agricultural counties, "
          f"{d.year.min()}-{d.year.max()}")

    print("\n" + "=" * 78)
    print("DATA-QUALITY NOTE affecting the OUTPUT axis of these figures")
    print("  County GVP has no independently measured cross-section in "
          f"{LOCKSTEP_YEARS}:")
    print("  1982 reproduces the 1981 county shares exactly, and 2013-2016 all")
    print("  reproduce ONE frozen cross-section, each rescaled by a provincial")
    print("  index (max |share_t - share_2013| ~ 1e-8).  Labour, capital and")
    print("  intermediates are NOT affected.  So within-period Y/L dispersion in")
    print("  the final period is partly mechanical; county growth to 2016 is a")
    print("  provincial index.  Candidate selection below measures growth to 2012.")
    print("=" * 78)

    cands = pick_candidates(d)
    print("\nCANDIDATE REPRESENTATIVE COUNTIES (pick one for the final figure):")
    for tag, cid, name, sid, why in cands:
        print(f"  [{tag}] {cid} {name} (SID {sid})\n        {why}")
    pd.DataFrame(cands, columns=["tag", "countyid", "name", "SID", "criterion"]) \
      .to_csv(os.path.join(OUT, "county_candidates.csv"), index=False)

    YEARS = [int(y) for y in a.years.split(",")]
    missing = [y for y in YEARS if y not in set(d.year)]
    if missing:
        raise SystemExit(f"years not in the panel: {missing}")
    ny = d[d.year.isin(YEARS)].groupby("year").countyid.nunique()
    print("\nYEARS for Fig 9/10/11 and their county counts:")
    for y in YEARS:
        print(f"  {y}: {ny[y]:,d} counties")
    spread = 100 * (ny.max() - ny.min()) / ny.min()
    print(f"  max-min spread {spread:.1f}%  "
          f"({'comparable' if spread < 10 else 'NOT comparable -- pick closer years'})")
    schemes = [("Y", [(y, y) for y in YEARS])]
    inputs = list(INPUTS) if a.input == "both" else [a.input]

    ystar = d.Y.median()
    for xkey in inputs:
        _, X, xdesc = INPUTS[xkey]
        ratio = f"{X}L"                                  # KL or ML
        for tag, PER in schemes:
            cols = plt.cm.viridis(np.linspace(0, .88, len(PER)))
            stub = f"{tag}_{ratio}"

            # ---------------- Fig 9: per-worker production function -----------
            fig, ax = plt.subplots(figsize=(9.5, 7))
            frows = []
            for (per, col) in zip(PER, cols):
                s = d[d.year.between(*per)]
                ax.scatter(s[ratio], s.YL, s=9, alpha=.24, color=col, linewidths=0,
                           rasterized=True)
            for (per, col) in zip(PER, cols):
                s = d[d.year.between(*per)]
                ref = trim_ref(s, "YL", a.trim)
                # Each period's curve is drawn over ITS OWN observed input range.
                # A common grid would extend every early-period curve rightwards
                # as a flat free-disposability tail into a region where that
                # period has no data, which reads as "no returns to capital"
                # when the truth is "no observations".  The published figure
                # likewise terminates each period's frontier at its own range.
                xlo, xhi = ref[ratio].quantile(.01), ref[ratio].quantile(.99)
                grid = np.exp(np.linspace(np.log(xlo), np.log(xhi), 90))
                fr = quantile_frontier(ref[ratio].values, ref.YL.values, grid,
                                       alpha=a.alpha)
                ax.plot(grid, fr, "-", color=col, lw=2.4,
                        label=f"{label(per)}  (n={len(s):,d})")
                frows.append(pd.DataFrame({"scheme": tag, "input": X,
                                           "period": label(per),
                                           f"x_{ratio}": grid, "f_YL": fr}))
            ax.set_xlim(d[ratio].quantile(.002), d[ratio].quantile(.999))
            ax.set_ylim(d.YL.quantile(.002), d.YL.quantile(.9995))
            pd.concat(frows).to_csv(os.path.join(OUT, f"fig9_frontier_{stub}.csv"),
                                    index=False)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(f"{X}/L  -- {xdesc} per labour day (log)")
            ax.set_ylabel("Y/L  -- real agricultural output per labour day (log)")
            ax.set_title(f"Production frontier, second input = {X}\n"
                         f"order-{chr(945)} quantile frontier ({chr(945)}={a.alpha:g}), "
                         f"separate cross-section per year", fontsize=11)
            ax.legend(fontsize=8, title="year"); ax.grid(alpha=.25, which="both")
            CAPTIONS[f"fig9_frontier_{stub}"] = (
                f"Figure 9 ({X}). Output per labour day against {xdesc} per labour "
                f"day, {', '.join(str(q[0]) for q in PER)}. Source: cleaned county "
                f"agricultural panel, {d.countyid.nunique():,d} agricultural counties, "
                "2005 prices. A SEPARATE cross-sectional frontier is computed for each "
                "year over only the counties observed in that year. Both axes are "
                "logarithmic because output and input per labour day span three or more "
                "orders of magnitude; the frontier itself is computed in levels. The "
                f"frontier is an order-alpha quantile boundary (alpha={a.alpha:g}): the "
                f"alpha-quantile of Y/L within equal-count {X}/L bins, then the concave "
                "free-disposal envelope of those bin points (Aragon, Daouia & "
                "Thomas-Agnan 2005; Daouia & Simar 2007). The raw full-envelope DEA "
                "frontier is not used because on ~59,000 county-years it is set by a "
                "handful of extreme observations and comes out flat. Reading: the curve "
                f"rises then flattens = diminishing returns to {X}/L; a later curve above "
                "an earlier one = technical progress; the vertical gap from a county's "
                "point to its own year's curve is that county's inefficiency. Points are "
                "county-years, coloured by year.")
            fig.tight_layout()
            fig.savefig(os.path.join(C.FIG_DIR, f"fig9_frontier_{stub}.png"), dpi=170)
            plt.close(fig)
            print(f"  saved fig9_frontier_{stub}.png")

            # ---------------- Fig 10: isoquant at y* --------------------------
            fig, ax = plt.subplots(figsize=(9.5, 7))
            # Grid on the y*-SCALED labour axis (the isoquant lives there).
            # POOLED-GRID BUG: this used to be built once from ALL years, but
            # robust_isoquant interpolates onto it and np.interp returns NaN
            # outside the hull's own x-range.  Each year's scaled-L range is far
            # narrower than the pooled one, so most evaluated points came back
            # NaN and the isoquant lines vanished.  The grid is now built inside
            # the period loop from that period's own range.
            _Ls_all = d.L * (ystar / d.Y)          # pooled, for axis limits only
            _iso_vals = []                          # to keep the lines on-axis
            irows = []
            # SUBSTITUTION DIAGNOSTIC.  An isoquant only has a downward slope to
            # trace if, at a GIVEN output, counties really do trade one input for
            # the other.  Here they do not: see the printout below.
            band = d[(d.Y > 0.85 * ystar) & (d.Y < 1.15 * ystar)]
            rho = np.corrcoef(np.log(band.L), np.log(band[X]))[0, 1]
            print(f"    [{X}] substitution check, counties within +-15% of y* "
                  f"(n={len(band):,d}): corr(ln L, ln {X}) = {rho:+.3f}")
            if rho > 0:
                print(f"        POSITIVE -> at the same output, counties using more labour also")
                print(f"        use more {X}.  The variation is an overall efficiency/intensity")
                print(f"        gradient, not factor-mix substitution, so the lower-left boundary")
                print(f"        is near-degenerate and the isoquant is close to flat.  This is a")
                print(f"        property of the data, not of the construction.")
            for (per, col) in zip(PER, cols):
                s = d[d.year.between(*per)]
                f_ = ystar / s.Y
                ax.scatter(s.L * f_, s[X] * f_, s=8, alpha=.20, color=col,
                           linewidths=0, rasterized=True)
            for (per, col) in zip(PER, cols):
                s = d[d.year.between(*per)]
                hx, hy = pareto_isoquant(s.L.values, s[X].values, s.Y.values,
                                         ystar, trim=a.trim)
                if len(hx) < 2:
                    print(f"    [{X}] {label(per)}: envelope has <2 vertices, "
                          f"skipped", file=sys.stderr)
                    continue
                print(f"    [{X}] {label(per)}: envelope from {len(hx)} vertices; "
                      f"slope {'DOWN (ok)' if hy[-1] < hy[0] else 'NOT down (!!)'}"
                      f"  L {hx[0]:.3g}->{hx[-1]:.3g}  {X} {hy[0]:.3g}->{hy[-1]:.3g}")
                _iso_vals.append(hy)
                ax.plot(hx, hy, "-", color=col, lw=2.4, label=f"{label(per)}")
                irows.append(pd.DataFrame({"scheme": tag, "input": X,
                                           "period": label(per),
                                           "L": hx, f"{X}_min": hy}))
            pd.concat(irows).to_csv(os.path.join(OUT, f"fig10_isoquant_{stub}.csv"),
                                    index=False)

            tr_rows = []
            styles = [("o-", "#d62728"), ("s-", "#1f77b4"), ("^-", "#2ca02c")]
            for (tagc, cid, name, sid, _), (mk, cc) in zip(cands, styles):
                if a.paths == "none":
                    continue
                tt = d[d.countyid == cid].sort_values("year").copy()
                # Radially scale each year's (L,X) to the reference output y* so the
                # path is on the same footing as the y* isoquant (Farrell
                # normalisation).  Raw levels are kept in the CSV.
                f_ = ystar / tt.Y
                tt["L_s"], tt["X_s"] = tt.L * f_, tt[X] * f_
                # Year-by-year the scaled path is pure spaghetti -- the radial
                # rescaling amplifies each year's output noise.  Averaging within
                # the SAME periods the isoquants use gives one marker per period,
                # which is what the figure is actually comparing.
                if a.paths == "period":
                    # PERIOD-BUCKET BUG: this used to fall back to PER[-1] for
                    # any year matching no period.  With single-YEAR periods
                    # (1986/2000/2015) that swept ~33 of 36 years into the 2015
                    # bucket, so the "2015" path point was a 30-year pooled
                    # median.  Non-matching years are now DROPPED.
                    _match = [next((q for q in PER if q[0] <= yy <= q[1]), None)
                              for yy in tt.year]
                    tt["_per"] = [label(q) if q is not None else None for q in _match]
                    n_drop = int(tt["_per"].isna().sum())
                    tt = tt[tt["_per"].notna()]
                    if tt.empty:
                        continue
                    pth = (tt.groupby("_per", sort=False)
                             .agg(L_s=("L_s", "median"), X_s=("X_s", "median"),
                                  y0=("year", "min"), y1=("year", "max"))
                             .reset_index())
                else:
                    pth = tt.rename(columns={"year": "y0"}).assign(y1=tt.year)
                ax.plot(pth.L_s, pth.X_s, mk, color=cc, ms=5.0, lw=1.4, alpha=.9,
                        label=f"{name} ({cid}) observed path")
                ax.annotate(str(int(pth.y0.iloc[0])), (pth.L_s.iloc[0], pth.X_s.iloc[0]),
                            fontsize=7.5, color=cc)
                ax.annotate(str(int(pth.y1.iloc[-1])), (pth.L_s.iloc[-1], pth.X_s.iloc[-1]),
                            fontsize=7.5, color=cc, fontweight="bold")
                tr_rows.append(tt.assign(tag=tagc, input=X)
                                 .rename(columns={X: "X"})[["tag", "input", "countyid",
                                                            "county_name", "year",
                                                            "L", "X", "Y", "L_s", "X_s"]])
            pd.concat(tr_rows).to_csv(os.path.join(OUT, f"fig10_county_paths_{stub}.csv"),
                                      index=False)
            _Xs = d[X] * (ystar / d.Y)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(_Ls_all.quantile(.01), _Ls_all.quantile(.99))
            # The isoquant is a LOW quantile within labour bins, so it can sit
            # below the pooled cloud's 1st percentile.  Setting the limits from
            # the cloud alone pushed whole periods off-axis: all three lines were
            # finite and drawn, but only 2015 fell inside the view.  The range
            # must span the drawn lines as well as the cloud.
            _lo, _hi = _Xs.quantile(.01), _Xs.quantile(.99)
            if _iso_vals:
                _iv = np.concatenate(_iso_vals)
                _lo = min(_lo, float(_iv.min()) * 0.75)
                _hi = max(_hi, float(_iv.max()) * 1.35)
            ax.set_ylim(_lo, _hi)
            ax.set_xlabel("L -- labour (man-days, log); all points scaled to y*")
            ax.set_ylabel(f"{X} -- {xdesc} (log)")
            ax.set_title(f"Unit isoquant at y* = median output, "
                         f"second input = {X}\n"
                         f"y* = {ystar:,.0f} (2005 prices); separate cross-section per year",
                         fontsize=11)
            ax.legend(fontsize=7.5, loc="upper left")
            ax.grid(alpha=.25, which="both")
            CAPTIONS[f"fig10_isoquant_{stub}"] = (
                f"Figure 10 ({X}). Unit isoquant at y* = median real GVP "
                f"({ystar:,.0f}, 2005 prices), {', '.join(str(q[0]) for q in PER)}. "
                "Every county-year is radially scaled to y* (Farrell normalisation) and "
                "a separate boundary is built for each year. The boundary is the "
"LOWER-LEFT ENVELOPE of the y*-scaled cloud: the Pareto-minimal "
                "chain (no observation below-left) and then its lower convex hull, the "
                "Farrell / Kumar & Russell (2002) convention. This is the SAME object "
                "and the SAME construction as Figure B of 52_frontier_isoquant_figs.py, "
                "which works in output-normalised ratios (L/Y, X/Y); the two clouds "
                "differ only by the constant factor y*, so the two figures now agree. "
                f"The most extreme {a.trim:.0%} of X/Y ratios are trimmed first, because "
                "the untrimmed envelope is decided by a handful of observations. Note "
                f"that among counties at the same output corr(ln L, ln {X}) = {rho:+.2f}: "
                "the cross-county variation is largely an efficiency/intensity gradient "
                "rather than factor-mix substitution, so the envelope is driven by the "
                "extreme lower-left counties rather than by a broad substitution margin. The marked lines are "
                "three counties' OBSERVED input paths (empirical expansion paths), not "
                "cost-minimising paths -- that would require county factor prices, which "
                "do not exist here; they are radially scaled to y* and raw levels are in "
                f"the CSV. The output cross-section is provincially rescaled in "
                f"{LOCKSTEP_YEARS}, so within-year dispersion in those years is partly "
                "mechanical.")
            fig.tight_layout()
            fig.savefig(os.path.join(C.FIG_DIR, f"fig10_isoquant_{stub}.png"), dpi=170)
            plt.close(fig)
            print(f"  saved fig10_isoquant_{stub}.png")

            # ---------------- Fig 11: input expansion path --------------------
            fig, ax = plt.subplots(figsize=(9.5, 7))
            erows = []
            for (per, col) in zip(PER, cols):
                s = d[d.year.between(*per)]
                e = expansion_path(s, X)
                ax.plot(e.L, e.X, "o-", color=col, lw=2.2, ms=5,
                        label=f"{label(per)}  (n={len(s):,d})")
                ax.annotate("low Y", (e.L.iloc[0], e.X.iloc[0]), fontsize=7,
                            color=col, xytext=(3, -9), textcoords="offset points")
                ax.annotate("high Y", (e.L.iloc[-1], e.X.iloc[-1]), fontsize=7,
                            color=col, fontweight="bold",
                            xytext=(3, 4), textcoords="offset points")
                erows.append(e.assign(scheme=tag, input=X, period=label(per)))
            pd.concat(erows).to_csv(os.path.join(OUT, f"fig11_expansion_{stub}.csv"),
                                    index=False)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("L -- labour (man-days, log), median within output decile")
            ax.set_ylabel(f"{X} -- {xdesc} (log), median within output decile")
            ax.set_title(f"Input expansion path, second input = {X}\n"
                         "median (L, {0}) within output deciles -- the input-space "
                         "analogue of an Engel curve".format(X), fontsize=11)
            ax.legend(fontsize=8, title="year"); ax.grid(alpha=.25, which="both")
            CAPTIONS[f"fig11_expansion_{stub}"] = (
                f"Figure 11 ({X}). Input expansion path: median labour and median "
                f"{xdesc} within output deciles, {', '.join(str(q[0]) for q in PER)}. "
                "This is the input-space analogue of an Engel curve -- an Engel curve "
                "traces spending against income, an expansion path traces the input "
                "bundle against output. Unlike an isoquant it does not require the "
                "inputs to substitute, which is why it is well identified on this panel "
                "and Figure 10 is not. Each marker is one output decile, so moving along "
                "a curve means larger counties. The SLOPE says how the input mix changes "
                f"with scale: a slope above 1 on these log axes means {xdesc} grows "
                "faster than labour as output grows. The SHIFT between years is the same "
                "output decile using a different bundle; down-and-left over time means "
                "less of both per unit output. Medians rather than means because both "
                "inputs are strongly right-skewed in levels. NOT a cost-minimising "
                "expansion path: no county factor prices exist.")
            fig.tight_layout()
            fig.savefig(os.path.join(C.FIG_DIR, f"fig11_expansion_{stub}.png"), dpi=170)
            plt.close(fig)
            print(f"  saved fig11_expansion_{stub}.png")

    print(f"\nCSVs -> {OUT}")

    # ---------------------------------------------------------- captions
    cap_md = os.path.join(OUT, "captions_fig9_10_11.md")
    with open(cap_md, "w", encoding="utf-8") as fh:
        fh.write("# Figure captions - Fig 9 / 10 / 11\n\n")
        fh.write(f"Years: {', '.join(map(str, YEARS))}. "
                 f"Sample: {d.countyid.nunique():,d} agricultural counties.\n\n")
        for k in sorted(CAPTIONS):
            fh.write(f"## {k}\n\n{CAPTIONS[k]}\n\n")
    print("\n" + "=" * 78)
    print("FIGURE CAPTIONS (also written to captions_fig9_10_11.md)")
    print("=" * 78)
    for k in sorted(CAPTIONS):
        print(f"\n[{k}]\n{CAPTIONS[k]}")
    print(f"\ncaptions -> {cap_md}")


if __name__ == "__main__":
    main()
