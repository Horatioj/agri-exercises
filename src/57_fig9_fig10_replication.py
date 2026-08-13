# -*- coding: utf-8 -*-
"""
Replication of Sheng et al. (2025 APEP) Figure 9 and Figure 10 for the Chinese
county agricultural panel, with a PARAMETRIC and a NON-PARAMETRIC frontier drawn
side by side, and a translog specification test.

WHAT THIS ADDS OVER THE EXISTING SCRIPTS
  50_io_surface.py   parametric, but in (Labour x Intermediate) space -- it never
                     plots capital per worker, which is the axis Figure 9 uses.
  52 / 53            non-parametric only, and only three years.
  HERE               k = K/L on the horizontal and y = Y/L on the vertical (the
                     published axes), SEVEN years, and both estimator families on
                     the same axes so they can be compared directly.

ONE YEAR PER FRONTIER.  Every curve is estimated on a single year's
cross-section -- never pooled, never averaged over a period.  Pooling mixes
technologies, which is exactly what a frontier shift figure must not do.

THE TWO ESTIMATOR FAMILIES
  NON-PARAMETRIC   order-alpha quantile frontier: the alpha-quantile of y within
                   equal-count bins of k, then the concave free-disposal
                   envelope.  Assumes nothing about functional form.  (The raw
                   full-envelope DEA hull is also available with --alpha 1.0, but
                   on ~1,900 counties it tracks one extreme observation.)
  PARAMETRIC       quantile REGRESSION at the same tau, in two nested forms:
                     Cobb-Douglas  ln y = a + b ln k
                     translog      ln y = a + b ln k + c (ln k)^2
                   Both are the per-worker form of a two-input CRS technology, so
                   they are directly comparable with the non-parametric curve.

  Using the same tau for both families is the point: they then estimate the SAME
  object (the tau-quantile frontier) under different assumptions, so any gap
  between them is functional-form error rather than a difference in target.

DOES THE DATA HOLD?  That is what the translog term c answers.
  c < 0 and significant  -> the per-worker frontier is CONCAVE: diminishing
                            returns to capital deepening, which is what
                            production theory requires.
  c ~ 0                  -> Cobb-Douglas is adequate (constant elasticity).
  c > 0                  -> convex, which would contradict theory and indicate a
                            measurement or specification problem.
  The printed table reports c, its t-statistic, and the Cobb-Douglas-vs-translog
  comparison for every year, so the specification is tested rather than assumed.

FIGURE 9  fig9_repl_frontier_kl.png
  (A) non-parametric frontier, one curve per year, over the county scatter
  (B) parametric translog frontier, same years, same axes
FIGURE 10 fig10_repl_isoquant_lk.png
  Unit isoquant in (L/Y, K/Y) -- input per unit output, the Farrell convention.
  (A) non-parametric lower-left boundary   (B) parametric, from the translog

  Fig 10 is expected to be NEARLY FLAT on this panel and that is not a defect:
  among counties at the same output corr(ln L, ln K) ~ 0, so there is no
  factor-substitution margin for an isoquant to trace.  It is reported for
  completeness and for its vertical ORDERING; Figure 9 is the informative view.

Input : src/clean/county_panel_clean.csv  (agricultural sample)
Output: src/figures/fig9_repl_frontier_kl.png, fig10_repl_isoquant_lk.png
        src/clean/descriptive/fig9_repl_*.csv, translog_test.csv
Run:  python src/57_fig9_fig10_replication.py
      python src/57_fig9_fig10_replication.py --years 1986,1996,2006,2015 --tau 0.9
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C
import _tfp as T

OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)

DEFAULT_YEARS = [1986, 1991, 1996, 2001, 2006, 2011, 2015]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load():
    d = T.load_panel()
    d = d.rename(columns={"real_gvp": "Y", "Laborday_impute": "L",
                          "capital_serv_q": "K"})
    d["k"] = d.K / d.L          # capital per labour day
    d["y"] = d.Y / d.L          # output per labour day
    d["lk"] = d.L / d.Y         # labour per unit output   (Fig 10 axes)
    d["kk"] = d.K / d.Y         # capital per unit output
    for c in ("k", "y", "lk", "kk"):
        d = d[np.isfinite(d[c]) & (d[c] > 0)]
    return d


# ------------------------------------------------------------ non-parametric
def np_frontier(x, y, grid, tau=0.95, nbins=24, minobs=25):
    """Order-alpha quantile frontier: tau-quantile of y within equal-count bins
    of x, then the concave non-decreasing envelope of those bin points."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    bx, by = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < minobs:
            continue
        bx.append(np.median(x[m])); by.append(np.quantile(y[m], tau))
    if len(bx) < 3:
        return np.full(len(grid), np.nan)
    bx, by = np.array(bx), np.array(by)
    # upper concave envelope through the bin points, then non-decreasing
    o = np.argsort(bx); bx, by = bx[o], by[o]
    H = []
    for p in zip(bx, by):
        while len(H) >= 2:
            (x1, y1), (x2, y2) = H[-2], H[-1]
            if (y2 - y1) * (p[0] - x1) - (p[1] - y1) * (x2 - x1) >= 0:
                H.pop()
            else:
                break
        H.append(p)
    hx = np.array([h[0] for h in H]); hy = np.array([h[1] for h in H])
    hy = np.maximum.accumulate(hy)                # free disposability of k
    return np.interp(grid, hx, hy, left=np.nan, right=hy[-1])


# ---------------------------------------------------------------- parametric
def fit_quantreg(x, y, tau, translog=True):
    """tau-quantile regression of ln y on ln k (+ its square for translog).

    ln k IS CENTRED before squaring:
        ln y = a + b (ln k - m) + c (ln k - m)^2 ,   m = mean ln k
    Without centring the quadratic is evaluated far from zero -- ln k is about
    -2.5 on this panel and never near 0 -- so `b` becomes the slope extrapolated
    back to k = 1, a point outside the data, and b and c trade off against each
    other.  Uncentred, b swung between 0.18 and 1.43 across years and the implied
    elasticity at median k came out NEGATIVE.  Centred, `b` is the output
    elasticity of capital AT THE MEAN k and `c` is the curvature, which is what
    the specification test needs them to mean.  Returns (result, m).
    """
    lx, ly = np.log(x), np.log(y)
    m = float(lx.mean())
    z = lx - m
    X = np.column_stack([z, z ** 2]) if translog else z[:, None]
    X = sm.add_constant(X)
    try:
        r = sm.QuantReg(ly, X).fit(q=tau, max_iter=5000)
    except Exception:
        return None, m
    return r, m


def predict_ln(res, m, grid, translog=True):
    z = np.log(grid) - m
    X = np.column_stack([z, z ** 2]) if translog else z[:, None]
    return np.exp(res.predict(sm.add_constant(X)))


def elasticity(res, m, k):
    """d ln y / d ln k = b + 2c (ln k - m).  Theory wants 0 < e < 1 for a single
    input under diminishing returns."""
    return res.params[1] + 2 * res.params[2] * (np.log(k) - m)


def concave_in_levels(res, m, kgrid):
    """Is the FITTED y(k) concave in LEVELS across the observed range?

    This, not the sign of c, is the production-theory test.  Cobb-Douglas is
    LINEAR in logs (c = 0) yet strictly concave in levels, so c > 0 does not by
    itself mean the frontier violates theory -- it means the elasticity rises
    with k.  Checked numerically on the fitted curve.
    """
    yv = predict_ln(res, m, kgrid)
    d2 = np.diff(yv, 2)
    frac = float(np.mean(d2 < 0))
    return bool(frac > 0.9), frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    ap.add_argument("--tau", type=float, default=0.95,
                    help="frontier quantile, used by BOTH estimator families")
    a = ap.parse_args()
    YEARS = [int(v) for v in a.years.split(",")]

    d = load()
    miss = [y for y in YEARS if y not in set(d.year)]
    if miss:
        raise SystemExit(f"years not in the panel: {miss}")
    ny = d[d.year.isin(YEARS)].groupby("year").countyid.nunique()
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} agricultural counties")
    print("years and county counts (each frontier uses ONE year only):")
    for y in YEARS:
        print(f"  {y}: {ny[y]:,d}")
    print(f"  max-min spread {100*(ny[YEARS].max()-ny[YEARS].min())/ny[YEARS].min():.1f}%")

    cols = plt.cm.viridis(np.linspace(0, .88, len(YEARS)))
    klo, khi = d.k.quantile(.02), d.k.quantile(.98)
    grid = np.exp(np.linspace(np.log(klo), np.log(khi), 120))

    # ---------------- specification test + fits, one year at a time ---------
    rows, fits_tl, fits_cd, npf = [], {}, {}, {}
    for y in YEARS:
        s = d[d.year == y]
        npf[y] = np_frontier(s.k.values, s.y.values, grid, tau=a.tau)
        rt, mt = fit_quantreg(s.k.values, s.y.values, a.tau, translog=True)
        rc, mc = fit_quantreg(s.k.values, s.y.values, a.tau, translog=False)
        fits_tl[y], fits_cd[y] = (rt, mt), (rc, mc)
        if rt is None:
            continue
        b, c, tc = rt.params[1], rt.params[2], rt.tvalues[2]
        kq = s.k.quantile([.1, .5, .9]).values
        e10, e50, e90 = elasticity(rt, mt, kq)
        kg = np.exp(np.linspace(np.log(s.k.quantile(.01)),
                                np.log(s.k.quantile(.99)), 200))
        conc, frac = concave_in_levels(rt, mt, kg)
        rows.append(dict(year=y, n=len(s), b_at_mean=b, c_curv=c, t_c=tc,
                         elas_p10=e10, elas_p50=e50, elas_p90=e90,
                         concave_levels=conc, frac_concave=frac,
                         reject_CD=bool(abs(tc) > 1.96)))
    tst = pd.DataFrame(rows)
    tst.to_csv(os.path.join(OUT, "translog_test.csv"), index=False)

    print(f"\nTRANSLOG SPECIFICATION TEST  (quantile regression at tau={a.tau:g})")
    print("  ln y = a + b ln k + c (ln k)^2      c=0 is the Cobb-Douglas restriction")
    print(f"  {'year':>6} {'n':>6} {'b@mean':>8} {'c':>8} {'t(c)':>7} "
          f"{'e(p10)':>7} {'e(p50)':>7} {'e(p90)':>7}  {'concave':>8} {'rejCD':>6}")
    for r in tst.itertuples():
        print(f"  {r.year:>6} {r.n:>6,d} {r.b_at_mean:>8.3f} {r.c_curv:>8.3f} "
              f"{r.t_c:>7.1f} {r.elas_p10:>7.3f} {r.elas_p50:>7.3f} {r.elas_p90:>7.3f}"
              f"  {'yes' if r.concave_levels else 'NO':>8} {'yes' if r.reject_CD else 'no':>6}")
    n_conc = int(tst.concave_levels.sum()); n_rej = int(tst.reject_CD.sum())
    print(f"\n  fitted y(k) concave IN LEVELS in {n_conc}/{len(tst)} years; "
          f"Cobb-Douglas rejected in {n_rej}/{len(tst)} years")
    print(f"  elasticity at median k: min {tst.elas_p50.min():.3f} "
          f"max {tst.elas_p50.max():.3f}   (theory wants 0 < e < 1)")
    if n_conc == len(tst) and tst.elas_p50.between(0, 1).all():
        print("  => holds: concave in levels every year and 0 < elasticity < 1 at")
        print("     median k, which is what production theory requires.")
    else:
        print("  => does NOT hold in every year -- see the flags above before")
        print("     treating the parametric frontier as a production function.")

    # ---------------- shared-denominator (ratio bias) diagnostic ------------
    # Figure 9's axes are y = Y/L and k = K/L: BOTH divide by the same L, and on
    # this panel L is `Laborday_impute`, an IMPUTED series.  Classical
    # measurement error in L then moves a county along a +1 slope in (ln k, ln y)
    # space, manufacturing a k-y relationship that is arithmetic rather than
    # technology.  The check: estimate the same tau-frontier WITHOUT the shared
    # denominator (ln Y on ln L and ln K) and compare the capital elasticity.  A
    # per-worker elasticity far above the level-form one is the signature.
    print("")
    print("SHARED-DENOMINATOR (RATIO BIAS) CHECK")
    print("  Fig 9 divides BOTH axes by L, and L is imputed on this panel.")
    hdr = "{:>6} {:>15} {:>11} {:>11} {:>7} {:>7}".format(
        "year", "e_K per-worker", "e_K level", "e_L level", "RTS", "ratio")
    print("  " + hdr)
    brows = []
    for y in YEARS:
        s = d[d.year == y]
        lY, lL, lK = np.log(s.Y), np.log(s.L), np.log(s.K)
        try:
            r1 = sm.QuantReg(lY - lL, sm.add_constant((lK - lL).values)).fit(
                q=a.tau, max_iter=5000)
            r2 = sm.QuantReg(lY, sm.add_constant(np.column_stack([lL, lK]))).fit(
                q=a.tau, max_iter=5000)
        except Exception:
            continue
        ew, bL, bK = r1.params[1], r2.params[1], r2.params[2]
        rat = ew / bK if abs(bK) > 1e-6 else float("nan")
        brows.append(dict(year=y, eK_perworker=ew, eK_level=bK, eL_level=bL,
                          rts=bL + bK, inflation=rat))
        print("  {:>6} {:>15.3f} {:>11.3f} {:>11.3f} {:>7.3f} {:>7.1f}".format(
            y, ew, bK, bL, bL + bK, rat))
    bb = pd.DataFrame(brows)
    bb.to_csv(os.path.join(OUT, "ratio_bias_check.csv"), index=False)
    if len(bb) and (bb.inflation.abs() > 3).mean() > 0.5:
        print("  => the per-worker capital elasticity is several times the level-form")
        print("     one in most years.  Read Figure 9's vertical SHIFT (technical")
        print("     progress) as informative and its SLOPE along k with caution: a")
        print("     large part of that slope is the shared L denominator, not")
        print("     capital deepening.")

    # ------------------------------------------------------------- FIGURE 9
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharex=True, sharey=True)
    for ax, (kind, title) in zip(axes, [
            ("np", f"(A)  Non-parametric  —  order-α quantile frontier (α={a.tau:g})"),
            ("tl", f"(B)  Parametric  —  translog quantile regression (τ={a.tau:g})")]):
        for y, col in zip(YEARS, cols):
            s = d[d.year == y]
            ax.scatter(s.k, s.y, s=2.5, alpha=.06, color=col, linewidths=0, zorder=1)
        for y, col in zip(YEARS, cols):
            s = d[d.year == y]
            lo, hi = s.k.quantile(.01), s.k.quantile(.99)
            g_ = grid[(grid >= lo) & (grid <= hi)]     # each year over its OWN range
            if kind == "np":
                v = np_frontier(s.k.values, s.y.values, g_, tau=a.tau)
            else:
                r, m_ = fits_tl[y]
                v = predict_ln(r, m_, g_) if r is not None else np.full(len(g_), np.nan)
            ax.plot(g_, v, "-", color=col, lw=2.3, label=f"{y}  (n={len(s):,d})", zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(klo, khi); ax.set_ylim(d.y.quantile(.01), d.y.quantile(.995))
        ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("k = K / L   capital service per labour day  (log)")
        ax.grid(alpha=.25, which="both")
    axes[0].set_ylabel("y = Y / L   real output per labour day  (log)")
    axes[1].legend(fontsize=8, title="year", loc="lower right")
    fig.suptitle("Figure 9 replication — per-worker production frontier, "
                 "one cross-section per year",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.5, 0.925,
             f"{d.countyid.nunique():,d} agricultural counties;  each curve is estimated on "
             f"that year's observations ONLY and drawn over that year's own k range;  "
             f"both panels target the same τ={a.tau:g} frontier",
             ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p9 = os.path.join(C.FIG_DIR, "fig9_repl_frontier_kl.png")
    fig.savefig(p9, dpi=175, facecolor="white"); plt.close(fig)
    print("\nsaved:", p9)

    frows = []
    for y in YEARS:
        r, m_ = fits_tl[y]
        frows.append(pd.DataFrame({"year": y, "k": grid, "np_frontier": npf[y],
                                   "translog": (predict_ln(r, m_, grid) if r is not None
                                                else np.nan)}))
    pd.concat(frows).to_csv(os.path.join(OUT, "fig9_repl_frontier.csv"), index=False)

    # ------------------------------------------------------------ FIGURE 10
    # Farrell unit isoquant: inputs per unit of output.
    llo, lhi = d.lk.quantile(.02), d.lk.quantile(.98)
    lgrid = np.exp(np.linspace(np.log(llo), np.log(lhi), 120))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharex=True, sharey=True)
    irows = []
    for ax, kind in zip(axes, ["np", "tl"]):
        for y, col in zip(YEARS, cols):
            s = d[d.year == y]
            ax.scatter(s.lk, s.kk, s=2.5, alpha=.06, color=col, linewidths=0, zorder=1)
        for y, col in zip(YEARS, cols):
            s = d[d.year == y]
            lo, hi = s.lk.quantile(.01), s.lk.quantile(.99)
            g_ = lgrid[(lgrid >= lo) & (lgrid <= hi)]
            if kind == "np":
                # lower boundary: 5th percentile of K/Y within bins of L/Y,
                # made non-increasing (free disposability of labour)
                v = np_frontier(s.lk.values, -s.kk.values, g_, tau=1 - 0.05)
                v = -v
            else:
                # parametric: invert the translog per-worker frontier.
                # y = f(k) with k=K/L  =>  at output y*, K/Y and L/Y follow from
                # L/Y = 1/y and K/Y = k/y along the fitted frontier.
                r, m_ = fits_tl[y]
                if r is None:
                    v = np.full(len(g_), np.nan)
                else:
                    kk_ = np.exp(np.linspace(np.log(d.k.quantile(.01)),
                                             np.log(d.k.quantile(.99)), 400))
                    yy_ = predict_ln(r, m_, kk_)
                    lk_, kv_ = 1.0 / yy_, kk_ / yy_
                    o = np.argsort(lk_)
                    v = np.interp(g_, lk_[o], kv_[o], left=np.nan, right=np.nan)
            ax.plot(g_, v, "-", color=col, lw=2.3, label=f"{y}", zorder=4)
            if kind == "np":
                irows.append(pd.DataFrame({"year": y, "L_per_Y": g_, "K_per_Y": v}))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(llo, lhi); ax.set_ylim(d.kk.quantile(.01), d.kk.quantile(.99))
        ax.set_xlabel("L / Y   labour days per unit of real GVP  (log)")
        ax.grid(alpha=.25, which="both")
    axes[0].set_title("(A)  Non-parametric  —  lower-left input-requirement boundary",
                      fontsize=11.5, loc="left", fontweight="bold", color=INK)
    axes[1].set_title("(B)  Parametric  —  implied by the translog frontier",
                      fontsize=11.5, loc="left", fontweight="bold", color=INK)
    axes[0].set_ylabel("K / Y   capital service per unit of real GVP  (log)")
    axes[1].legend(fontsize=8, title="year", loc="upper right")
    band = d[(d.Y > 0.85 * d.Y.median()) & (d.Y < 1.15 * d.Y.median())]
    rho = np.corrcoef(np.log(band.L), np.log(band.K))[0, 1]
    fig.suptitle("Figure 10 replication — unit isoquant, one cross-section per year",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.5, 0.925,
             f"Inputs per unit of output (Farrell).  EXPECT THIS TO BE NEARLY FLAT: among "
             f"counties at the same output corr(ln L, ln K) = {rho:+.2f}, so this panel has "
             f"no factor-substitution\nmargin for an isoquant to trace.  The vertical "
             f"ORDERING across years is meaningful; the slope is not.  Figure 9 is the "
             f"informative view.",
             ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    p10 = os.path.join(C.FIG_DIR, "fig10_repl_isoquant_lk.png")
    fig.savefig(p10, dpi=175, facecolor="white"); plt.close(fig)
    print("saved:", p10)
    if irows:
        pd.concat(irows).to_csv(os.path.join(OUT, "fig10_repl_isoquant.csv"), index=False)
    print(f"\nCSVs -> {OUT}")


if __name__ == "__main__":
    main()
