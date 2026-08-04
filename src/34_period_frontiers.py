# -*- coding: utf-8 -*-
"""
Descriptive period frontiers (OECD Figure 9 / Figure 10 analogs).

*** STANDALONE.  This does NOT touch or reuse the f_t(x,w) machinery in
*** 33_cps_decomposition.py.  Different technology object entirely: two RAW
*** inputs (labour, capital), NO weather, and a FRESH reference set per period
*** (deliberately NOT sequential/cumulative -- the point of the figure is to
*** show the inter-period shift, which a cumulative frontier would absorb).

FIG 9 ANALOG -- production frontier in (k, y) = (K/L, Y/L)
    f(k) = max SUM lam_i y_i  s.t.  k >= SUM lam_i k_i,  SUM lam_i <= 1,  lam >= 0
  With ONE input and ONE output this has a closed form and needs no LP: the
  attainable set is conv({0} u {(k_i,y_i)}), so f is the UPPER CONCAVE ENVELOPE
  of the period's points together with the origin, made non-decreasing (free
  disposability of k flattens it after the peak).  Computed by monotone chain.

FIG 10 ANALOG -- isoquants, two views
  (A) Farrell UNIT isoquant in (L/y, K/y): the lower-left convex hull of the
      per-unit-output input points.  Scale-free and directly comparable across
      periods, which is what makes "unit isoquant" well defined -- but note it
      imposes CONSTANT returns, since dividing by y is only innocuous under CRS.
  (B) isoquant at y* = sample median output, in (L, K) LEVELS, so the observed
      county paths can be overlaid in the same units.  This one needs a small
      LP per grid point (min capital subject to a labour cap and y >= y*),
      solved fresh here -- again, unrelated to the CPS reference sets.

  OBSERVED INPUT PATH (not an expansion path):  county trajectories are the
  ACTUAL (L, K) a county used, joined chronologically.  This is descriptive.
  It is NOT a cost-minimising expansion path: that would require county factor
  prices and a cost-minimisation problem, and we have neither.  Labelled as
  such on the figure.

PERIODS
  5-period (Kalirajan et al., via Gong 2018 JDE), truncated to the 1981 start:
      1981-84, 1985-89, 1990-93, 1994-97, 1998-2016
  6-period variant splits the last at 2004 (Zhang & Bruemmer 2011; national
  phase-out of the agricultural tax):
      ... 1998-2003, 2004-2016

OUTPUTS (src/clean/periods/ and src/figures/)
  fig_period_frontier_{p5,p6}.png        Figure 9 analog
  fig_period_isoquant_{p5,p6}.png        Figure 10 analog + county paths
  frontier_points_{p5,p6}.csv            frontier vertices (k, y) per period
  unit_isoquant_{p5,p6}.csv              Farrell unit-isoquant vertices
  level_isoquant_{p5,p6}.csv             (L,K) isoquant at median output
  county_paths.csv                       candidate county trajectories
  county_candidates.csv                  the 3 candidates and why
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C

OUT = os.path.join(C.CLEAN_DIR, "periods")
os.makedirs(OUT, exist_ok=True)

L_VAR, K_VAR, Y_VAR = "Laborday_impute", "capital_serv_q", "real_gvp"
P5 = [(1981, 1984), (1985, 1989), (1990, 1993), (1994, 1997), (1998, 2016)]
P6 = [(1981, 1984), (1985, 1989), (1990, 1993), (1994, 1997), (1998, 2003), (2004, 2016)]
COLS = ["#1f4e9c", "#c0392b", "#1a1a1a", "#2e8b57", "#8e44ad", "#d68910"]
MK = ["o", "^", "o", "s", "D", "v"]


def plab(p):
    return f"{p[0]}–{p[1]}"


def upper_concave_envelope(k, y):
    """Vertices of the upper concave hull of {(0,0)} u {(k,y)}, non-decreasing.

    Monotone chain over points sorted by k; keeps only left turns so the chain
    is concave; then a running max flattens it (free disposability of input).
    """
    pts = sorted(set(zip(np.r_[0.0, k].tolist(), np.r_[0.0, y].tolist())))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop hull[-1] if it lies on/below the segment hull[-2] -> p
            if (y2 - y1) * (p[0] - x1) - (p[1] - y1) * (x2 - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx = np.array([h[0] for h in hull]); hy = np.array([h[1] for h in hull])
    hy = np.maximum.accumulate(hy)                 # non-decreasing
    return hx, hy


def farrell_unit_isoquant(l, k):
    """Lower-left convex hull of per-unit-output input points (l, k)."""
    pts = sorted(set(zip(l.tolist(), k.tolist())))
    # Pareto-minimal set first: keep points not dominated in BOTH coordinates
    par, best = [], np.inf
    for x, yv in pts:
        if yv < best - 1e-15:
            par.append((x, yv)); best = yv
    if len(par) < 3:
        return np.array([p[0] for p in par]), np.array([p[1] for p in par])
    hull = []                                      # lower convex chain
    for p in par:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) - (p[1] - y1) * (x2 - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return np.array([h[0] for h in hull]), np.array([h[1] for h in hull])


def level_isoquant(L, K, Y, ystar, ngrid=45):
    """min K s.t. SUM lam L_i <= L, SUM lam Y_i >= ystar, SUM lam <= 1, over a grid of L."""
    lo = np.percentile(L, 1); hi = np.percentile(L, 99)
    grid = np.linspace(lo, hi, ngrid)
    A_ub = np.vstack([L, -Y, np.ones_like(L)])     # rows: labour, -output, sum-lambda
    out = []
    for Lb in grid:
        b_ub = np.array([Lb, -ystar, 1.0])
        r = linprog(K, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * len(K), method="highs")
        if r.success:
            out.append((Lb, float(r.fun)))
    if not out:
        return np.array([]), np.array([])
    a = np.array(out)
    keep = a[:, 1] > 0
    return a[keep, 0], a[keep, 1]


def pick_counties(d):
    """Three candidate 'representative' counties under different criteria."""
    full = d.groupby("countyid").year.nunique()
    complete = set(full[full == d.year.nunique()].index)
    g = d[d.countyid.isin(complete)]
    cand = []

    # (a) complete panel, output growth closest to the median growth
    gr = (g.sort_values("year").groupby("countyid")
            .apply(lambda s: np.log(s[Y_VAR].iloc[-1] / s[Y_VAR].iloc[0]) /
                             (s.year.iloc[-1] - s.year.iloc[0])))
    med = gr.median()
    a_id = (gr - med).abs().idxmin()
    cand.append(("a_median_growth", a_id,
                 f"complete panel; growth {gr[a_id]*100:.2f}%/yr vs median {med*100:.2f}%/yr"))

    # (b) major grain county in Heilongjiang (23) or Henan (41), complete, largest output
    sub = g[(g.countyid // 10000).isin([23, 41])]
    if len(sub):
        tot = sub.groupby("countyid")[Y_VAR].mean()
        b_id = tot.idxmax()
        cand.append(("b_grain_belt", b_id,
                     f"complete panel, Heilongjiang/Henan, largest mean output "
                     f"({tot[b_id]:,.0f})"))

    # (c) most 'typical': smallest mean distance to the year-by-year median (lnL, lnK)
    gg = g.copy()
    gg["lL"] = np.log(gg[L_VAR]); gg["lK"] = np.log(gg[K_VAR])
    medyr = gg.groupby("year")[["lL", "lK"]].median().rename(
        columns={"lL": "mL", "lK": "mK"})
    gg = gg.join(medyr, on="year")
    gg["dist"] = np.hypot(gg.lL - gg.mL, gg.lK - gg.mK)
    dd = gg.groupby("countyid").dist.mean()
    c_id = dd.idxmin()
    cand.append(("c_typical_trajectory", c_id,
                 f"complete panel; mean distance to year-median (lnL,lnK) = {dd[c_id]:.3f}"))
    return cand


def run(periods, tag, d, names):
    print(f"\n================ {tag}: {len(periods)} periods ================")
    d = d.copy()
    d["period"] = np.nan
    for i, p in enumerate(periods):
        d.loc[d.year.between(*p), "period"] = i
    d = d.dropna(subset=["period"]); d["period"] = d.period.astype(int)

    # ---------------- Figure 9 analog ----------------
    fr_rows = []
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for i, p in enumerate(periods):
        s = d[d.period == i]
        ax.scatter(s.k, s.yl, s=7, alpha=.20, color=COLS[i], marker=MK[i],
                   linewidths=.4, facecolors="none" if i % 2 else COLS[i],
                   label=f"{plab(p)} all", zorder=1)
    for i, p in enumerate(periods):
        s = d[d.period == i]
        hx, hy = upper_concave_envelope(s.k.values, s.yl.values)
        ax.plot(hx, hy, "-", color=COLS[i], lw=2.4, zorder=3, label=f"{plab(p)} frontier")
        fr_rows += [dict(variant=tag, period=plab(p), k=a, y_per_L=b) for a, b in zip(hx, hy)]
        print(f"  {plab(p):10s} n={len(s):6,d}  frontier vertices={len(hx):3d}  "
              f"max y/L={hy.max():.4f}")
    ax.set_xlim(0, np.percentile(d.k, 99.5)); ax.set_ylim(0, np.percentile(d.yl, 99.7))
    ax.set_xlabel("k = K / L   (capital service per man-day)")
    ax.set_ylabel("y = Y / L   (real agricultural GVP per man-day)")
    ax.set_title(f"Empirically constructed production frontiers by period, "
                 f"China counties ({tag})\nfresh (non-cumulative) NIRS frontier per period",
                 fontsize=11)
    ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, f"fig_period_frontier_{tag}.png"), dpi=170)
    plt.close(fig)
    pd.DataFrame(fr_rows).to_csv(os.path.join(OUT, f"frontier_points_{tag}.csv"), index=False)

    # ---------------- Figure 10 analog ----------------
    ystar = d[Y_VAR].median()
    uni_rows, lev_rows = [], []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 7))
    for i, p in enumerate(periods):
        s = d[d.period == i]
        ux, uy = farrell_unit_isoquant((s[L_VAR] / s[Y_VAR]).values,
                                       (s[K_VAR] / s[Y_VAR]).values)
        a1.plot(ux, uy, "-", color=COLS[i], lw=2.2, label=plab(p))
        uni_rows += [dict(variant=tag, period=plab(p), labor_per_y=x, capital_per_y=y2)
                     for x, y2 in zip(ux, uy)]
        lx, lk = level_isoquant(s[L_VAR].values, s[K_VAR].values, s[Y_VAR].values, ystar)
        if len(lx):
            a2.plot(lx / 1e6, lk / 1e6, "-", color=COLS[i], lw=2.2, label=plab(p))
            lev_rows += [dict(variant=tag, period=plab(p), labor=x, capital=y2)
                         for x, y2 in zip(lx, lk)]
        print(f"  {plab(p):10s} unit-isoquant vertices={len(ux):3d}  level pts={len(lx):3d}")
    a1.set_xlabel("labour per unit output  (L / y)")
    a1.set_ylabel("capital per unit output  (K / y)")
    a1.set_title("(A) Farrell unit isoquant by period\n"
                 "scale-free; note dividing by y imposes CONSTANT returns", fontsize=10)
    a1.set_xlim(0, np.percentile(d[L_VAR] / d[Y_VAR], 97))
    a1.set_ylim(0, np.percentile(d[K_VAR] / d[Y_VAR], 97))
    a1.legend(fontsize=8); a1.grid(alpha=.25)

    # observed county paths (levels)
    for j, (why, cid, note) in enumerate(names):
        t = d[d.countyid == cid].sort_values("year")
        if not len(t):
            continue
        a2.plot(t[L_VAR] / 1e6, t[K_VAR] / 1e6, "-", color="0.25", lw=1.1,
                alpha=.85, zorder=5)
        a2.scatter(t[L_VAR] / 1e6, t[K_VAR] / 1e6, s=12, c=t.year, cmap="viridis",
                   zorder=6, label=f"{why} ({cid})")
        a2.annotate(str(int(t.year.iloc[0])), (t[L_VAR].iloc[0] / 1e6, t[K_VAR].iloc[0] / 1e6),
                    fontsize=7, color="0.3")
        a2.annotate(str(int(t.year.iloc[-1])), (t[L_VAR].iloc[-1] / 1e6, t[K_VAR].iloc[-1] / 1e6),
                    fontsize=7, color="0.3", fontweight="bold")
    a2.set_xlabel("labour (million man-days)"); a2.set_ylabel("capital service (millions)")
    a2.set_title(f"(B) Isoquant at y* = median output, in LEVELS\n"
                 f"+ OBSERVED input paths (descriptive, NOT cost-minimising)", fontsize=10)
    a2.legend(fontsize=7.5); a2.grid(alpha=.25)
    fig.suptitle(f"Unit isoquants and observed county input paths ({tag})",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(C.FIG_DIR, f"fig_period_isoquant_{tag}.png"), dpi=170)
    plt.close(fig)
    pd.DataFrame(uni_rows).to_csv(os.path.join(OUT, f"unit_isoquant_{tag}.csv"), index=False)
    pd.DataFrame(lev_rows).to_csv(os.path.join(OUT, f"level_isoquant_{tag}.csv"), index=False)


def main():
    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    d = d.dropna(subset=[L_VAR, K_VAR, Y_VAR])
    d = d[(d[[L_VAR, K_VAR, Y_VAR]] > 0).all(axis=1)].copy()
    d["k"] = d[K_VAR] / d[L_VAR]
    d["yl"] = d[Y_VAR] / d[L_VAR]
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} counties, "
          f"{d.year.min()}-{d.year.max()}")

    cand = pick_counties(d)
    cc = pd.DataFrame([dict(criterion=w, countyid=c, note=n) for w, c, n in cand])
    nm = (d.groupby("countyid").county_name.last()
          if "county_name" in d.columns else pd.Series(dtype=object))
    cc["county_name"] = cc.countyid.map(nm)
    cc.to_csv(os.path.join(OUT, "county_candidates.csv"), index=False)
    print("\ncandidate representative counties:")
    print(cc.to_string(index=False))
    paths = d[d.countyid.isin(cc.countyid)][
        ["countyid", "county_name", "year", L_VAR, K_VAR, Y_VAR, "k", "yl"]]
    paths.to_csv(os.path.join(OUT, "county_paths.csv"), index=False)

    run(P5, "p5", d, cand)
    run(P6, "p6", d, cand)
    print(f"\nsaved figures -> src/figures/, tables -> {OUT}")


if __name__ == "__main__":
    main()
