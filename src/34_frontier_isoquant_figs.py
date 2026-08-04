# -*- coding: utf-8 -*-
"""
Illustrative single-year frontier / isoquant figures (OECD Figure 9 & 10 analogs).

STANDALONE from the CPS sequential decomposition: each year gets its OWN
cross-sectional DEA frontier over the counties observed that year. Raw
disaggregated inputs are used (L, K, land, M) -- the Tornqvist scalar collapses
exactly the two-dimensional structure these figures exist to show.

  Figure A  y = Y/L against k = K/L (and a land/L variant), per-year frontier,
            NIRS + FREE disposability of the input ratio, with the flat
            free-disposability tail beyond the largest observed ratio.
  Figure B  unit isoquant (y = 1) from output-normalised inputs (L/y, K/y),
            lower-left input-requirement boundary, free-disposability
            extensions on both axes.
  Figure C  w = GDD against the Tornqvist input index, output-normalised.
  Figure D  w = 0-28cm soil moisture, same construction.

DISPOSABILITY -- the visually checkable difference between A/B and C/D:
  A and B assume FREE disposability of the conventional inputs, so their
  boundaries carry flat extensions (using more input is always feasible).
  C and D assume WEAK disposability of w -- the equality constraint used in the
  main CPS decomposition -- so weather is NOT freely disposable and the curve
  gets NO flat extension: it terminates at the extreme feasible w. The stubs at
  the ends of the C/D curves are the assumption made visible.

Palette: dataviz categorical slots 1-3 (blue/orange/aqua), validated
(worst adjacent CVD dE 9.2, normal-vision 27.6). Aqua sits below 3:1 contrast,
so every series also carries a direct line-end label as the required relief.

Outputs: src/figures/frontier/*.png and *.pdf, plus captions.md
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import _common as C

OUT = os.path.join(C.FIG_DIR, "frontier")
os.makedirs(OUT, exist_ok=True)

YEARS = [1986, 2000, 2015]
COL = {1986: "#2a78d6", 2000: "#eb6834", 2015: "#1baf7a"}     # categorical 1-3
INK, INK2, GRID = "#1a1a19", "#5c5b55", "#e6e5e0"
BETA = {"Laborday_impute": .439, "Inter_all_real": .436,
        "capital_serv_q": .069, "Land_serv_q": .056}

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def upper_concave_frontier(k, y, x_tail):
    """FIG A boundary: NIRS + free disposability.
    Upper concave hull of {(0,0)} u pts, made non-decreasing, then extended flat
    to x_tail (the free-disposability tail)."""
    d = pd.DataFrame({"k": k, "y": y}).groupby("k", as_index=False).y.max()
    pts = [(0.0, 0.0)] + list(map(tuple, d.sort_values("k").values))
    H = []
    for p in pts:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) >= 0:
            H.pop()
        H.append(p)
    hx = np.array([h[0] for h in H]); hy = np.array([h[1] for h in H])
    j = int(np.argmax(hy))                      # non-decreasing: cut at the peak
    hx, hy = hx[:j+1], hy[:j+1]
    if x_tail > hx[-1]:                          # flat free-disposability tail
        hx = np.append(hx, x_tail); hy = np.append(hy, hy[-1])
    return hx, hy


def lower_convex_isoquant(a, b):
    """FIG B/C/D boundary: minimal-input Pareto set, then its convex lower hull.
    a on the horizontal, b on the vertical. Returns the efficient chain only --
    the caller decides whether to add free-disposability extensions."""
    d = pd.DataFrame({"a": a, "b": b}).sort_values(["a", "b"])
    keep, best = [], np.inf
    for r in d.itertuples():                     # Pareto-minimal (no point below-left)
        if r.b < best - 1e-15:
            keep.append((r.a, r.b)); best = r.b
    if len(keep) < 3:
        q = np.array(keep); return q[:, 0], q[:, 1]
    H = []                                        # lower convex hull, convex to origin
    for p in keep:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) <= 0:
            H.pop()
        H.append(p)
    return np.array([h[0] for h in H]), np.array([h[1] for h in H])


def frame():
    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    need = ["real_gvp", "Laborday_impute", "capital_serv_q", "Land_serv_q",
            "Inter_all_real", "gdd_growing_season", "sm_0_28"]
    d = d.dropna(subset=need)
    d = d[(d[need[:5]] > 0).all(axis=1)].copy()
    s = sum(BETA.values())
    d["xagg"] = np.exp(sum((v/s) * np.log(d[k]) for k, v in BETA.items()))
    return d


def logaxes(ax, xlo, xhi, ylo, yhi):
    """Both variables span orders of magnitude, so the cloud and the DEA
    boundary cannot both be read on linear axes. Frontiers are still computed in
    LEVELS -- only the display is logarithmic."""
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)


def label_at(ax, hx, hy, yr, frac):
    """Direct label placed at a different position per series (contrast relief
    for the aqua slot, and it keeps the three labels from colliding)."""
    m = np.isfinite(hx) & np.isfinite(hy) & (hx > 0) & (hy > 0)
    if not m.any():
        return
    xs, ys = hx[m], hy[m]
    j = min(len(xs) - 1, max(0, int(frac * (len(xs) - 1))))
    ax.annotate(str(yr), (xs[j], ys[j]), fontsize=9, color=INK, fontweight="bold",
                xytext=(5, 5), textcoords="offset points", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def finish(fig, ax, title, caption, fname, legend_loc="lower right"):
    ax.set_title(title, fontsize=11.5, loc="left", pad=10, color=INK)
    lg = ax.legend(loc=legend_loc, frameon=False, fontsize=9,
                   labelcolor=INK, title="Year", title_fontsize=9)
    lg.get_title().set_color(INK2)
    fig.text(.012, .015, caption, fontsize=8, color=INK2, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, .11, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", fname)


# ------------------------------------------------------------------ FIGURE A
def figure_A(d, num, den, dlab, fname, title, hexbin=False):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sub = d[d.year.isin(YEARS)]
    kk = (sub[num] / sub[den]).replace([np.inf, -np.inf], np.nan).dropna()
    yy = (sub.real_gvp / sub[den]).replace([np.inf, -np.inf], np.nan).dropna()
    xlo, xhi = np.nanpercentile(kk, [1, 99.5])
    ylo = np.nanpercentile(yy, 1)
    yhi_pts = np.nanpercentile(yy, 99.5)
    fr_max = 0.0
    for i, yr in enumerate(YEARS):
        g = d[d.year == yr]
        k = (g[num] / g[den]).values
        yv = (g.real_gvp / g[den]).values
        m = np.isfinite(k) & np.isfinite(yv) & (k > 0) & (yv > 0)
        k, yv = k[m], yv[m]
        if hexbin and yr == YEARS[-1]:
            ax.hexbin(k, yv, gridsize=45, cmap="Blues", mincnt=1,
                      xscale="log", yscale="log", zorder=1)
        elif not hexbin:
            ax.scatter(k, yv, s=4, alpha=.10, color=COL[yr], linewidths=0, zorder=1)
        hx, hy = upper_concave_frontier(k, yv, xhi)
        pos = (hx > 0) & (hy > 0)                 # the origin cannot be shown on a log axis
        hx, hy = hx[pos], hy[pos]
        fr_max = max(fr_max, hy.max())
        ax.plot(hx, hy, lw=2, color=COL[yr], zorder=4, label=str(yr),
                solid_capstyle="round")
        label_at(ax, hx, hy, yr, [0.45, 0.62, 0.80][i])
    logaxes(ax, xlo, xhi, ylo, max(yhi_pts, fr_max) * 1.6)
    ax.set_xlabel(f"{dlab} per labour day  (log scale)")
    ax.set_ylabel("Real GVP per labour day  (log scale)")
    cap = ("Source: cleaned county agricultural panel (1,800 agricultural counties, "
           "2005 prices); a separate cross-sectional frontier is computed for each year "
           "over the counties observed that year. Axes: output per labour day against "
           f"{dlab.lower()} per labour day, both on log scales because the two span "
           "orders of magnitude across counties (the frontier itself is computed in "
           "levels). The frontier is the DEA best-practice boundary under non-increasing "
           "returns and FREE disposability of the input ratio; the flat right-hand "
           "segment is the free-disposability tail beyond the largest observed ratio. "
           "Points are counties.")
    finish(fig, ax, title, cap, fname, legend_loc="lower right")


# ------------------------------------------------------------------ FIGURE B
def figure_B(d, second, slab, fname, title):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sub = d[d.year.isin(YEARS)]
    L = (sub.Laborday_impute / sub.real_gvp).replace([np.inf, -np.inf], np.nan).dropna()
    S = (sub[second] / sub.real_gvp).replace([np.inf, -np.inf], np.nan).dropna()
    # compute every boundary FIRST so the axis limits can contain them -- with a
    # DEA hull the efficient chain can sit outside the cloud's percentile range,
    # in which case only its flat extension would otherwise be visible
    ch = {}
    for yr in YEARS:
        g = d[d.year == yr]
        l = (g.Laborday_impute / g.real_gvp).values
        sv = (g[second] / g.real_gvp).values
        m = np.isfinite(l) & np.isfinite(sv) & (l > 0) & (sv > 0)
        ch[yr] = (l[m], sv[m]) + lower_convex_isoquant(l[m], sv[m])
    xlo = min(np.nanpercentile(L, 1), min(c[2].min() for c in ch.values())) * .75
    xhi = np.nanpercentile(L, 99)
    ylo = min(np.nanpercentile(S, 1), min(c[3].min() for c in ch.values())) * .75
    yhi = max(np.nanpercentile(S, 99), max(c[3].max() for c in ch.values())) * 1.3
    for i, yr in enumerate(YEARS):
        l, sv, hx, hy = ch[yr]
        ax.scatter(l, sv, s=4, alpha=.10, color=COL[yr], linewidths=0, zorder=1)
        # FREE disposability on both conventional inputs -> flat extensions
        ex = np.concatenate([[hx[0]], hx, [xhi]])
        ey = np.concatenate([[yhi], hy, [hy[-1]]])
        ax.plot(ex, ey, lw=2, color=COL[yr], zorder=4, label=str(yr),
                solid_capstyle="round")
        label_at(ax, hx, hy, yr, [0.30, 0.50, 0.70][i])
    logaxes(ax, xlo, xhi, ylo, yhi)
    ax.set_xlabel("Labour days per unit of real GVP  (log scale)")
    ax.set_ylabel(f"{slab} per unit of real GVP  (log scale)")
    cap = ("Source: cleaned county agricultural panel. Unit isoquants (y = 1) built by "
           "dividing each county's inputs by its own observed output and taking the "
           "lower-left input-requirement boundary of the normalised cloud (Kumar & "
           "Russell 2002 convention), separately for each year. Both conventional inputs "
           "are FREELY disposable, so the boundary carries flat extensions on both axes. "
           "An isoquant closer to the origin is the more productive technology. "
           "Log scales; points are counties.")
    finish(fig, ax, title, cap, fname, legend_loc="upper right")


# --------------------------------------------------------------- FIGURES C/D
def figure_CD(d, wvar, wlab, fname, title, wunit):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sub = d[d.year.isin(YEARS)]
    X = (sub.xagg / sub.real_gvp).replace([np.inf, -np.inf], np.nan).dropna()
    W = (sub[wvar] / sub.real_gvp).replace([np.inf, -np.inf], np.nan).dropna()
    ch = {}
    for yr in YEARS:
        g = d[d.year == yr]
        xn = (g.xagg / g.real_gvp).values
        wn = (g[wvar] / g.real_gvp).values
        m = np.isfinite(xn) & np.isfinite(wn) & (xn > 0) & (wn > 0)
        bw, bx = lower_convex_isoquant(wn[m], xn[m])   # min input as a function of w
        o = np.argsort(bw)
        ch[yr] = (xn[m], wn[m], bx[o], bw[o])
    xlo = min(np.nanpercentile(X, 1), min(c[2].min() for c in ch.values())) * .75
    xhi = np.nanpercentile(X, 99)
    ylo = min(np.nanpercentile(W, 1), min(c[3].min() for c in ch.values())) * .75
    yhi = max(np.nanpercentile(W, 99), max(c[3].max() for c in ch.values())) * 1.3
    for i, yr in enumerate(YEARS):
        xn, wn, bx, bw = ch[yr]
        ax.scatter(xn, wn, s=4, alpha=.10, color=COL[yr], linewidths=0, zorder=1)
        # WEAK disposability in w: NO flat extension. End bars mark the
        # terminating feasible weather level -- the assumption made visible.
        ax.plot(bx, bw, lw=2, color=COL[yr], zorder=4, label=str(yr),
                solid_capstyle="butt", marker="_", markevery=[0, -1],
                markersize=12, markeredgewidth=2.5)
        label_at(ax, bx, bw, yr, [0.30, 0.55, 0.78][i])
    logaxes(ax, xlo, xhi, ylo, yhi)
    ax.set_xlabel("Törnqvist aggregate input per unit of real GVP  (log scale)")
    ax.set_ylabel(f"{wlab} per unit of real GVP {wunit}  (log scale)")
    cap = ("Source: cleaned county agricultural panel; weather from ERA5 aggregated to "
           "counties over the cropland-NDVI growing season. Output-normalised as in "
           "Figure B. The aggregate input is freely disposable but the weather variate is "
           "WEAKLY disposable — the equality constraint used in the main CPS "
           "decomposition — so, unlike Figures A and B, this curve has NO flat extension "
           "in the weather direction: it terminates at the extreme feasible weather "
           "level, marked by the horizontal end bars. Weather cannot be freely disposed "
           "of. Log scales; points are counties.")
    finish(fig, ax, title, cap, fname, legend_loc="upper right")


def main():
    d = frame()
    print("county counts:", {y: int((d.year == y).sum()) for y in YEARS})
    print("Figure A ...")
    figure_A(d, "capital_serv_q", "Laborday_impute", "Capital service",
             "figA_output_vs_capital_per_labour",
             "Figure A. Output per labour day and capital per labour day, 1986 / 2000 / 2015")
    figure_A(d, "Land_serv_q", "Laborday_impute", "Land service",
             "figA_output_vs_land_per_labour",
             "Figure A (variant). Output per labour day and land per labour day")
    figure_A(d, "capital_serv_q", "Laborday_impute", "Capital service",
             "figA_hexbin_capital_per_labour",
             "Figure A (density variant). 2015 county density with all three frontiers",
             hexbin=True)
    print("Figure B ...")
    figure_B(d, "capital_serv_q", "Capital service",
             "figB_unit_isoquant_labour_capital",
             "Figure B. Unit isoquants, labour and capital, 1986 / 2000 / 2015")
    figure_B(d, "Land_serv_q", "Land service",
             "figB_unit_isoquant_labour_land",
             "Figure B (variant). Unit isoquants, labour and land")
    print("Figures C/D ...")
    figure_CD(d, "gdd_growing_season", "Growing degree days",
              "figC_isoquant_input_gdd",
              "Figure C. Unit isoquants in input–weather space: growing degree days",
              "(°C·day)")
    figure_CD(d, "sm_0_28", "Root-zone soil moisture",
              "figD_isoquant_input_soilmoisture",
              "Figure D. Unit isoquants in input–weather space: soil moisture",
              "(m³/m³)")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
