# -*- coding: utf-8 -*-
"""
ALL weather figures: descriptive overview, input-weather frontiers, and the
three-dimensional output surface over inputs AND weather.

Merges 56_weather_overview.py and 57_weather_frontier.py.  They were two halves
of one question -- what the weather variables look like, and what the technology
looks like once weather is an argument of it -- and they shared the panel, the
year set and the palette while keeping private copies of each.

FAMILIES (--family, default all)
  overview   national weather series with the cross-county IQR band, and maps of
             where each variable sits and where soil moisture is changing.
  frontier   Figure 9w / 10w: output against WEATHER in two dimensions, with the
             conventional inputs entering through the output normalisation.
  surface    NEW.  The frontier as a SURFACE: the aggregate Tornqvist input index
             on the horizontal axis, WEATHER on the depth axis (toward the
             viewer), and real GVP vertical.  This is the object the
             Chambers-Pieralli technology actually is -- f(x, w) -- rather than
             either of its two-dimensional shadows.

DISPOSABILITY, and why the surface is not smoothed the same way in both
directions.  The conventional input index is FREELY disposable, so f must be
non-decreasing in x and the surface is made monotone along that axis.  Weather is
WEAKLY disposable -- the equality constraint of the CPS programme -- and it is
not chosen by the farmer, so NOTHING is imposed along the w axis.  The expected
shape there is a HUMP, not a ramp: too little heat and the crop does not develop,
too much and it is damaged; likewise drought against waterlogging for soil
moisture.  Imposing monotonicity in w would destroy exactly the shape the figure
exists to show, so the w direction is the raw cell quantile.

The surface is an order-alpha quantile over a grid of (ln x, w) cells rather than
a DEA hull.  On ~1,850 counties a year the hull is set by a handful of points
even in one dimension (see 59_frontier_audit.py); in two it would be worse, and a
surface drawn through five observations is not a technology.

Run:  python src/56_weather_figs.py
      python src/56_weather_figs.py --family surface
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers the 3d projection)
import _common as C
import _tfp as T
import _frontier as F

INK = "#1a1a19"

OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)
CAPTIONS = {}

DEFAULT_YEARS = [1981, 1985, 1989, 1994, 1999, 2003, 2008, 2012]

# weather variates: column -> (short, axis label, whether a z-score)
WVARS = {
    "gdd_growing_season": ("GDD", "Growing-season GDD  (°C·day, beneficial heat 8–32 °C)", False),
    "hdd_growing_season": ("HDD", "Growing-season HDD  (°C·day, harmful heat > 32 °C)", False),
    "sm_0_28_z":          ("SMz", "Root-zone soil-moisture anomaly  (sd from the county-month normal)", True),
}


def wf_load():
    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    need = ["real_gvp", "Laborday_impute", "Land_serv_q", "capital_serv_q",
            "Inter_all_real"] + list(WVARS)
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"io_weather_panel.csv lacks {miss} — re-run 12/13")
    d = d.dropna(subset=need)
    d = d[(d[need[:5]] > 0).all(axis=1)].copy()
    # Tornqvist aggregate input, CRS weights from the converged BC92 SFA
    b = T.BETA_AG
    s = sum(b.values())
    d["x"] = np.exp(sum((v / s) * np.log(d[k]) for k, v in b.items()))
    d["YL"] = d.real_gvp / d.Laborday_impute
    return d


def bin_quantile(w, v, alpha, nbins=14, minobs=40):
    """alpha-quantile of v within equal-count bins of w.

    NO monotone step.  Weather is weakly disposable and its effect is expected
    to be humped, so forcing the curve up or down would impose the answer.
    """
    w = np.asarray(w, float); v = np.asarray(v, float)
    ok = np.isfinite(w) & np.isfinite(v)
    w, v = w[ok], v[ok]
    if len(w) < nbins * minobs:
        nbins = max(4, len(w) // minobs)
    edges = np.unique(np.quantile(w, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(edges, w, side="right") - 1, 0, len(edges) - 2)
    bw, bv, bn = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < minobs:
            continue
        bw.append(np.median(w[m])); bv.append(np.quantile(v[m], alpha))
        bn.append(int(m.sum()))
    return np.array(bw), np.array(bv), np.array(bn)




OUT_OV = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT_OV, exist_ok=True)

# dataviz reference palette
BLUE6 = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
ORANGE6 = ["#fde3d5", "#fbc3a8", "#f79f76", "#f0784a", "#d95926", "#a8410f"]
# diverging pair: red (dry) <-> blue (wet), neutral gray midpoint
DIVERGE = ["#a8410f", "#d95926", "#f79f76", "#f0efec", "#6da7ec", "#256abf", "#184f95"]
NODATA = "#eceae6"
INK, INK2 = "#0b0b0b", "#52514e"

VARS = [
    ("gdd_growing_season", "(A)  Growing-season GDD",
     "°C·day, beneficial heat 8–32 °C", ORANGE6[3]),
    ("hdd_growing_season", "(B)  Growing-season HDD",
     "°C·day, harmful heat > 32 °C", ORANGE6[5]),
    ("sm_0_28", "(C)  Root-zone soil moisture — LEVEL",
     "m³/m³, 0–28 cm", BLUE6[4]),
    ("sm_0_28_z", "(D)  Root-zone soil moisture — ANOMALY",
     "sd from the 1981–2016 normal", BLUE6[5]),
]


def ov_load():
    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    need = [v for v, *_ in VARS]
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"io_weather_panel.csv lacks {miss} — re-run 12_county_gdd_hdd.py")
    return d.dropna(subset=need).copy()


def quantile_breaks(v, k):
    return np.unique(np.quantile(v.dropna(), np.linspace(0, 1, k + 1)))


def fmt(b, unit, dec=1):
    out = [f"≤ {b[1]:,.{dec}f}"]
    for lo, hi in zip(b[1:-2], b[2:-1]):
        out.append(f"{lo:,.{dec}f} – {hi:,.{dec}f}")
    out.append(f"≥ {b[-2]:,.{dec}f}")
    return [f"{s} {unit}" for s in out]


# --------------------------------------------------------------- Figure 1
def figure_timeseries(d):
    plt.rcParams.update({"font.size": 10, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    rows = []
    for ax, (var, title, unit, col) in zip(axes.ravel(), VARS):
        g = d.groupby("year")[var]
        m, q1, q3 = g.mean(), g.quantile(.25), g.quantile(.75)
        ax.fill_between(m.index, q1, q3, color=col, alpha=.16, linewidth=0,
                        label="county interquartile range")
        ax.plot(m.index, m.values, "-", color=col, lw=2.0, label="national mean")
        # OLS trend, reported per decade so the slope is readable
        x = m.index.values.astype(float)
        b1, b0 = np.polyfit(x, m.values, 1)
        ax.plot(x, b0 + b1 * x, "--", color=INK2, lw=1.2,
                label=f"trend {10*b1:+.3g} per decade")
        if var.endswith("_z"):
            ax.axhline(0, color=INK2, lw=.7, ls=":")
        ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("year"); ax.set_ylabel(unit, fontsize=9, color=INK2)
        ax.grid(alpha=.22); ax.legend(fontsize=8, loc="best", framealpha=.9)
        rows.append(pd.DataFrame({"year": m.index, "variable": var, "mean": m.values,
                                  "p25": q1.values, "p75": q3.values}))
    pd.concat(rows).to_csv(os.path.join(OUT_OV, "weather_national_annual.csv"), index=False)
    fig.suptitle("China county agriculture: growing-season weather, 1981–2016",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.5, 0.938, f"{d.countyid.nunique():,d} agricultural counties;  "
             "cropland-weighted county values, then the national mean",
             ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    p = os.path.join(C.FIG_DIR, "fig_weather_timeseries.png")
    fig.savefig(p, dpi=180, facecolor="white"); plt.close(fig)
    print("saved:", p)


# --------------------------------------------------------------- Figure 2
def figure_maps(d):
    from _county_match import load_boundaries, match_counties
    ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
              "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")
    g = load_boundaries()
    # 12_county_gdd_hdd.py already resolved countyid -> map_code when it built the
    # panel, so reuse that rather than re-running the matcher: a second match
    # could disagree with the one the weather was actually joined on.
    if "map_code" in d.columns:
        dd = d.dropna(subset=["map_code"]).copy()
        print(f"  using map_code from the panel ({dd.map_code.nunique():,d} polygons)")
    else:
        roster = (d.dropna(subset=["county_name"]).groupby("countyid", as_index=False)
                    .agg(county_name=("county_name", "last")))
        roster = match_counties(roster, g)
        dd = d.merge(roster[["countyid", "map_code"]], on="countyid", how="left") \
              .dropna(subset=["map_code"])

    # per-county means, and the anomaly TREND (slope per decade)
    def slope(s):
        if s.notna().sum() < 10:
            return np.nan
        return 10 * np.polyfit(s.index.values.astype(float), s.values, 1)[0]

    agg = dd.groupby("map_code").agg(gdd=("gdd_growing_season", "mean"),
                                     hdd=("hdd_growing_season", "mean")).reset_index()
    tr = (dd.set_index("year").groupby("map_code")["sm_0_28_z"]
            .apply(slope).rename("sm_trend").reset_index())
    agg = agg.merge(tr, on="map_code", how="left")
    agg["map_code"] = agg.map_code.astype(int)
    agg.to_csv(os.path.join(OUT_OV, "weather_county_summary.csv"), index=False)
    print(f"  county summary: {len(agg):,d} polygons; "
          f"soil-moisture trend median {agg.sm_trend.median():+.3f} sd/decade, "
          f"{100*(agg.sm_trend < 0).mean():.0f}% drying")

    gm = g.to_crs(ALBERS).merge(agg, left_on="code", right_on="map_code", how="left")
    prov = gm.dissolve(by=gm.code // 10000)

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.4))
    specs = [("gdd", "(A)  Mean growing-season GDD", ORANGE6, "°C·day", 0, False),
             ("hdd", "(B)  Mean growing-season HDD", ORANGE6, "°C·day", 1, False),
             ("sm_trend", "(C)  Soil-moisture anomaly trend", DIVERGE,
              "sd / decade", 2, True)]
    for ax, (col, title, ramp, unit, _, diverging) in zip(axes, specs):
        v = gm[col]
        if diverging:
            # symmetric breaks about zero so the neutral midpoint means "no change"
            lim = np.nanpercentile(np.abs(v.dropna()), 95)
            breaks = np.array([-lim, -lim*.6, -lim*.25, -lim*.05,
                               lim*.05, lim*.25, lim*.6, lim])
            cmap = ListedColormap(ramp); dec = 2
        else:
            breaks = quantile_breaks(v, len(ramp)); cmap = ListedColormap(ramp); dec = 0
        cmap.set_bad(NODATA)
        norm = BoundaryNorm(breaks, ncolors=cmap.N, clip=True)
        gm.plot(column=col, ax=ax, cmap=cmap, norm=norm, linewidth=0,
                missing_kwds=dict(color=NODATA), zorder=1)
        prov.boundary.plot(ax=ax, color="white", linewidth=.45, zorder=2)
        ax.set_title(title, fontsize=12.5, loc="left", fontweight="bold", color=INK)
        ax.set_axis_off(); ax.autoscale_view()
        labs = fmt(breaks, unit, dec)
        handles = [Patch(facecolor=c, edgecolor="white", linewidth=.6, label=l)
                   for c, l in zip(ramp, labs)]
        handles.append(Patch(facecolor=NODATA, edgecolor="white", linewidth=.6,
                             label="no data"))
        leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                        framealpha=.94, edgecolor="#d8d6d1", fontsize=8,
                        title=("symmetric classes" if diverging else "equal-count classes"),
                        title_fontsize=8.5, borderpad=.6, labelspacing=.35)
        leg.get_title().set_color(INK2)
        for tx in leg.get_texts():
            tx.set_color(INK2)

    fig.suptitle("Where the weather is, and where soil moisture is changing",
                 fontsize=14.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.945, "growing season = cropland-NDVI multi-peak climatology (A2);  "
             "anomaly = per county-month vs its own 1981–2016 normal (B2);  "
             "negative trend = drying",
             ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    p = os.path.join(C.FIG_DIR, "fig_weather_maps.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved:", p)




# ============================================================ FAMILY: surface
XCOLS = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]


def surface_quantile(lx, w, ly, alpha, nx=12, nw=10, minobs=12, xe=None, we=None):
    """Order-alpha frontier surface over a grid of (ln x, w) cells.

    Within each cell take the alpha-quantile of ln y, then enforce free
    disposability in x ONLY -- a running maximum along the x axis inside each w
    slice.  Nothing is imposed along w, because weather is weakly disposable and
    the shape there is expected to be humped rather than monotone.

    Cells below `minobs` are left NaN and drawn as holes rather than filled by
    interpolation: a surface is only claimed where there are observations to
    support it.
    """
    # `xe`/`we` let the CALLER fix the grid.  Stacking years in one axes requires
    # it: if every year sets its own quantile-based edges, the cells are not the
    # same object across years and a vertical gap between two surfaces would be
    # partly a change of grid rather than a change of technology.
    if xe is None:
        xe = np.linspace(np.quantile(lx, .01), np.quantile(lx, .99), nx + 1)
    if we is None:
        we = np.linspace(np.quantile(w, .01), np.quantile(w, .99), nw + 1)
    nx, nw = len(xe) - 1, len(we) - 1
    xi = np.clip(np.searchsorted(xe, lx, "right") - 1, 0, nx - 1)
    wi = np.clip(np.searchsorted(we, w, "right") - 1, 0, nw - 1)
    Z = np.full((nw, nx), np.nan)
    for i in range(nw):
        for j in range(nx):
            m = (wi == i) & (xi == j)
            if m.sum() >= minobs:
                Z[i, j] = np.quantile(ly[m], alpha)
    for i in range(nw):                       # free disposal in x, within a w slice
        run = -np.inf
        for j in range(nx):
            if np.isfinite(Z[i, j]):
                run = max(run, Z[i, j])
                Z[i, j] = run
    xc = 0.5 * (xe[:-1] + xe[1:])
    wc = 0.5 * (we[:-1] + we[1:])
    return xc, wc, Z


def dea_surface(x, w, y, xc, wc, nirs=True):
    """TRUE DEA frontier surface f(x, w), by linear programming.

    Nonparametric does NOT mean two-dimensional.  The DEA technology with one
    output and (here) two arguments is

        f(x0, w0) = max  SUM_i lam_i y_i
             s.t.  SUM_i lam_i x_i <= x0     free disposability in x
                   SUM_i lam_i w_i  = w0     WEAK disposability in w (equality)
                   SUM_i lam_i     <= 1      NIRS
                   lam >= 0

    and evaluating that LP over a grid of (x0, w0) traces a SURFACE, exactly as
    evaluating it over a grid of x0 alone traces a curve.  The object is
    piecewise linear and concave over the convex hull of the data -- a polyhedral
    sheet, not a fitted function -- so no functional form has to be chosen.  This
    is the same programme 41_cps_decomposition.py solves for the CPS
    decomposition, with one weather variate instead of two.

    Solved as the DUAL, which has three variables and one constraint per
    reference point rather than one variable per point:
        min  x0*u + v + w0*p   s.t.  x_i*u + v + w_i*p >= y_i,  u,v >= 0, p free
    The dual is always feasible (u=p=0, v=max y), so DUAL UNBOUNDED is exactly
    PRIMAL INFEASIBLE.

    INFEASIBILITY IS THE POINT, not a nuisance.  Because w enters as an EQUALITY,
    f is undefined wherever w0 lies outside conv({w_i}) -- weather cannot be
    freely disposed of, so a weather level nobody experienced is not attainable
    by scaling anyone down.  Those cells come back NaN and are drawn as holes.
    A free-disposability (inequality) version would fill them in and would also
    force the surface to be monotone in w, destroying the hump.

    Everything is in LEVELS, because convexity of the technology set is a
    level-space assumption; the caller takes logs only to draw.
    """
    P = np.column_stack([x, w, y])
    if len(P) > 8:                      # exact pruning: the max sits at a vertex
        s = np.abs(P).max(axis=0); s[s == 0] = 1.0
        try:
            v = np.unique(ConvexHull(np.vstack([P / s, np.zeros(3)]),
                                     qhull_options="Qx").vertices)
            v = v[v < len(P)]
            if len(v) >= 4:
                P = P[v]
        except Exception:
            pass
    xr, wr, yr = P[:, 0], P[:, 1], P[:, 2]
    A_ub = -np.column_stack([xr, np.ones_like(xr), wr])
    b_ub = -yr
    bnd = [(0, None), (0, None) if nirs else (None, None), (None, None)]
    Z = np.full((len(wc), len(xc)), np.nan)
    nfeas = 0
    for i, w0 in enumerate(wc):
        for j, x0 in enumerate(xc):
            r = linprog(np.array([x0, 1.0, w0]), A_ub=A_ub, b_ub=b_ub,
                        bounds=bnd, method="highs")
            if r.status == 3 or not r.success or r.fun is None or r.fun <= 0:
                continue                # dual unbounded <=> primal infeasible
            Z[i, j] = np.log(r.fun)
            nfeas += 1
    return Z, nfeas, len(P)


def fam_surface(a):
    """All years' frontier surfaces STACKED IN ONE AXES, plus the two slices that
    make the stack readable.

    Separate per-year panels show what was feasible in each year but cannot show
    MOVEMENT: the reader has to compare eight perspective projections by eye, and
    a vertical shift of a few percent is invisible that way.  Stacking puts every
    year in one coordinate system, so technical progress is a physical gap
    between sheets and adaptation is a change in the SHAPE of the sheets along
    the weather axis.

    All years share ONE grid (edges pooled over the whole sample), which is what
    makes the gaps comparable -- see surface_quantile.  Surfaces are drawn as
    WIREFRAMES rather than filled: eight translucent filled sheets occlude each
    other into mush, while wireframes let the lower years show through.

    Panels:
      (A) the stack itself, one colour per year.
      (B) f at the MEDIAN x, against weather.  This is the adaptation readout:
          if the sheets shift up in parallel, weather sensitivity is unchanged
          and only the level moved; if the curve FLATTENS or its peak MOVES, the
          sector's response to that weather variable has changed.
      (C) f at the MEDIAN weather, against x.  The ordinary frontier shift --
          technical progress with weather held fixed.
      (D) the gap between the LATE and EARLY period at each weather level.  If
          progress is uniform across w the line is flat; if it is larger at
          harmful weather levels, that is adaptation localised where it matters.
    """
    d = wf_load()
    d["x"], w8 = F.aggregate_index(d, XCOLS, dict(T.BETA_AG))
    print("  index weights: " + ", ".join("%s=%.4f" % (k.split("_")[0], v)
                                          for k, v in w8.items()))
    YEARS = [int(y) for y in a.years.split(",")]
    for wcol, (short, wlab, is_z) in WVARS.items():
        if wcol not in d.columns:
            print("  [%s] not in the panel -- skipped" % short)
            continue
        yrs = [y for y in YEARS if (d.year == y).sum() > 300]
        if len(yrs) < 2:
            print("  [%s] fewer than two usable years -- skipped" % short)
            continue
        pool = d[d.year.isin(yrs)].dropna(subset=["x", "real_gvp", wcol])
        pool = pool[(pool.x > 0) & (pool.real_gvp > 0)]
        # ONE grid for every year, from the pooled support
        xe = np.linspace(np.quantile(np.log(pool.x), .01),
                         np.quantile(np.log(pool.x), .99), 13)
        we = np.linspace(np.quantile(pool[wcol], .01),
                         np.quantile(pool[wcol], .99), 11)
        cols = plt.cm.viridis(np.linspace(0, .92, len(yrs)))

        xc = 0.5 * (xe[:-1] + xe[1:])
        wc = 0.5 * (we[:-1] + we[1:])
        Zs, rows = {}, []
        for yr in yrs:
            g = pool[pool.year == yr]
            if a.surface_kind == "dea":
                # LEVELS in, logs out -- convexity is a level-space assumption
                Z, nf, nv = dea_surface(g.x.values, g[wcol].values,
                                        g.real_gvp.values, np.exp(xc), wc)
                print("    [%s %d] DEA surface: %d of %d cells feasible, "
                      "%d hull vertices" % (short, yr, nf, Z.size, nv))
            else:
                _, _, Z = surface_quantile(np.log(g.x.values), g[wcol].values,
                                           np.log(g.real_gvp.values), a.alpha,
                                           xe=xe, we=we)
            Zs[yr] = Z
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if np.isfinite(Z[i, j]):
                        rows.append(dict(year=yr, wvar=short, ln_x=xc[j],
                                         w=wc[i], ln_f=Z[i, j]))
        X, W = np.meshgrid(xc, wc)

        fig = plt.figure(figsize=(17.5, 11.5))
        # ---- (A) the stack
        ax = fig.add_subplot(2, 2, 1, projection="3d")
        for yr, col in zip(yrs, cols):
            ax.plot_wireframe(X, W, Zs[yr], color=col, linewidth=.85,
                              rstride=1, cstride=1, alpha=.95)
        ax.set_xlabel("ln aggregate input x", fontsize=8, labelpad=2)
        ax.set_ylabel(short, fontsize=8, labelpad=2)
        ax.set_zlabel("ln real GVP", fontsize=8, labelpad=2)
        ax.set_title("(A)  all years stacked", fontsize=10.5, loc="left",
                     fontweight="bold")
        ax.view_init(elev=20, azim=-60)
        ax.tick_params(labelsize=6.5)

        # ---- (B) weather response at the median x  -> ADAPTATION
        jx = Zs[yrs[0]].shape[1] // 2
        axb = fig.add_subplot(2, 2, 2)
        for yr, col in zip(yrs, cols):
            axb.plot(wc, Zs[yr][:, jx], "-o", ms=3.5, lw=1.8, color=col, label=str(yr))
        axb.set_xlabel("%s" % wlab, fontsize=9)
        axb.set_ylabel("ln f at the median input level", fontsize=9)
        axb.set_title("(B)  weather response, x held at its median  —  ADAPTATION",
                      fontsize=10.5, loc="left", fontweight="bold")
        axb.grid(alpha=.25); axb.legend(fontsize=7, title="year", ncol=2)

        # ---- (C) input response at the median weather -> TECHNICAL PROGRESS
        iw = Zs[yrs[0]].shape[0] // 2
        axc = fig.add_subplot(2, 2, 3)
        for yr, col in zip(yrs, cols):
            axc.plot(xc, Zs[yr][iw, :], "-o", ms=3.5, lw=1.8, color=col, label=str(yr))
        axc.set_xlabel("ln aggregate input x", fontsize=9)
        axc.set_ylabel("ln f at the median weather level", fontsize=9)
        axc.set_title("(C)  frontier shift, weather held at its median  —  "
                      "TECHNICAL PROGRESS", fontsize=10.5, loc="left",
                      fontweight="bold")
        axc.grid(alpha=.25); axc.legend(fontsize=7, title="year", ncol=2)

        # ---- (D) late minus early, at each weather level
        # Period means rather than single endpoint years: one year's cell
        # quantile is noisy, and the question is the trend, not 1986 vs 2015.
        # A w-row where EITHER period has no cell stays NaN -- averaging over an
        # empty slice would otherwise emit a warning and a meaningless zero.
        ne = max(1, len(yrs) // 3)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells
            early = np.nanmean(np.stack([Zs[y] for y in yrs[:ne]]), axis=0)
            late = np.nanmean(np.stack([Zs[y] for y in yrs[-ne:]]), axis=0)
            diff = late - early
            ok = np.isfinite(diff)
            gap = np.where(ok.any(axis=1),
                           np.nansum(np.where(ok, diff, 0), axis=1)
                           / np.maximum(ok.sum(axis=1), 1), np.nan)
        axd = fig.add_subplot(2, 2, 4)
        # Quantify the tilt instead of leaving it to the eye: OLS of the gap on
        # w.  A slope indistinguishable from 0 means the sheets moved up in
        # PARALLEL (level progress, unchanged weather sensitivity); a nonzero
        # slope means the progress was concentrated at one end of the weather
        # range, which is the adaptation claim.
        okg = np.isfinite(gap)
        slope = np.nan
        if okg.sum() >= 4:
            slope = np.polyfit(wc[okg], gap[okg], 1)[0]
        axd.plot(wc, gap, "-o", ms=4, lw=2.0, color="#1f4e79")
        axd.axhline(np.nanmean(gap), color="0.5", ls="--", lw=1,
                    label="mean shift %.3f ln  (= %+.0f%% output)"
                          % (np.nanmean(gap), 100 * (np.exp(np.nanmean(gap)) - 1)))
        if np.isfinite(slope):
            axd.plot(wc[okg], np.polyval(np.polyfit(wc[okg], gap[okg], 1), wc[okg]),
                     ":", color="#b3541e", lw=1.6,
                     label="tilt %+.4f ln per unit %s" % (slope, short))
        axd.set_xlabel("%s" % wlab, fontsize=9)
        axd.set_ylabel("ln f (late) − ln f (early)", fontsize=9)
        axd.set_title("(D)  where the progress happened: %s−%s minus %s−%s"
                      % (yrs[-ne], yrs[-1], yrs[0], yrs[ne - 1]),
                      fontsize=10.5, loc="left", fontweight="bold")
        axd.grid(alpha=.25); axd.legend(fontsize=8)

        fig.suptitle("Frontier surface over inputs and %s "
                     % (wlab.split("  ")[0], a.alpha),
                     fontsize=13.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .94))
        stub = "fig_surface_%s_%s" % (short, a.surface_kind)
        p = os.path.join(C.FIG_DIR, stub + ".png")
        fig.savefig(p, dpi=165, facecolor="white")
        plt.close(fig)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(OUT, stub + ".csv"), index=False)
        print("  saved %s | (D) mean shift %.3f ln (%+.0f%% output), spread over w "
              "%.3f, tilt %+.5f per unit %s"
              % (os.path.basename(p), np.nanmean(gap),
                 100 * (np.exp(np.nanmean(gap)) - 1),
                 np.nanmax(gap) - np.nanmin(gap), slope, short))
        CAPTIONS[stub] = (
            "Frontier surface over inputs and weather, %s, all years in ONE coordinate "
            "system. Horizontal axis: the constant-returns Tornqvist aggregate of "
            "labour, land service, capital service and intermediates, in logs. Depth "
            "axis: %s. Vertical: log real GVP. The surface is the %g-quantile of log "
            "output within cells of (ln x, w); free disposability of x is imposed as a "
            "running maximum along the x axis, and NOTHING is imposed along the weather "
            "axis because weather is weakly disposable and its effect is expected to be "
            "humped. Every year shares ONE grid taken from the pooled support, so a "
            "vertical gap between sheets is a change in technology and not a change of "
            "grid. Panel A stacks the years as wireframes (filled sheets would occlude "
            "each other). Panel B is the weather response with input held at its median: "
            "parallel upward shifts mean the level moved but sensitivity did not, while "
            "a flattening or a moving peak is adaptation. Panel C is the ordinary "
            "frontier shift with weather held at its median. Panel D is the late-minus-"
            "early gap at each weather level, averaged over x: a flat line means "
            "progress was uniform across weather, a sloped one means it was concentrated "
            "where that weather variable bites."
            % (", ".join(map(str, yrs)), wlab, a.alpha))


# ========================================================= FAMILY: trajectory
def fam_trajectory(a):
    """The SAME three variables as the surface, drawn as a PATH THROUGH TIME.

    The surface answers "what was feasible in year t"; it cannot answer "how did
    the sector move", because each year is a separate panel and nothing connects
    them.  This answers the second question and is the more legible object: one
    point per year, joined in time order, so the reader follows a line instead of
    comparing eight surfaces by eye.

    Three devices carry the third dimension, because a 3-D scatter alone is
    ambiguous on a flat page:
      * the DROPLINES from each point to the floor fix its height, which is
        otherwise impossible to read off a perspective projection;
      * the dashed FLOOR PROJECTION is the (x, w) path with output removed -- the
        input-weather history on its own;
      * COLOUR repeats the year, so direction of travel is readable even where
        the path crosses itself in projection.

    Aggregation is OUTPUT-WEIGHTED over counties, matching 30_aggregate_tfp.py:
    a national point is what the sector did, not what the median county did, and
    an unweighted mean over ~1,850 counties would let the smallest units move the
    path as much as the largest.  Weather is weighted the same way so all three
    coordinates describe the same aggregate.
    """
    d = wf_load()
    d["x"], w8 = F.aggregate_index(d, XCOLS, dict(T.BETA_AG))
    d = F.add_region(d)
    groups = ([(F.REGION_SHORT[r], d[d.region == r]) for r, n in F.region_counts(d)
               if n >= 25] if a.by_region else [("All agricultural counties", d)])

    for wcol, (short, wlab, is_z) in WVARS.items():
        if wcol not in d.columns:
            print("  [%s] not in the panel -- skipped" % short)
            continue
        ncol = min(3, len(groups))
        nrow = int(np.ceil(len(groups) / ncol))
        fig = plt.figure(figsize=(6.6 * ncol, 5.6 * nrow))
        rows = []
        for k, (gname, g) in enumerate(groups):
            g = g.dropna(subset=["x", "real_gvp", wcol])
            g = g[(g.x > 0) & (g.real_gvp > 0)]
            if g.empty:
                continue
            wt = g.real_gvp
            p = (g.assign(_lx=np.log(g.x) * wt, _ly=np.log(g.real_gvp) * wt,
                          _w=g[wcol] * wt, _wt=wt)
                  .groupby("year")[["_lx", "_ly", "_w", "_wt"]].sum())
            p = pd.DataFrame({"lx": p._lx / p._wt, "ly": p._ly / p._wt,
                              "w": p._w / p._wt}).sort_index()
            yrs = p.index.to_numpy()
            ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
            zfloor = p.ly.min() - 0.06 * (p.ly.max() - p.ly.min() + 1e-9)
            # droplines first so the path draws over them
            for xx, ww, zz in zip(p.lx, p.w, p.ly):
                ax.plot([xx, xx], [ww, ww], [zfloor, zz], color="0.80", lw=.6,
                        ls=":", zorder=1)
            ax.plot(p.lx, p.w, np.full(len(p), zfloor), color="0.65", lw=1.2,
                    ls="--", zorder=2)                       # floor projection
            ax.plot(p.lx, p.w, p.ly, color="0.45", lw=1.1, zorder=3)
            sc = ax.scatter(p.lx, p.w, p.ly, c=yrs, cmap="viridis", s=42,
                            edgecolor="white", linewidth=.6, depthshade=False,
                            zorder=4)
            mark = [yrs[0]] + [y for y in yrs if y % 10 == 0] + [yrs[-1]]
            for y in dict.fromkeys(mark):
                r = p.loc[y]
                ax.text(r.lx, r.w, r.ly, "  %d" % y, fontsize=8,
                        fontweight="bold", color=INK, zorder=6)
            ax.set_xlabel("ln aggregate input x", fontsize=8, labelpad=2)
            ax.set_ylabel(short, fontsize=8, labelpad=2)
            ax.set_zlabel("ln real GVP", fontsize=8, labelpad=2)
            ax.set_zlim(zfloor, p.ly.max() + 0.02 * (p.ly.max() - p.ly.min() + 1e-9))
            ax.set_title("%s   (%d-%d)" % (gname, yrs[0], yrs[-1]), fontsize=10)
            ax.view_init(elev=20, azim=-62)
            ax.tick_params(labelsize=6.5)
            rows.append(p.assign(group=gname, wvar=short).reset_index())
        cb = fig.colorbar(sc, ax=fig.axes, shrink=.55, pad=.02)
        cb.set_label("year", fontsize=8); cb.ax.tick_params(labelsize=7)
        fig.suptitle("Sector path through input, weather and output space: %s\n"
                     "one point per year, output-weighted over counties; droplines and "
                     "the dashed floor track give the third dimension"
                     % wlab.split("  ")[0], fontsize=12.5, fontweight="bold")
        stub = "fig_path_%s%s" % (short, "_byregion" if a.by_region else "")
        p_ = os.path.join(C.FIG_DIR, stub + ".png")
        fig.savefig(p_, dpi=165, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        if rows:
            pd.concat(rows).to_csv(os.path.join(OUT, stub + ".csv"), index=False)
        print("  saved", os.path.basename(p_))
        CAPTIONS[stub] = (
            "Sector path through input-weather-output space, %d-%d. Horizontal axis: "
            "the constant-returns Tornqvist aggregate of labour, land service, capital "
            "service and intermediates, in logs. Depth axis: %s. Vertical axis: log real "
            "GVP. One marker per year, joined in time order and coloured by year; "
            "dotted droplines run from each marker to the floor, and the dashed floor "
            "curve is the same path with output removed, i.e. the input-weather history "
            "on its own. All three coordinates are OUTPUT-WEIGHTED means over counties, "
            "matching the aggregation in 30_aggregate_tfp.py, so the path describes what "
            "the sector did rather than what the median county did. This is the "
            "companion to the frontier surface: the surface is what was feasible in a "
            "given year, this is how the sector actually moved."
            % (rows[0].year.min(), rows[0].year.max(), wlab))


# ---------------------------------------------------------------------------
def fam_overview(a):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    d = ov_load()
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} counties, "
          f"{d.year.min()}-{d.year.max()}")
    figure_timeseries(d)
    if not a.no_maps:
        figure_maps(d)


def fam_frontier(a):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="upper quantile for Fig 9W (output frontier)")
    ap.add_argument("--iso-alpha", type=float, default=0.05,
                    help="lower quantile for Fig 10W (input requirement)")
    a = ap.parse_args()

    d = wf_load()
    YEARS = [int(y) for y in a.years.split(",")]
    missing = [y for y in YEARS if y not in set(d.year)]
    if missing:
        raise SystemExit(f"years not in the panel: {missing}")
    ystar = d.real_gvp.median()
    cols = plt.cm.viridis(np.linspace(0, .88, len(YEARS)))
    ny = d[d.year.isin(YEARS)].groupby("year").countyid.nunique()
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} counties")
    print("years:", ", ".join(f"{y} (n={ny[y]:,d})" for y in YEARS))
    print(f"y* = {ystar:,.0f} (2005 prices)\n")

    for wcol, (short, wlab, is_z) in WVARS.items():
        # ---------------------------------------------------- FIGURE 9W
        fig, ax = plt.subplots(figsize=(9.5, 6.6))
        rows = []
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            ax.scatter(s[wcol], s.YL, s=8, alpha=.20, color=col, linewidths=0,
                       rasterized=True)
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            bw, bv, bn = bin_quantile(s[wcol].values, s.YL.values, a.alpha)
            if len(bw) < 3:
                print(f"  [{short}] {yr}: too few bins, skipped", file=sys.stderr)
                continue
            ax.plot(bw, bv, "-o", color=col, lw=2.2, ms=4,
                    label=f"{yr}  (n={len(s):,d})")
            rows.append(pd.DataFrame({"year": yr, "wvar": short, "w": bw,
                                      "f_YL": bv, "n_bin": bn}))
            k = int(np.argmax(bv))
            print(f"  [{short}] {yr}: peak Y/L at {short}={bw[k]:,.3g} "
                  f"(bin {k+1}/{len(bw)}) -> {'INTERIOR (humped)' if 0 < k < len(bw)-1 else 'at an endpoint (monotone)'}")
        pd.concat(rows).to_csv(os.path.join(OUT, f"fig9w_frontier_{short}.csv"), index=False)
        ax.set_yscale("log")
        ax.set_xlabel(wlab); ax.set_ylabel("Y/L — real output per labour day (log)")
        ax.set_title(f"Figure 9W. Output frontier in weather space — {short}\n"
                     f"order-{chr(945)} quantile ({chr(945)}={a.alpha:g}) within weather bins; "
                     "no monotone step (weather is weakly disposable)", fontsize=11)
        ax.legend(fontsize=8, title="year"); ax.grid(alpha=.25, which="both")
        if is_z:
            ax.axvline(0, color="0.5", lw=.8, ls=":")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, f"fig9w_frontier_{short}.png"), dpi=170)
        plt.close(fig)
        print(f"  saved fig9w_frontier_{short}.png")
        CAPTIONS[f"fig9w_frontier_{short}"] = (
            f"Figure 9W ({short}). Output per labour day against {wlab.lower()}, "
            f"{', '.join(map(str, YEARS))}. Source: cleaned county agricultural panel "
            f"joined to ERA5/CACD weather over the cropland-NDVI growing season; "
            f"{d.countyid.nunique():,d} agricultural counties. A separate boundary is "
            "computed for each year using only that year's observations. The boundary "
            f"is the {a.alpha:.0%} quantile of Y/L within equal-count weather bins. "
            "Unlike the conventional Figure 9 it is NOT made monotone: weather is "
            "weakly disposable in the Chambers-Pieralli technology and is not chosen by "
            "the farmer, and its effect is expected to be humped rather than "
            "monotone -- too little heat and the crop does not develop, too much and it "
            "is damaged. An interior peak is therefore the meaningful signature, and "
            "the printed diagnostic reports whether each year's peak is interior. "
            "Output is on a log scale; weather is on its natural scale. Points are "
            "county-years.")

        # ---------------------------------------------------- FIGURE 10W
        fig, ax = plt.subplots(figsize=(9.5, 6.6))
        rows = []
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            f_ = ystar / s.real_gvp
            ax.scatter(s[wcol], s.x * f_, s=8, alpha=.20, color=col,
                       linewidths=0, rasterized=True)
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            xs = (s.x * (ystar / s.real_gvp)).values
            bw, bv, bn = bin_quantile(s[wcol].values, xs, a.iso_alpha)
            if len(bw) < 3:
                print(f"  [{short}] {yr}: too few bins, skipped", file=sys.stderr)
                continue
            ax.plot(bw, bv, "-o", color=col, lw=2.2, ms=4,
                    label=f"{yr}  (n={len(s):,d})")
            rows.append(pd.DataFrame({"year": yr, "wvar": short, "w": bw,
                                      "x_min": bv, "n_bin": bn}))
            k = int(np.argmin(bv))
            print(f"  [{short}] {yr}: least input at {short}={bw[k]:,.3g} "
                  f"(bin {k+1}/{len(bw)}) -> {'INTERIOR (U-shaped)' if 0 < k < len(bw)-1 else 'at an endpoint'}")
        pd.concat(rows).to_csv(os.path.join(OUT, f"fig10w_isoquant_{short}.csv"), index=False)
        ax.set_yscale("log")
        ax.set_xlabel(wlab)
        ax.set_ylabel(f"Törnqvist aggregate input needed for y* (log)")
        ax.set_title(f"Figure 10W. Input requirement in weather space at y* — {short}\n"
                     f"{a.iso_alpha:.0%} quantile of y*-scaled input within weather bins; "
                     "weak disposability, no flat extension", fontsize=11)
        ax.legend(fontsize=8, title="year"); ax.grid(alpha=.25, which="both")
        if is_z:
            ax.axvline(0, color="0.5", lw=.8, ls=":")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, f"fig10w_isoquant_{short}.png"), dpi=170)
        plt.close(fig)
        print(f"  saved fig10w_isoquant_{short}.png\n")
        CAPTIONS[f"fig10w_isoquant_{short}"] = (
            f"Figure 10W ({short}). Input requirement at y* = median real GVP "
            f"({ystar:,.0f}, 2005 prices) against {wlab.lower()}, "
            f"{', '.join(map(str, YEARS))}. Every county-year is radially scaled to y* "
            "(Farrell normalisation) and the boundary is the "
            f"{a.iso_alpha:.0%} quantile of the y*-scaled Törnqvist aggregate input "
            "within equal-count weather bins, separately for each year. The aggregate "
            "input uses the CRS weights from the converged BC92 SFA. Reads as: the "
            "least input needed to produce y* at this weather. A U-shape is the "
            "expected signature -- benign weather needs less input, extreme weather on "
            "either side needs more -- and the printed diagnostic reports whether each "
            "year's minimum is interior. Weather is weakly disposable, so unlike the "
            "conventional unit isoquant the boundary carries no flat free-disposal "
            "extension. Note this object requires INPUT-WEATHER substitution, not "
            "factor-factor substitution, which is why it can be informative on a panel "
            "where the conventional labour-capital isoquant is degenerate. Input is on "
            "a log scale; weather is on its natural scale.")

    cap_md = os.path.join(OUT, "captions_fig9w_10w.md")
    with open(cap_md, "w", encoding="utf-8") as fh:
        fh.write("# Figure captions - Fig 9W / 10W (input-weather space)\n\n")
        for k in sorted(CAPTIONS):
            fh.write(f"## {k}\n\n{CAPTIONS[k]}\n\n")
    print("=" * 78)
    print("FIGURE CAPTIONS (also written to captions_fig9w_10w.md)")
    print("=" * 78)
    for k in sorted(CAPTIONS):
        print(f"\n[{k}]\n{CAPTIONS[k]}")
    print(f"\ncaptions -> {cap_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all",
                    choices=["all", "overview", "frontier", "surface",
                             "trajectory"])
    ap.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--surface-kind", choices=["alpha", "dea"], default="alpha",
                    help="alpha = order-alpha quantile over (ln x, w) cells "
                         "(robust, the default); dea = the true LP frontier "
                         "surface, weak disposability in w -- exact but set by a "
                         "handful of counties, see 59_frontier_audit.py")
    ap.add_argument("--by-region", action="store_true",
                    help="trajectory family: one path per 九大农业区")
    a = ap.parse_args()
    fams = (["overview", "frontier", "surface", "trajectory"]
            if a.family == "all" else [a.family])
    disp = {"overview": fam_overview, "frontier": fam_frontier,
            "surface": fam_surface, "trajectory": fam_trajectory}
    for fam in fams:
        print("\n--- family: %s ---" % fam)
        disp[fam](a)
    if CAPTIONS:
        cap = os.path.join(OUT, "captions_weather.md")
        with open(cap, "w", encoding="utf-8") as fh:
            fh.write("# Figure captions - weather figures\n\n")
            for k in sorted(CAPTIONS):
                fh.write("## %s\n\n%s\n\n" % (k, CAPTIONS[k]))
        print("\ncaptions ->", cap)


if __name__ == "__main__":
    main()
