# -*- coding: utf-8 -*-
"""
Descriptive overview of the weather variables — what they look like before any
frontier or regression touches them.

FIGURE 1  fig_weather_timeseries.png
  National annual series for each weather variable, cropland-weighted county
  means, with the cross-county interquartile band behind them and an OLS trend.
  The four panels are deliberately NOT on shared axes and NOT dual-axis: the
  units differ (degC-day vs m3/m3 vs sd), and one axis per measure is the only
  honest way to show them together.

  The pairing that matters is (C) against (D).  They are the SAME soil moisture:
  (C) is the level, (D) the standardised anomaly.  The level barely moves and
  looks like noise; the anomaly shows the drying plainly.  That contrast is the
  whole argument for using the anomaly (strategy B2) rather than the level, made
  visible instead of asserted.

FIGURE 2  fig_weather_maps.png
  Where each variable sits, and where soil moisture is changing.
    (A) mean growing-season GDD      sequential, one hue
    (B) mean growing-season HDD      sequential, second hue
    (C) soil-moisture anomaly TREND  diverging -- this one is genuinely polar
        (wetting vs drying), so it gets two hues and a neutral midpoint at zero.
  Classed by quantile for (A) and (B) because ~2,000 small polygons wash out
  under a continuous ramp; (C) is classed symmetrically around zero so the
  midpoint keeps its meaning.

Input : src/clean/io_weather_panel.csv  (12_county_gdd_hdd.py)
Output: src/figures/fig_weather_timeseries.png, fig_weather_maps.png
        src/clean/descriptive/weather_national_annual.csv
        src/clean/descriptive/weather_county_summary.csv
Run:  python src/56_weather_overview.py
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import _common as C

OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)

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


def load():
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
    pd.concat(rows).to_csv(os.path.join(OUT, "weather_national_annual.csv"), index=False)
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
    agg.to_csv(os.path.join(OUT, "weather_county_summary.csv"), index=False)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    d = load()
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} counties, "
          f"{d.year.min()}-{d.year.max()}")
    figure_timeseries(d)
    if not a.no_maps:
        figure_maps(d)


if __name__ == "__main__":
    main()
