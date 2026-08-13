# -*- coding: utf-8 -*-
"""
Combined map: TFP in the CHOROPLETH, real agricultural GVP in the SPIKES.

  (A) choropleth = mean annual TFP growth (%/yr)
  (B) choropleth = TFP volatility, sd of annual growth (pp)
  both        = spike height is the county's MEAN real GVP over the panel

Reading the two channels together is the point.  A tall spike on a pale cell is
a large producer whose productivity barely moved; a tall spike on a dark cell is
a large producer that also gained fast; and the forest of short spikes carries
little weight in the national aggregate however extreme its colour.  A
choropleth on its own gives every county equal visual weight regardless of how
much output it actually accounts for, which is exactly what misleads the eye in
western China.

DESIGN NOTES (why it looks the way it does)

  SEQUENTIAL, NOT DIVERGING.  Only ~0.3% of counties have negative mean TFP
  growth, so a red/blue ramp centred on zero would spend half its range on an
  empty tail and leave every real county in one mid-tone.  The job of the colour
  here is MAGNITUDE, so both panels use a single-hue light->dark ramp: blue for
  growth, orange for volatility (the second sequential context takes the next
  hue rather than a second blue).  No rainbow ramp, and no hue at a midpoint.

  CLASSED, NOT CONTINUOUS.  With ~2,000 small polygons a continuous ramp
  compresses almost everything into indistinguishable neighbours -- the previous
  version of this figure was a uniform green wash.  Colour is assigned by
  QUANTILE class instead, so each class holds ~1/6 of counties and the map uses
  its whole range by construction.  The legend prints the actual break values,
  so the classes stay readable as numbers and not just as shades.

  RECESSIVE SPIKES.  1,975 spikes at eastern-China density will smear into a
  black mass and hide the choropleth underneath -- which is the half of the
  figure carrying the productivity signal.  They are therefore hairline-width,
  semi-transparent, shorter than the map is tall, and a single neutral ink that
  stays legible over both ramps without competing with either.  They are drawn
  north-to-south so southern spikes occlude northern ones.

Statistics are derived here from the per-county series rather than read from a
pre-baked table, so this script depends only on the pipeline outputs below.

Inputs : src/clean/dea_malmquist_county.csv   (20_dea_vs_fe_tfp.py)   --source dea
         src/clean/sfa_bc92_county_year.csv   (21_sfa_bc92.do)        --source sfa
         src/clean/county_panel_clean.csv     (03_clean_panel.py)
Output : src/figures/fig_spike_choropleth_tfp[_sfa].png
         src/clean/map_county_tfp[_sfa].csv
Run:  python src/55_spike_choropleth_tfp.py
      python src/55_spike_choropleth_tfp.py --source sfa
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import _common as C
import _tfp as T
from _county_match import load_boundaries, match_counties

ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")

# --- marks -----------------------------------------------------------------
BASE_W_KM = 6.0        # hairline: 1,975 spikes must not become a black mass
MAX_H_KM = 260.0       # shorter than the old 430 so spikes stay inside the map
SPIKE_INK = "#22303f"  # one neutral ink, legible over both ramps
SPIKE_ALPHA = 0.62

# --- sequential ramps (dataviz reference palette; light -> dark) ------------
BLUE6 = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
ORANGE6 = ["#fde3d5", "#fbc3a8", "#f79f76", "#f0784a", "#d95926", "#a8410f"]
NODATA = "#eceae6"     # recedes toward the surface; never a ramp colour
INK, INK2 = "#0b0b0b", "#52514e"


def county_stats(source):
    """Per-county mean annual TFP growth (%/yr) and volatility (sd, pp)."""
    if source == "dea":
        p = os.path.join(C.CLEAN_DIR, "dea_malmquist_county.csv")
        if not os.path.exists(p):
            sys.exit("dea_malmquist_county.csv missing — run 20_dea_vs_fe_tfp.py")
        d = pd.read_csv(p).rename(columns={"lnM": "dln"})[["countyid", "year", "dln"]]
        lab = "DEA sequential-NIRS Malmquist"
    else:
        p = os.path.join(C.CLEAN_DIR, "sfa_bc92_county_year.csv")
        if not os.path.exists(p):
            sys.exit("sfa_bc92_county_year.csv missing — run 21_sfa_bc92.do")
        d = T.county_dln(pd.read_csv(p), "ln_tfp_chen")
        lab = "BC92 SFA (land-normalised, Chen-style TFP)"
    d = d.dropna(subset=["dln"])
    d["grw"] = 100 * np.expm1(d["dln"])
    return d, lab


def quantile_classes(v, k=6):
    """k quantile breaks -> (boundaries, ListedColormap-ready labels)."""
    qs = np.linspace(0, 1, k + 1)
    b = np.unique(np.quantile(v.dropna(), qs))
    return b


def fmt_breaks(b, unit):
    """Class labels, with the two tails written open-ended.

    The extreme classes are set by a handful of counties (top growth class runs
    past +170%/yr, top volatility class past 900 pp), so printing the raw
    endpoint invites the reader to think the class spans that range uniformly.
    "<= x" and ">= x" state the break, which is the meaningful part.
    """
    out = ["\u2264 {:,.1f} {}".format(b[1], unit)]
    for lo, hi in zip(b[1:-2], b[2:-1]):
        out.append("{:,.1f} \u2013 {:,.1f} {}".format(lo, hi, unit))
    out.append("\u2265 {:,.1f} {}".format(b[-2], unit))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dea", "sfa"], default="dea")
    ap.add_argument("--classes", type=int, default=6)
    a = ap.parse_args()

    dln, method = county_stats(a.source)

    panel = pd.read_csv(C.CLEAN_PANEL,
                        usecols=["countyid", "county_name", "year", "real_gvp", "ag_county"])
    panel = panel[panel["ag_county"] == 1]

    g = load_boundaries()
    roster = (panel.dropna(subset=["county_name"])
                   .groupby("countyid", as_index=False)
                   .agg(county_name=("county_name", "last")))
    roster = match_counties(roster, g)

    # ---- choropleth statistics, per boundary polygon ----------------------
    # Several production counties can map to the same polygon (merges/splits),
    # so statistics are averaged over the sources and n_src records how many.
    d = dln.merge(roster[["countyid", "map_code"]], on="countyid", how="inner") \
           .dropna(subset=["map_code"])
    per = d.groupby(["map_code", "countyid"])["grw"].agg(["mean", "std"]).reset_index()
    tfp = (per.groupby("map_code")
              .agg(growth_pct=("mean", "mean"), vol_pp=("std", "mean"),
                   n_src=("countyid", "nunique")).reset_index())
    tfp["map_code"] = tfp.map_code.astype(int)
    suffix = "" if a.source == "dea" else "_sfa"
    tfp.to_csv(os.path.join(C.CLEAN_DIR, f"map_county_tfp{suffix}.csv"), index=False)
    print(f"[{a.source}] {len(tfp):,d} boundary polygons from "
          f"{d.countyid.nunique():,d} agricultural counties")
    print(f"  growth  %/yr : p5 {tfp.growth_pct.quantile(.05):.1f}  "
          f"median {tfp.growth_pct.median():.1f}  p95 {tfp.growth_pct.quantile(.95):.1f}")
    print(f"  volatility pp: p5 {tfp.vol_pp.quantile(.05):.1f}  "
          f"median {tfp.vol_pp.median():.1f}  p95 {tfp.vol_pp.quantile(.95):.1f}")

    # ---- spikes: mean real GVP -------------------------------------------
    pan = panel.merge(roster[["countyid", "map_code"]], on="countyid", how="left")
    gvp = (pan.dropna(subset=["real_gvp", "map_code"]).query("real_gvp > 0")
              .groupby("map_code", as_index=False).real_gvp.mean()
              .rename(columns={"real_gvp": "gvp_mean"}))
    gvp["map_code"] = gvp.map_code.astype(int)

    gm = g.to_crs(ALBERS)
    cent = gm.geometry.representative_point()
    xy = pd.DataFrame({"code": gm.code.values, "x": cent.x.values, "y": cent.y.values})
    sp = gvp.merge(xy, left_on="map_code", right_on="code", how="inner") \
            .sort_values("y", ascending=False)                 # north first
    ref = np.nanpercentile(sp.gvp_mean, 99.5)                  # robust tallest
    scale = (MAX_H_KM * 1000.0) / ref
    w = BASE_W_KM * 1000.0
    verts = [np.array([[x - w / 2, y], [x, y + v * scale], [x + w / 2, y]])
             for x, y, v in zip(sp.x.values, sp.y.values, sp.gvp_mean.values)]
    print(f"  {len(sp):,d} spikes; key top = {ref/1e4:,.1f} bn yuan "
          f"(99.5th pct mean real GVP)")

    gm2 = gm.merge(tfp, left_on="code", right_on="map_code", how="left")
    prov = gm2.dissolve(by=gm2.code // 10000)

    # ---- render -----------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(18.5, 10.4))
    specs = [("growth_pct", "(A)  Mean annual TFP growth", BLUE6, "%/yr"),
             ("vol_pp", "(B)  TFP volatility  (sd of annual growth)", ORANGE6, "pp")]

    for ax, (col, title, ramp, unit) in zip(axes, specs):
        ramp = ramp[:a.classes]
        breaks = quantile_classes(gm2[col], len(ramp))
        cmap = ListedColormap(ramp)
        cmap.set_bad(NODATA)
        norm = BoundaryNorm(breaks, ncolors=len(ramp), clip=True)

        gm2.plot(column=col, ax=ax, cmap=cmap, norm=norm, linewidth=0,
                 missing_kwds=dict(color=NODATA), zorder=1)
        prov.boundary.plot(ax=ax, color="white", linewidth=.5, zorder=2)
        ax.add_collection(PolyCollection(verts, facecolors=SPIKE_INK,
                                         edgecolors="none", linewidths=0,
                                         alpha=SPIKE_ALPHA, zorder=3))
        ax.set_title(title, fontsize=13.5, loc="left", fontweight="bold",
                     color=INK, pad=10)
        ax.set_axis_off(); ax.autoscale_view()

        # discrete legend: the class breaks, as numbers
        handles = [Patch(facecolor=c, edgecolor="white", linewidth=.6, label=l)
                   for c, l in zip(ramp, fmt_breaks(breaks, unit))]
        handles.append(Patch(facecolor=NODATA, edgecolor="white", linewidth=.6,
                             label="no data"))
        leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                        framealpha=.94, edgecolor="#d8d6d1", fontsize=8.6,
                        title=f"equal-count classes  ({unit})", title_fontsize=9,
                        borderpad=.7, labelspacing=.42,
                        bbox_to_anchor=(0.005, 0.005))
        leg.get_title().set_color(INK2)
        for t in leg.get_texts():
            t.set_color(INK2)

    # ---- spike-height key, boxed, on the right-hand panel -----------------
    # SPIKE-HEIGHT KEY.  Drawn on its OWN figure-level axes between the two
    # panels rather than inside a map: every in-map placement tried either
    # collided with the coastline or sat on a white plate covering real
    # counties (Xinjiang, in the north-west corner).  A key must never hide
    # the data it explains.
    kax = fig.add_axes([0.425, 0.020, 0.155, 0.135])
    kax.set_axis_off()
    kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    kax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=kax.transAxes,
                                facecolor="white", edgecolor="#d8d6d1",
                                linewidth=.6, zorder=0))
    kax.text(0.06, 0.92, "spike height", fontsize=8.8, color=INK,
             fontweight="bold", va="top")
    kax.text(0.06, 0.74, "mean real agricultural GVP\n1981-2016, 2005 prices",
             fontsize=7.9, color=INK2, va="top", linespacing=1.35)
    bx = 0.14
    for frac in (1.0, 0.5, 0.25):
        hh = 0.30 * frac
        kax.add_patch(plt.Polygon([[bx - 0.022, 0.06], [bx, 0.06 + hh],
                                   [bx + 0.022, 0.06]],
                                  facecolor=SPIKE_INK, edgecolor="none",
                                  alpha=SPIKE_ALPHA, zorder=2))
        kax.text(bx + 0.045, 0.06 + hh, "{:,.0f} bn".format(frac * ref / 1e4),
                 fontsize=7.9, va="center", color=INK2)
        bx += 0.30

    fig.suptitle("China county agriculture: productivity in colour, output in height",
                 fontsize=15.5, fontweight="bold", color=INK, y=0.975)
    fig.text(0.5, 0.937,
             f"choropleth = {method};  spike = mean real agricultural GVP   |   "
             f"{len(tfp):,d} agricultural counties, 1981–2016",
             ha="center", fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0.16, 1, 0.925))
    p = os.path.join(C.FIG_DIR, f"fig_spike_choropleth_tfp{suffix}.png")
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)

    # ---- what the two channels together reveal ----------------------------
    m = tfp.merge(gvp, on="map_code", how="inner")
    hi = m[m.gvp_mean >= m.gvp_mean.quantile(.9)]
    lo = m[m.gvp_mean <= m.gvp_mean.quantile(.1)]
    print(f"\ntop-decile producers by mean GVP : TFP growth {hi.growth_pct.mean():+.2f}%/yr, "
          f"volatility {hi.vol_pp.mean():.1f} pp")
    print(f"bottom-decile producers          : TFP growth {lo.growth_pct.mean():+.2f}%/yr, "
          f"volatility {lo.vol_pp.mean():.1f} pp")
    print(f"corr(mean GVP, TFP growth) = {m.gvp_mean.corr(m.growth_pct):+.3f}   "
          f"corr(mean GVP, volatility) = {m.gvp_mean.corr(m.vol_pp):+.3f}")


if __name__ == "__main__":
    main()
