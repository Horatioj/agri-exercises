# -*- coding: utf-8 -*-
"""
Combined map: TFP in the CHOROPLETH, real agricultural GVP in the SPIKES.

  (A) choropleth = mean annual DEA-Malmquist TFP growth (%/yr)
  (B) choropleth = TFP volatility, sd of annual growth (pp)
  both        = spike height is the county's MEAN real GVP, 1981-2016

Reading the two channels together is the point.  A tall spike on a pale cell is
a large producer whose productivity barely moved; a tall spike on a dark cell is
a large producer that also gained fast; and the forest of short spikes carries
little weight in the national aggregate however extreme its colour.  A
choropleth on its own gives every county equal visual weight regardless of how
much output it actually accounts for, which is exactly what misleads the eye in
western China.

Spike height is the PERIOD MEAN GVP rather than one year, because the
choropleth statistics are themselves period statistics.

Spikes are a single dark neutral so they stay legible over both the diverging
(growth) and sequential (volatility) ramps, and are drawn north-to-south so
southern spikes occlude northern ones.

(Distinct from 30_spike_map_tfp.py, where the SPIKE encodes TFP itself.)

Inputs : src/clean/map_county_tfp.csv (27_map_tfp.py), src/clean/county_panel_clean.csv
Output : src/figures/fig_spike_choropleth_tfp.png
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
import _common as C
from _county_match import load_boundaries, match_counties

ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")
BASE_W_KM = 8.0
MAX_H_KM = 430.0
SPIKE_F, SPIKE_E = "#22303f", "#0b1420"      # dark neutral, legible over both ramps

tfp = pd.read_csv(os.path.join(C.CLEAN_DIR, "map_county_tfp.csv"))
panel = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "county_name", "year", "real_gvp"])

g = load_boundaries()
roster = panel.dropna(subset=["county_name"]).groupby("countyid", as_index=False) \
              .agg(county_name=("county_name", "last"))
roster = match_counties(roster, g)
panel = panel.merge(roster[["countyid", "map_code"]], on="countyid", how="left")
gvp = (panel.dropna(subset=["real_gvp", "map_code"]).query("real_gvp > 0")
            .groupby("map_code", as_index=False).real_gvp.mean()
            .rename(columns={"real_gvp": "gvp_mean"}))
gvp["map_code"] = gvp.map_code.astype(int)

gm = g.to_crs(ALBERS)
cent = gm.geometry.representative_point()
xy = pd.DataFrame({"code": gm.code.values, "x": cent.x.values, "y": cent.y.values})
sp = gvp.merge(xy, left_on="map_code", right_on="code", how="inner") \
        .sort_values("y", ascending=False)                      # north first
ref = np.nanpercentile(sp.gvp_mean, 99.5)                       # robust tallest
scale = (MAX_H_KM * 1000.0) / ref
w = BASE_W_KM * 1000.0
verts = [np.array([[x - w / 2, y], [x, y + v * scale], [x + w / 2, y]])
         for x, y, v in zip(sp.x.values, sp.y.values, sp.gvp_mean.values)]
print(f"{len(sp):,d} spikes; key top = {ref/1e4:,.1f} bn yuan (99.5th pct mean real GVP)")

gm2 = gm.merge(tfp, left_on="code", right_on="map_code", how="left")
prov = gm2.dissolve(by=gm2.code // 10000)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.6))
specs = [("growth_pct", "(A) Mean annual TFP growth", "RdYlGn", True, "TFP growth (%/yr)", "%.2f"),
         ("vol_pp", "(B) TFP volatility (sd of annual growth)", "magma_r", False,
          "sd of annual growth (pp)", "%.1f")]

for ax, (col, title, cmap, diverge, clab, fmt) in zip(axes, specs):
    v = gm2[col].dropna()
    lo, hi = np.percentile(v, [2, 98])
    if diverge:
        m = max(abs(lo), abs(hi)); norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
    else:
        norm = plt.Normalize(vmin=lo, vmax=hi)
    gm2.plot(column=col, ax=ax, cmap=cmap, norm=norm, linewidth=0,
             missing_kwds=dict(color="0.92"), zorder=1)
    prov.boundary.plot(ax=ax, color="white", linewidth=.45, zorder=2)
    ax.add_collection(PolyCollection(verts, facecolors=SPIKE_F, edgecolors=SPIKE_E,
                                     linewidths=.3, alpha=.70, zorder=3))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=.5, pad=.01)
    cb.set_label(clab, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_axis_off(); ax.autoscale_view()
    ax.annotate(f"median {fmt % np.median(v)}   |   grey = no data",
                (0.01, 0.02), xycoords="axes fraction", fontsize=8, color="0.3")

# spike-height key (left panel)
axl = axes[0]
xr_, yr_ = axl.get_xlim(), axl.get_ylim()
x0 = xr_[0] + 0.05 * (xr_[1] - xr_[0]); y0 = yr_[0] + 0.11 * (yr_[1] - yr_[0])
for frac in (1.0, 0.5, 0.25):
    hh = frac * MAX_H_KM * 1000.0
    axl.add_collection(PolyCollection(
        [np.array([[x0 - w / 2, y0], [x0, y0 + hh], [x0 + w / 2, y0]])],
        facecolors=SPIKE_F, edgecolors=SPIKE_E, linewidths=.3, alpha=.70, zorder=5))
    axl.text(x0 + 14000, y0 + hh, f"{frac*ref/1e4:,.1f} bn", fontsize=7.5,
             va="center", color="0.2", zorder=6)
    x0 += 84000
axl.text(xr_[0] + 0.05 * (xr_[1] - xr_[0]), y0 - 62000,
         "spike height = mean real GVP, 1981–2016 (2005 prices)",
         fontsize=8, color="0.2")

fig.suptitle("China county agriculture: TFP in colour, output in height\n"
             "choropleth = DEA sequential-NIRS Malmquist TFP; spike = mean real agricultural GVP",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
p = os.path.join(C.FIG_DIR, "fig_spike_choropleth_tfp.png")
fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
print("saved:", p)

# what the two channels together reveal
j = tfp.merge(gvp, on="map_code", how="inner")
big = j.nlargest(max(1, int(len(j) * .10)), "gvp_mean")
small = j.nsmallest(max(1, int(len(j) * .10)), "gvp_mean")
print(f"\ntop-decile producers by mean GVP : TFP growth {big.growth_pct.mean():+.2f}%/yr, "
      f"volatility {big.vol_pp.mean():.1f} pp")
print(f"bottom-decile producers          : TFP growth {small.growth_pct.mean():+.2f}%/yr, "
      f"volatility {small.vol_pp.mean():.1f} pp")
print(f"corr(mean GVP, TFP growth) = {j.gvp_mean.corr(j.growth_pct):+.3f}   "
      f"corr(mean GVP, volatility) = {j.gvp_mean.corr(j.vol_pp):+.3f}")
