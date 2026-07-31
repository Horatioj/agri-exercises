# -*- coding: utf-8 -*-
"""
Spike map of county real agricultural GVP (ESRI-style spikes, academic palette).

Each county is drawn as a narrow isoceles triangle rising from its centroid,
height proportional to real GVP.  Spikes are drawn back-to-front (north first)
so southern spikes correctly occlude northern ones, which is what gives the
technique its depth.  Height is LINEAR in GVP -- the point of a spike map is to
show concentration honestly, and a sqrt/log height would flatten exactly the
disparity worth seeing.

Two panels, same height scale, so the growth of output is directly readable:
1981 versus 2015.

Palette is deliberately restrained (single muted slate-blue hue, low-alpha fill,
thin darker stroke) rather than a rainbow: value is encoded by HEIGHT, so colour
only needs to separate the spikes from the basemap.

Output: src/figures/fig_spike_map_gvp.png
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
import _common as C
from _county_match import load_boundaries, match_counties

ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")

YEARS = (1981, 2015)
BASE_W_KM = 9.0           # spike base width
MAX_H_KM = 620.0          # height of the tallest spike in the LATER year
FILL = "#3b6ea5"          # muted slate blue
EDGE = "#1d3f63"

panel = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "county_name", "year", "real_gvp"])
g = load_boundaries()
roster = panel.dropna(subset=["county_name"]).groupby("countyid", as_index=False) \
              .agg(county_name=("county_name", "last"))
roster = match_counties(roster, g)
panel = panel.merge(roster[["countyid", "map_code"]], on="countyid", how="left")

gm = g.to_crs(ALBERS)
cent = gm.geometry.representative_point()          # inside the polygon, unlike centroid
xy = pd.DataFrame({"code": gm.code.values, "x": cent.x.values, "y": cent.y.values})

# common height scale: set by the LATER year so the two panels are comparable
ref = panel[(panel.year == YEARS[1])].real_gvp.max()
scale = (MAX_H_KM * 1000.0) / ref
print(f"height scale: {MAX_H_KM:.0f} km per {ref:,.0f} (10k yuan, 2005 prices)")

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(17, 8.4))

for ax, yr in zip(axes, YEARS):
    d = panel[panel.year == yr].dropna(subset=["real_gvp", "map_code"]).copy()
    d["map_code"] = d.map_code.astype(int)
    d = d.merge(xy, left_on="map_code", right_on="code", how="inner")
    d = d[d.real_gvp > 0]

    # basemap: county outlines in the sample, province borders above them
    gm.plot(ax=ax, color="#f2f2f0", edgecolor="none", zorder=1)
    sub = gm[gm.code.isin(d.map_code)]
    sub.plot(ax=ax, color="#e4e6e3", edgecolor="none", zorder=2)
    gm.dissolve(by=gm.code // 10000).boundary.plot(
        ax=ax, color="white", linewidth=.5, zorder=3)

    # spikes, drawn north -> south so nearer (southern) spikes overlap farther ones
    d = d.sort_values("y", ascending=False)
    w = BASE_W_KM * 1000.0
    h = d.real_gvp.values * scale
    verts = [np.array([[x - w / 2, y], [x, y + hh], [x + w / 2, y]])
             for x, y, hh in zip(d.x.values, d.y.values, h)]
    ax.add_collection(PolyCollection(
        verts, facecolors=FILL, edgecolors=EDGE, linewidths=.35,
        alpha=.62, zorder=4))

    ax.set_title(f"{yr}", fontsize=13, loc="left", fontweight="bold")
    ax.set_axis_off()
    ax.autoscale_view()
    tot = d.real_gvp.sum() / 1e4
    ax.annotate(f"{len(d):,d} counties · total {tot:,.0f} billion yuan (2005 prices)",
                (0.01, 0.03), xycoords="axes fraction", fontsize=8.5, color="0.35")

# shared height legend on the right panel
axr = axes[1]
x0, y0 = axr.get_xlim()[0] + 0.06 * (axr.get_xlim()[1] - axr.get_xlim()[0]), axr.get_ylim()[0] + 0.10 * (axr.get_ylim()[1] - axr.get_ylim()[0])
for frac in (1.0, 0.5, 0.25):
    hh = frac * MAX_H_KM * 1000.0
    axr.add_collection(PolyCollection(
        [np.array([[x0 - BASE_W_KM*500, y0], [x0, y0 + hh], [x0 + BASE_W_KM*500, y0]])],
        facecolors=FILL, edgecolors=EDGE, linewidths=.35, alpha=.62, zorder=6))
    axr.text(x0 + 14000, y0 + hh, f"{frac*ref/1e4:,.0f} bn",
             fontsize=7.5, va="center", color="0.25", zorder=7)
    x0 += 78000
axr.text(axr.get_xlim()[0] + 0.06 * (axr.get_xlim()[1] - axr.get_xlim()[0]),
         y0 - 60000, "spike height = real agricultural GVP",
         fontsize=8, color="0.25")

fig.suptitle("China county agricultural output, 1981 vs 2015 — spike height is real GVP "
             "(2005 prices, common scale)", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
p = os.path.join(C.FIG_DIR, "fig_spike_map_gvp.png")
fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
print("saved:", p)
