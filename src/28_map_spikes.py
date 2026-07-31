# -*- coding: utf-8 -*-
"""
Map where the cleaning had to intervene on real GVP.

A "spike" here is a county-year the cleaner flagged as a spike-and-revert and
replaced by the provincial-trend imputation (anomaly_log.csv, action=impute).
Mapping them answers a question the summary tables cannot: is bad reporting
CONCENTRATED in particular provinces -- in which case any TFP result driven by
those places is suspect -- or scattered at random, in which case it is noise the
cleaning absorbs.

Panels
  (A) count of imputed real_gvp county-years per county (0-36)
  (B) same as a share of the county's own years, all five I-O variables pooled
  (C) when: national count of imputed real_gvp county-years by year

Inputs : src/clean/anomaly_log.csv, src/clean/county_panel_clean.csv
Output : src/figures/fig_map_spikes.png, src/clean/county_spike_counts.csv
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C
from _county_match import load_boundaries, match_counties

ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")

alog = pd.read_csv(os.path.join(C.CLEAN_DIR, "anomaly_log.csv"))
alog = alog[alog["action"] == "impute"]
panel = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "county_name", "year"])

nyears = panel.groupby("countyid").year.nunique().rename("n_years")
gvp = alog[alog["var"] == "real_gvp"].groupby("countyid").size().rename("gvp_spikes")
allv = alog.groupby("countyid").size().rename("all_spikes")

cty = pd.concat([nyears, gvp, allv], axis=1).reset_index()
cty[["gvp_spikes", "all_spikes"]] = cty[["gvp_spikes", "all_spikes"]].fillna(0)
cty["gvp_spike_share"] = cty.gvp_spikes / cty.n_years
cty["all_spike_share"] = cty.all_spikes / (cty.n_years * len(C.IO_VARS))

roster = panel.dropna(subset=["county_name"]).groupby("countyid", as_index=False) \
              .agg(county_name=("county_name", "last"))
g = load_boundaries()
roster = match_counties(roster, g)
cty = cty.merge(roster[["countyid", "map_code"]], on="countyid", how="left")
cty.to_csv(os.path.join(C.CLEAN_DIR, "county_spike_counts.csv"), index=False)

print(f"counties: {len(cty):,d}; with >=1 imputed real_gvp year: "
      f"{int((cty.gvp_spikes>0).sum()):,d} ({100*(cty.gvp_spikes>0).mean():.1f}%)")
print(f"  imputed real_gvp county-years: {int(cty.gvp_spikes.sum()):,d}")
cty["prov"] = cty.countyid // 10000
PN = {11:"北京",12:"天津",13:"河北",14:"山西",15:"内蒙古",21:"辽宁",22:"吉林",23:"黑龙江",
      31:"上海",32:"江苏",33:"浙江",34:"安徽",35:"福建",36:"江西",37:"山东",41:"河南",
      42:"湖北",43:"湖南",44:"广东",45:"广西",46:"海南",50:"重庆",51:"四川",52:"贵州",
      53:"云南",54:"西藏",61:"陕西",62:"甘肃",63:"青海",64:"宁夏",65:"新疆"}
pv = cty.groupby("prov").agg(n=("countyid","size"), share=("gvp_spike_share","mean")).dropna()
pv = pv[pv.n >= 8].sort_values("share", ascending=False)
print("\nprovinces with the MOST imputed real_gvp (share of county-years):")
for p, r in pv.head(6).iterrows():
    print(f"  {PN.get(p,p):8s} n={int(r.n):4d}  {100*r.share:5.2f}%")
print("least:")
for p, r in pv.tail(4).iterrows():
    print(f"  {PN.get(p,p):8s} n={int(r.n):4d}  {100*r.share:5.2f}%")

gm = g.merge(cty.dropna(subset=["map_code"]).assign(map_code=lambda d: d.map_code.astype(int)),
             left_on="code", right_on="map_code", how="left").to_crs(ALBERS)
prov = gm.dissolve(by=gm.code // 10000)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
fig = plt.figure(figsize=(16, 7.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=.22)
for k, (col, title, cmap, vmax) in enumerate([
        ("gvp_spikes", "(A) Imputed real-GVP county-years (count)", "OrRd", 6),
        ("all_spike_share", "(B) All I-O variables: share of cells imputed", "OrRd", 0.06)]):
    ax = fig.add_subplot(gs[0, k])
    gm.plot(column=col, ax=ax, cmap=cmap, vmin=0, vmax=vmax, linewidth=0,
            missing_kwds=dict(color="0.9"))
    prov.boundary.plot(ax=ax, color="white", linewidth=.35)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=.5, pad=.01); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10); ax.set_axis_off()
    v = gm[col].dropna()
    ax.annotate(f"median {v.median():.3g} | grey = no data",
                (0.01, 0.02), xycoords="axes fraction", fontsize=7.5, color="0.35")

ax3 = fig.add_subplot(gs[0, 2])
byyear = alog[alog["var"] == "real_gvp"].groupby("year").size()
ax3.bar(byyear.index, byyear.values, color="#d7301f", width=.8)
ax3.set_title("(C) Imputed real-GVP county-years by year\n"
              "clustered in 1983 and 2001-03 = caliber breaks, not places",
              fontsize=10)
ax3.set_xlabel("year"); ax3.set_ylabel("counties imputed"); ax3.grid(alpha=.3, axis="y")

fig.suptitle("Where the cleaning intervened: spike-and-revert imputations in the agricultural-county panel",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
p = os.path.join(C.FIG_DIR, "fig_map_spikes.png")
fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
print("\nsaved:", p)
