# -*- coding: utf-8 -*-
"""
Descriptive 3D view of one-input / one-weather / output, cross-section of counties.
Axes: x = GDD (growing-season degree days), y = Labour (man-days), z = Output (real GVP).
A line connects all counties (ordered by labour); dotted lines show the projections
onto the GDD-Labour plane (floor) and the Labour-Output plane (back wall).
NOTE: climate (GDD) starts in 1982, so 1981 is unavailable -> using 1982.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

YEAR = 1982
d = pd.read_csv("data/climate_tfp_irrigation_panel_1982_2015.csv",
                usecols=["year","countyid","gdd_growing_season","Laborday_impute",
                         "GVP_allagr_impute","PPP_2005"])
d = d[d.year == YEAR].dropna()
d = d[(d.Laborday_impute > 0) & (d.GVP_allagr_impute > 0)]
d["output"] = d["GVP_allagr_impute"] / d["PPP_2005"]      # real GVP, 10k RMB
d = d.sort_values("Laborday_impute").reset_index(drop=True)

g = d["gdd_growing_season"].values
L = d["Laborday_impute"].values / 1e6       # million man-days (readable)
Y = d["output"].values / 1e4                # 10^8 RMB (= output/1e4 since GVP in 1e4 RMB)

gmin, ymin = g.min(), 0.0                    # back wall at min GDD, floor at output=0

fig = plt.figure(figsize=(11, 8.5))
ax = fig.add_subplot(111, projection="3d")

# scatter of counties
ax.scatter(g, L, Y, s=10, c=Y, cmap="viridis", depthshade=True, alpha=0.7)
# line connecting all counties (ordered by labour)
ax.plot(g, L, Y, color="0.35", lw=0.6, alpha=0.7, label=f"counties, {YEAR} (ordered by labour)")
# projection onto GDD-Labour plane (floor, output=0): dotted
ax.plot(g, L, np.full_like(Y, ymin), ":", color="tab:red", lw=0.9, alpha=0.8,
        label="proj. onto GDD-Labour plane (floor)")
# projection onto Labour-Output plane (back wall, GDD=min): dotted
ax.plot(np.full_like(g, gmin), L, Y, ":", color="tab:blue", lw=0.9, alpha=0.8,
        label="proj. onto Labour-Output plane (wall)")

ax.set_xlabel("GDD (growing-season degree days)")
ax.set_ylabel("Labour (million man-days)")
ax.set_zlabel("Output: real GVP (1e8 RMB)")
ax.set_title(f"County agricultural output vs labour & GDD — {YEAR} cross-section "
             f"({len(d)} counties)", fontweight="bold")
ax.view_init(elev=22, azim=-60)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("src/figures/fig_3d_io_1982.png", dpi=150)
print("saved src/figures/fig_3d_io_1982.png ; counties:", len(d),
      "| corr(lnL,lnY)=%.2f" % np.corrcoef(np.log(L), np.log(Y))[0,1],
      "| corr(GDD,lnY)=%.2f" % np.corrcoef(g, np.log(Y))[0,1])
