# -*- coding: utf-8 -*-
"""
Per-year input-output relationships (cross-section of counties), 4 inputs.
For each input: ln(output) vs ln(input), RAW logs (so each year's level shows).
 - 1981 counties as a scatter cloud.
 - OLS fit line for several years (1981/1995/2005/2015) -> shows the relationship
   AND its outward shift over time (technical progress).
 - Upper-envelope of 1981 (max ln Y per ln X bin) = the 'best-practice frontier'
   that DEA would wrap -> a preview of the DEA step.
Why a FITTED line, not a line connecting county points: counties have no natural
order, so connecting raw points just zig-zags and conveys nothing; the OLS line is
the average relationship, the upper envelope is the frontier.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import _common as C

C.set_cjk_font(plt)
np.seterr(divide="ignore", invalid="ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})

Y = "real_gvp"
INPUTS = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
YEARS = [1981, 1995, 2005, 2015]
SCATTER_YEAR = 1981
cmap = plt.cm.viridis(np.linspace(0, 0.85, len(YEARS)))

m = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "year", Y] + INPUTS)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, xv in zip(axes.ravel(), INPUTS):
    # ---- 1981 scatter ----
    s0 = m[m.year == SCATTER_YEAR].copy()
    s0 = s0[(s0[Y] > 0) & (s0[xv] > 0)]
    x0, y0 = np.log(s0[xv].values), np.log(s0[Y].values)
    ax.scatter(x0, y0, s=6, alpha=0.18, color="0.55", label=f"{SCATTER_YEAR} counties")

    # ---- OLS fit per year ----
    xs = np.linspace(np.percentile(x0, 1), np.percentile(x0, 99), 100)
    for col, yr in zip(cmap, YEARS):
        s = m[m.year == yr]; s = s[(s[Y] > 0) & (s[xv] > 0)]
        lx, ly = np.log(s[xv].values), np.log(s[Y].values)
        b1, b0 = np.polyfit(lx, ly, 1)
        ax.plot(xs, b0 + b1 * xs, color=col, lw=2,
                label=f"{yr} fit (slope {b1:.2f})")

    # ---- 1981 upper-envelope (DEA-style best-practice frontier) ----
    fr, edges, _ = binned_statistic(x0, y0, statistic="max", bins=18)
    cen = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(fr)
    ax.plot(cen[ok], fr[ok], "--", color="crimson", lw=1.6, label=f"{SCATTER_YEAR} upper envelope (DEA)")

    ax.set_xlabel(f"ln {C.VLAB[xv]}")
    ax.set_ylabel("ln Real GVP (output)")
    ax.set_title(f"Output vs {C.VLAB[xv]} — cross-section by year")
    ax.legend(fontsize=6.5, loc="upper left")

fig.suptitle("China county agriculture: input-output relationship per year "
             "(cleaned panel) — fits shift up over time = technical progress",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
# DEPRECATED single-input view (straight fits + jagged envelope). Superseded by
# 20_io_frontier.py (aggregate input + concave frontier + isoquants). Kept for
# reference only; writes to a distinct name so it can't clobber fig_io_byyear.png.
out = os.path.join(C.FIG_DIR, "fig_io_byyear_singleinput_DEPRECATED.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("saved:", out)

# quick numeric: bivariate slope drift over time for labour
print("\nBivariate ln-ln slope (NOT the true elasticity — single-input view):")
for xv in INPUTS:
    row = []
    for yr in YEARS:
        s = m[m.year == yr]; s = s[(s[Y] > 0) & (s[xv] > 0)]
        b1 = np.polyfit(np.log(s[xv]), np.log(s[Y]), 1)[0]
        row.append(f"{yr}:{b1:.2f}")
    print(f"  {C.VLAB[xv]:22s} " + "  ".join(row))
