# -*- coding: utf-8 -*-
"""
Isoquant maps that NEST cleanly by year, for several input pairs, plus a 3-D
isoquant surface.

Parametric Cobb-Douglas isoquants (fixed frontier elasticities -> smooth, and
they nest by construction): for a fixed output Y* and the OTHER inputs at their
medians,
      beta_i ln X_i + beta_j ln X_j = ln Y* - A_t - sum_other beta_o ln X_o(med)
where A_t = that year's FRONTIER technology (90th-pct Solow residual, rising over
time).  A_t up over years -> isoquant shifts toward the origin = technical
progress; the downward-convex shape = input substitution.

Outputs: src/figures/fig_isoquants_pairs.png (2-D pairs + 3-D surface)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import _common as C

C.set_cjk_font(plt)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

INP = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
B   = {"Laborday_impute": 0.47, "Land_serv_q": 0.015, "capital_serv_q": 0.07, "Inter_all_real": 0.45}
LAB = {"Laborday_impute": "Labour (man-days)", "Land_serv_q": "Land service",
       "capital_serv_q": "Capital service", "Inter_all_real": "Intermediate (real)"}
YEARS = [1981, 1990, 2000, 2010, 2015]
reds  = ["#fdae61", "#f46d43", "#d73027", "#b2182b", "#67001f"]

d = pd.read_csv(C.CLEAN_PANEL, usecols=["year", "real_gvp"] + INP)
d = d.dropna(subset=["real_gvp"] + INP); d = d[(d[["real_gvp"] + INP] > 0).all(axis=1)].copy()
lnY = np.log(d.real_gvp)
res = lnY - sum(B[k] * np.log(d[k]) for k in B)
A = {yr: res[d.year == yr].quantile(0.90) for yr in YEARS}   # frontier technology by year
med_ln = {k: np.log(d[k]).median() for k in INP}
lnYstar = lnY.median()
rng = {k: (np.log(d[k]).quantile(.08), np.log(d[k]).quantile(.92)) for k in INP}

def iso_pair(ax, xi, xj):
    others = [k for k in INP if k not in (xi, xj)]
    lnxi = np.linspace(*rng[xi], 100)
    for yr, col in zip(YEARS, reds):
        const = lnYstar - A[yr] - sum(B[o] * med_ln[o] for o in others)
        lnxj = (const - B[xi] * lnxi) / B[xj]
        ax.plot(np.exp(lnxi), np.exp(lnxj), "-", color=col, lw=2, label=str(yr))
    ax.set_xscale("log"); ax.set_yscale("log")               # readable for all elasticity ratios
    ax.set_xlim(np.exp(rng[xi][0]), np.exp(rng[xi][1]))
    lnxj_all = (lnYstar - A[YEARS[0]] - sum(B[o]*med_ln[o] for o in others) - B[xi]*lnxi) / B[xj]
    ax.set_ylim(np.exp(lnxj_all.min()) * .4, np.exp(lnxj_all.max()) * 2.5)
    mrts = -B[xi] / B[xj]
    ax.set_xlabel(LAB[xi]); ax.set_ylabel(LAB[xj])
    ax.set_title(f"{LAB[xi].split(' ')[0]}–{LAB[xj].split(' ')[0]}  (slope dln{LAB[xj][0]}/dln{LAB[xi][0]}={mrts:.1f})", fontsize=9)
    ax.grid(alpha=.25, which="both")

fig = plt.figure(figsize=(13, 9))
ax1 = fig.add_subplot(2, 2, 1); iso_pair(ax1, "Laborday_impute", "Inter_all_real")
ax2 = fig.add_subplot(2, 2, 2); iso_pair(ax2, "Laborday_impute", "capital_serv_q")
ax3 = fig.add_subplot(2, 2, 3); iso_pair(ax3, "capital_serv_q", "Inter_all_real")
ax1.legend(title="year", fontsize=7.5, ncol=2)

# ---- 3-D isoquant surface: Labour x Capital x Intermediate (Land at median) ---
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
xi, xj, xk = "Laborday_impute", "capital_serv_q", "Inter_all_real"
gL = np.linspace(*rng[xi], 24); gK = np.linspace(*rng[xj], 24)
LL, KK = np.meshgrid(gL, gK)
for yr, cmap, cc in [(1981, "autumn", "#f46d43"), (2015, "winter", "#2166ac")]:
    const = lnYstar - A[yr] - B["Land_serv_q"] * med_ln["Land_serv_q"]
    lnM = (const - B[xi] * LL - B[xj] * KK) / B[xk]          # isoquant surface (log intermediate)
    ax4.plot_surface(LL, KK, lnM, alpha=.55, color=cc, linewidth=0, antialiased=True)
    ax4.plot([], [], [], color=cc, lw=6, label=str(yr))
ax4.set_xlabel("ln Labour"); ax4.set_ylabel("ln Capital"); ax4.set_zlabel("ln Intermediate")
ax4.set_title("3-D isoquant surface (fixed output)\nsurface drops over time = tech progress", fontsize=9)
ax4.legend(title="year", fontsize=8, loc="upper left")
ax4.view_init(elev=22, azim=-60)

fig.suptitle("China county agriculture — input-substitution isoquants by year (nested = technical progress)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(C.FIG_DIR, "fig_isoquants_pairs.png")
fig.savefig(out, dpi=150); plt.close(fig)
print("saved:", out)
print("A_t (frontier tech):", {y: round(A[y], 2) for y in YEARS}, " lnY* =", round(lnYstar, 2))
