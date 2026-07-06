# -*- coding: utf-8 -*-
"""
3-D production frontier / isoquant surface (extends Panel B to 3-D).
  vertical axis  = ln output (adjusted: the two NON-shown inputs are partialled
                   out at the frontier elasticities, so the surface is a clean
                   function of the two displayed inputs)
  two floor axes = ln Labour  and  ln (second input)
  floor contours = ISOQUANTS (constant output) -> downward curves = substitution
  surface        = smooth concave best-practice frontier (quadratic fit to the
                   upper-quantile of output within a 2-D input grid)
Two panels compare labour with Capital and with Intermediate.  Year 2005.
Output: src/figures/fig_io_3d_frontier.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
import _common as C

BETA = {"Laborday_impute": 0.47, "Land_serv_q": 0.015, "capital_serv_q": 0.07, "Inter_all_real": 0.45}
YEAR = 2005
LAB = {"Laborday_impute": "Labour (man-days)", "capital_serv_q": "Capital service",
       "Inter_all_real": "Intermediate input", "Land_serv_q": "Land service"}

m = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "year", "real_gvp"] + list(BETA))
m = m.dropna().pipe(lambda d: d[(d[["real_gvp"] + list(BETA)] > 0).all(axis=1)])
d = m[m.year == YEAR].copy()
print(f"{YEAR}: {len(d)} counties")

def frontier_surface(x1, x2, z, q=0.85, nb=9):
    ex = np.linspace(np.quantile(x1, .02), np.quantile(x1, .98), nb + 1)
    ey = np.linspace(np.quantile(x2, .02), np.quantile(x2, .98), nb + 1)
    ix, iy = np.digitize(x1, ex), np.digitize(x2, ey)
    cx, cy, cz = [], [], []
    for a in range(1, nb + 1):
        for b in range(1, nb + 1):
            mask = (ix == a) & (iy == b)
            if mask.sum() >= 6:
                cx.append(.5*(ex[a-1]+ex[a])); cy.append(.5*(ey[b-1]+ey[b])); cz.append(np.quantile(z[mask], q))
    cx, cy, cz = map(np.array, (cx, cy, cz))
    # separable quadratic with the two curvature terms constrained <=0 -> CONCAVE
    # (diminishing returns), consistent with Panel A's concave frontier.
    A = np.column_stack([np.ones_like(cx), cx, cy, cx**2, cy**2])
    lo = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    hi = np.array([ np.inf,  np.inf,  np.inf, 0.0, 0.0])
    coef = lsq_linear(A, cz, bounds=(lo, hi)).x
    return coef, (ex[0], ex[-1]), (ey[0], ey[-1])

def surf(coef, X, Y):
    return coef[0] + coef[1]*X + coef[2]*Y + coef[3]*X**2 + coef[4]*Y**2

fig = plt.figure(figsize=(15, 6.5))
pairs = [("capital_serv_q", ["Land_serv_q", "Inter_all_real"]),
         ("Inter_all_real", ["Land_serv_q", "capital_serv_q"])]
for i, (x2v, others) in enumerate(pairs):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    x1 = np.log(d.Laborday_impute).values
    x2 = np.log(d[x2v]).values
    # adjusted output: strip the two inputs NOT shown, at frontier elasticities
    zadj = np.log(d.real_gvp).values - sum(BETA[o]*np.log(d[o]).values for o in others)
    coef, xr, yr = frontier_surface(x1, x2, zadj)
    XX, YY = np.meshgrid(np.linspace(*xr, 40), np.linspace(*yr, 40))
    ZZ = surf(coef, XX, YY)
    # scatter (light), frontier surface, isoquant contours on the floor
    ax.scatter(x1, x2, zadj, s=3, alpha=.08, color="0.4")
    ax.plot_surface(XX, YY, ZZ, cmap=cm.viridis, alpha=.55, linewidth=0, antialiased=True, rstride=2, cstride=2)
    zfloor = np.floor(np.quantile(zadj, .02))
    ax.contour(XX, YY, ZZ, zdir="z", offset=zfloor, levels=9, cmap=cm.autumn, linewidths=1.3)
    ax.set_zlim(bottom=zfloor)
    ax.set_xlabel("\nln " + LAB["Laborday_impute"], fontsize=8)
    ax.set_ylabel("\nln " + LAB[x2v], fontsize=8)
    ax.set_zlabel("adj. ln Output", fontsize=8)
    ax.set_title(f"Labour × {LAB[x2v]} → Output\nsurface = frontier; floor curves = isoquants (substitution)", fontsize=9)
    ax.view_init(elev=22, azim=-62)
    ax.tick_params(labelsize=6.5)

fig.suptitle(f"China county agriculture ({YEAR}): 3-D production frontier — output vs two inputs, "
             f"floor contours = isoquants", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(C.FIG_DIR, "fig_io_3d_frontier.png")
fig.savefig(out, dpi=160); plt.close(fig)
print("saved:", out)
