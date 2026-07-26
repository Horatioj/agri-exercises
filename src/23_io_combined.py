# -*- coding: utf-8 -*-
"""
ONE figure (consolidates the io surface/isoquant/projection scripts):
  (left)  3-D concave production surface y=f(Labour,Intermediate), ONE surface
          PER YEAR (later years sit above earlier = technical progress).
  (right, stacked) the three 2-D projections, drawn per year so annual variation
          shows:
    View B  Output vs Labour        (concave, shifts up = tech progress)
    View C  Output vs Intermediate  (concave, shifts up)
    View D  Labour-Intermediate isoquant at fixed output y* (convex, one per year)

Per-year Cobb-Douglas; the two off-axis inputs held at fixed overall medians so
year differences reflect the technology shift, not median drift.
Output: src/figures/fig_io_surface_combined.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import _common as C

YV, X1, X2 = "real_gvp", "Laborday_impute", "Inter_all_real"
iov = [YV, X1, "Land_serv_q", "capital_serv_q", X2]
YEARS = [1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2005, 2009, 2010, 2015]
cols = plt.cm.plasma(np.linspace(0.05, 0.85, len(YEARS)))
sL, sM, sY = 1e6, 1e6, 1e3
LBL1, LBL2, LBLY = "Labour (M man-days)", "Intermediate (M, real)", "Real GVP (k)"

d = pd.read_csv(C.CLEAN_PANEL).dropna(subset=iov)
d = d[(d[iov] > 0).all(axis=1)].copy()
mlL, mlM = np.log(d[X1]).median(), np.log(d[X2]).median()
mlLand, mlK = np.log(d.Land_serv_q).median(), np.log(d.capital_serv_q).median()
ystar = d[YV].median()
Lg = np.linspace(d[X1].quantile(.08), d[X1].quantile(.92), 120)
Mg = np.linspace(d[X2].quantile(.08), d[X2].quantile(.92), 120)

# per-year Cobb-Douglas (used for BOTH the 3-D surfaces and the projections)
fit = {}
print("per-year Cobb-Douglas elasticities:")
for yr in YEARS:
    s = d[d.year == yr]
    p = sm.OLS(np.log(s[YV]).values,
               sm.add_constant(np.column_stack([np.log(s[X1]), np.log(s.Land_serv_q),
                                                 np.log(s.capital_serv_q), np.log(s[X2])]))).fit().params
    fit[yr] = dict(c=p[0], bL=p[1], bLand=p[2], bK=p[3], bM=p[4])
    print(f"  {yr}: bL={p[1]:.2f} bM={p[4]:.2f} (bLand={p[2]:.2f} bK={p[3]:.2f}) const={p[0]:.2f}")

def surf(f, L, M):
    return np.exp(f["c"] + f["bL"]*np.log(L) + f["bM"]*np.log(M) + f["bLand"]*mlLand + f["bK"]*mlK)

def outL(f, L): return np.exp(f["c"]+f["bL"]*np.log(L)+f["bM"]*mlM+f["bLand"]*mlLand+f["bK"]*mlK)
def outM(f, M): return np.exp(f["c"]+f["bL"]*mlL+f["bM"]*np.log(M)+f["bLand"]*mlLand+f["bK"]*mlK)
def isoq(f, L):
    rhs = np.log(ystar)-f["c"]-f["bLand"]*mlLand-f["bK"]*mlK
    return np.exp((rhs-f["bL"]*np.log(L))/f["bM"])

fig = plt.figure(figsize=(15.5, 8.8))
gs = GridSpec(3, 2, width_ratios=[1.35, 1], hspace=0.6, wspace=0.22)

# ---------------- LEFT: 3-D surfaces by year -------------------------------
ax = fig.add_subplot(gs[:, 0], projection="3d")
Lgr = np.linspace(d[X1].quantile(.10), d[X1].quantile(.90), 45)
Mgr = np.linspace(d[X2].quantile(.10), d[X2].quantile(.90), 45)
LL, MM = np.meshgrid(Lgr, Mgr)
zmax = 0
for yr, col in zip(YEARS, cols):
    Z = surf(fit[yr], LL, MM)
    ax.plot_surface(LL/sL, MM/sM, Z/sY, color=col, alpha=.45, linewidth=0,
                    antialiased=True, rstride=2, cstride=2, shade=False)
    zmax = max(zmax, (Z/sY).max())
# isoquant contours (2015) dropped on the floor
ax.contour(LL/sL, MM/sM, surf(fit[YEARS[-1]], LL, MM)/sY, levels=7, zdir="z", offset=0,
           colors="0.6", linewidths=.6)
ax.set_zlim(0, zmax*1.05)
ax.set_xlabel(LBL1, fontsize=8); ax.set_ylabel(LBL2, fontsize=8); ax.set_zlabel(LBLY, fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title("(A) 3-D production surface by year\nsurfaces rise over time = technical progress", fontsize=9.5)
ax.legend(handles=[Patch(color=c, alpha=.6, label=str(y)) for y, c in zip(YEARS, cols)],
          title="year", fontsize=7, loc="upper left")
ax.view_init(elev=22, azim=-56)

# ---------------- RIGHT: three per-year projections ------------------------
a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1]); a3 = fig.add_subplot(gs[2, 1])
for yr, col in zip(YEARS, cols):
    f = fit[yr]
    a1.plot(Lg/sL, outL(f, Lg)/sY, color=col, lw=2, label=yr)
    a2.plot(Mg/sM, outM(f, Mg)/sY, color=col, lw=2, label=yr)
    a3.plot(Lg/sL, isoq(f, Lg)/sM, color=col, lw=2, label=yr)
a1.set_title("(B) Output vs Labour — by year (concave, shifts up)", fontsize=9)
a1.set_xlabel(LBL1, fontsize=8); a1.set_ylabel(LBLY, fontsize=8)
a2.set_title("(C) Output vs Intermediate — by year (concave, shifts up)", fontsize=9)
a2.set_xlabel(LBL2, fontsize=8); a2.set_ylabel(LBLY, fontsize=8)
a3.set_title(f"(D) Labour–Intermediate isoquant, y*≈{ystar/sY:.0f}k — each curve a year", fontsize=9)
a3.set_xlabel(LBL1, fontsize=8); a3.set_ylabel(LBL2, fontsize=8); a3.set_ylim(0, d[X2].quantile(.92)/sM)
for a in (a1, a2, a3):
    a.tick_params(labelsize=7); a.legend(title="year", fontsize=6.5, ncol=2); a.grid(alpha=.3)

fig.suptitle("China county agriculture: per-year 3-D production surface + its three projections (Cobb-Douglas, cleaned panel)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(C.FIG_DIR, "fig_io_surface_combined.png")
fig.savefig(out, dpi=160); plt.close(fig)
print("saved:", out)
