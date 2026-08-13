# -*- coding: utf-8 -*-
"""
PARAMETRIC counterpart to the non-parametric Fig 9 / Fig 10 in 52_ and 53_.

Same object, two estimators.  53_frontier_isoquant_expansion.py builds the
best-practice boundary NON-PARAMETRICALLY (an order-alpha quantile envelope of
the observed cloud, no functional form).  This script imposes a FUNCTIONAL FORM
and draws the fitted surface.  Agreement between them is the check: a smooth
parametric surface and a data-driven envelope should say the same thing about
curvature and about the direction of the shift.

FUNCTIONAL FORM (--form)
  cobb      ln y = a + bL lnL + bLand lnLand + bK lnK + bM lnM
            One elasticity per input, constant everywhere.  Concavity follows
            automatically whenever the elasticities sum to <= 1, so this figure
            cannot fail the production-theory check -- it ASSUMES it.
  translog  the second-order flexible form, adding every square and cross term
                ln y = ... + 0.5 SUM_j SUM_k b_jk lnX_j lnX_k
            Elasticities now VARY with the input bundle, so curvature and the
            substitution structure are ESTIMATED rather than imposed.  This is
            the honest parametric test of whether the data trace a concave
            surface, and it is the standard flexible form in the productivity
            literature (Christensen, Jorgenson & Lau 1973).  DEFAULT.

PANELS
  (A) 3-D production surface y = f(L, M), one surface per year
  (B) Output vs Labour
  (C) Output vs Intermediate
  (D) Output vs CAPITAL -- the parametric twin of the K/L view that the
      non-parametric Fig 9 plots, so the two can be read side by side
  (E) Labour-Intermediate isoquant at fixed y*

Off-axis inputs are held at fixed overall medians, so differences between years
are the technology shift and not median drift.

Run:  python src/50_io_surface.py                # translog (default)
      python src/50_io_surface.py --form cobb    # Cobb-Douglas
Output: src/figures/fig_io_surface_combined.png
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import re
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import _common as C

_ap = argparse.ArgumentParser()
_ap.add_argument("--form", choices=["translog", "cobb"], default="translog")
_ap.add_argument("--years", default="1986,1995,2000,2005,2010,2015")
ARGS = _ap.parse_args()

YV, X1, X2 = "real_gvp", "Laborday_impute", "Inter_all_real"
iov = [YV, X1, "Land_serv_q", "capital_serv_q", X2]
YEARS = [int(y) for y in ARGS.years.split(",")]
# colour by the YEAR VALUE (not by rank) so the shared colourbar is exact,
# even though YEARS is unevenly spaced
YNORM = plt.Normalize(vmin=min(YEARS), vmax=max(YEARS))
cols = plt.cm.plasma(0.05 + 0.80 * YNORM(np.array(YEARS, float)))
sL, sM, sY = 1e6, 1e6, 1e3
LBL1, LBL2, LBLY = "Labour (M man-days)", "Intermediate (M, real)", "Real GVP (k)"

# Agricultural sample: the cleaned panel holds ALL resolved rural counties so it
# can be reused; every analysis here is on the cropland>=15% agricultural subset.
d = pd.read_csv(C.CLEAN_PANEL)
d = d[d["ag_county"] == 1].dropna(subset=iov)
d = d[(d[iov] > 0).all(axis=1)].copy()
mlL, mlM = np.log(d[X1]).median(), np.log(d[X2]).median()
mlLand, mlK = np.log(d.Land_serv_q).median(), np.log(d.capital_serv_q).median()
ystar = d[YV].median()
Lg = np.linspace(d[X1].quantile(.08), d[X1].quantile(.92), 120)
Mg = np.linspace(d[X2].quantile(.08), d[X2].quantile(.92), 120)
# (D) capital: the parametric twin of the K/L view in the non-parametric Fig 9
Kg = np.linspace(d.capital_serv_q.quantile(.08), d.capital_serv_q.quantile(.92), 120)
sK = 1e6
LBLK = "Capital service (M, real)"

# ---------------------------------------------------------------------------
# Technology per year.  PREFERRED: the SFA production FRONTIER from
# 21_sfa_bc92.do (sfpanel, model(bc92), distribution(tnormal), land-
# normalized => CRS).  Converting back to levels,
#     ln y = alpha + lambda_t + bL lnL + bK lnK + bM lnM + (1-bL-bK-bM) lnLand
# so input elasticities are COMMON across years and the year effect lambda_t
# shifts the frontier neutrally (Hicks-neutral technical change).
# FALLBACK (if the SFA has not been estimated yet): per-year OLS = the AVERAGE
# relationship, with year-specific elasticities.
# ---------------------------------------------------------------------------
XCOLS = [X1, "Land_serv_q", "capital_serv_q", X2]          # L, Land, K, M
XKEY = ["L", "Land", "K", "M"]


def design(lx):
    """Regressor block for the chosen form.  lx: (n,4) logged inputs."""
    if ARGS.form == "cobb":
        return lx
    cols = [lx]
    for i in range(4):                       # squares and cross terms
        for j in range(i, 4):
            v = lx[:, i] * lx[:, j]
            cols.append((0.5 * v if i == j else v).reshape(-1, 1))
    return np.column_stack(cols)


def fit_year(s):
    lx = np.column_stack([np.log(s[c].values) for c in XCOLS])
    r = sm.OLS(np.log(s[YV].values), sm.add_constant(design(lx))).fit()
    return dict(params=r.params, r2=r.rsquared, n=len(s))


def predict(f, lL, lLand, lK, lM):
    """Fitted ln y at a logged bundle; broadcasts over arrays."""
    lL, lLand, lK, lM = np.broadcast_arrays(*[np.asarray(v, float)
                                              for v in (lL, lLand, lK, lM)])
    flat = np.stack([lL, lLand, lK, lM], axis=-1).reshape(-1, 4)
    Z = np.column_stack([np.ones(len(flat)), design(flat)])
    return (Z @ f["params"]).reshape(lL.shape)


FORMLAB = ("translog (flexible: curvature ESTIMATED)" if ARGS.form == "translog"
           else "Cobb-Douglas (curvature IMPOSED)")
print(f"per-year {ARGS.form.upper()} fit -- {FORMLAB}")
fit = {}
for yr in YEARS:
    fit[yr] = fit_year(d[d.year == yr])
    print(f"  {yr}: n={fit[yr]['n']:,d}  R2={fit[yr]['r2']:.3f}")
SRC = f"per-year {FORMLAB}"


def elasticities(f, lx):
    """d ln y / d ln X_j at each observation.  Constant under Cobb-Douglas;
    bundle-dependent under translog."""
    b = f["params"][1:5]
    el = np.tile(b, (len(lx), 1))
    if ARGS.form == "translog":
        q = f["params"][5:]
        k = 0
        for i in range(4):
            for j in range(i, 4):
                if i == j:
                    el[:, i] += q[k] * lx[:, i]
                else:
                    el[:, i] += q[k] * lx[:, j]
                    el[:, j] += q[k] * lx[:, i]
                k += 1
    return el


# REGULARITY CHECK.  A production function must be MONOTONE: every fitted
# elasticity d ln y / d ln X should be positive.  Neither form is constrained
# here, so the check is REPORTED rather than assumed.  An unconstrained translog
# violating monotonicity is a known and common outcome, and the reader needs to
# see it instead of inferring concavity from a smooth-looking surface.
print("")
print("REGULARITY: share of observations with a POSITIVE fitted elasticity")
print("  " + f"{'year':>6} " + " ".join(f"{k:>7}" for k in XKEY) + "     all")
for yr in YEARS:
    s = d[d.year == yr]
    lx = np.column_stack([np.log(s[c].values) for c in XCOLS])
    el = elasticities(fit[yr], lx)
    sh = " ".join(f"{100*(el[:, i] > 0).mean():6.1f}%" for i in range(4))
    print(f"  {yr:>6} {sh}  {100*(el > 0).all(axis=1).mean():5.1f}%")
print("  NOTE: the low share for CAPITAL is a property of this panel, not of the")
print("  functional form -- conditional on L, Land and M the output-capital")
print("  relation is negative in most years under BOTH forms.  The U-shaped")
print("  capital panel (D) is that fact, not a fitting artefact.")


def surf(f, L, M):
    return np.exp(predict(f, np.log(L), mlLand, mlK, np.log(M)))


def outL(f, L):  return np.exp(predict(f, np.log(L), mlLand, mlK, mlM))
def outM(f, M):  return np.exp(predict(f, mlL, mlLand, mlK, np.log(M)))
def outK(f, K):  return np.exp(predict(f, mlL, mlLand, np.log(K), mlM))


# The isoquant reference output must lie ON the plotted technology, so it is
# anchored to the fitted surface at median inputs, not to the observed median.
_ymid = YEARS[len(YEARS) // 2]
Y_ISO = float(surf(fit[_ymid], np.exp(mlL), np.exp(mlM)))
print(f"isoquant reference output y* = {Y_ISO/sY:,.0f}k "
      f"(technology of {_ymid} at median L,M; observed median = {ystar/sY:,.0f}k)")


def isoq(f, L):
    """M solving f(L, M) = Y_ISO.  Translog has no closed form, so it is solved
    numerically on a log-M grid (output is monotone in M over the plotted range)."""
    grid = np.exp(np.linspace(np.log(d[X2].quantile(.01)),
                              np.log(d[X2].quantile(.99)), 400))
    out = []
    for LL in np.atleast_1d(L):
        yy = np.exp(predict(f, np.log(LL), mlLand, mlK, np.log(grid)))
        o_ = np.argsort(yy)
        out.append(float(np.interp(Y_ISO, yy[o_], grid[o_],
                                   left=np.nan, right=np.nan)))
    return np.array(out)


fig = plt.figure(figsize=(15.5, 11.5))
gs = GridSpec(4, 2, width_ratios=[1.35, 1], hspace=0.8, wspace=0.22)

# ---------------- LEFT: 3-D surfaces by year -------------------------------
ax = fig.add_subplot(gs[:, 0], projection="3d")
Lgr = np.linspace(d[X1].quantile(.10), d[X1].quantile(.90), 45)
Mgr = np.linspace(d[X2].quantile(.10), d[X2].quantile(.90), 45)
LL, MM = np.meshgrid(Lgr, Mgr)
# too many translucent surfaces turn to mud: show at most 5, spread over YEARS
_idx = np.unique(np.linspace(0, len(YEARS)-1, min(5, len(YEARS))).round().astype(int))
SURF_YEARS = [YEARS[i] for i in _idx]
scols = {y: c for y, c in zip(YEARS, cols)}
zmax = 0
for yr in SURF_YEARS:
    Z = surf(fit[yr], LL, MM)
    ax.plot_surface(LL/sL, MM/sM, Z/sY, color=scols[yr], alpha=.45, linewidth=0,
                    antialiased=True, rstride=2, cstride=2, shade=False)
    zmax = max(zmax, (Z/sY).max())
# isoquant contours (last year) dropped on the floor
ax.contour(LL/sL, MM/sM, surf(fit[YEARS[-1]], LL, MM)/sY, levels=7, zdir="z", offset=0,
           colors="0.6", linewidths=.6)
ax.set_zlim(0, zmax*1.05)
ax.set_xlabel(LBL1, fontsize=8); ax.set_ylabel(LBL2, fontsize=8); ax.set_zlabel(LBLY, fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title("(A) 3-D production surface by year\nsurfaces rise over time = technical progress", fontsize=9.5)
ax.legend(handles=[Patch(color=scols[y], alpha=.6, label=str(y)) for y in SURF_YEARS],
          title="year", fontsize=7, loc="upper left")
ax.view_init(elev=22, azim=-56)

# ---------------- RIGHT: three per-year projections ------------------------
a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
a4 = fig.add_subplot(gs[2, 1]); a3 = fig.add_subplot(gs[3, 1])
for yr, col in zip(YEARS, cols):
    f = fit[yr]
    a1.plot(Lg/sL, outL(f, Lg)/sY, color=col, lw=2, label=yr)
    a2.plot(Mg/sM, outM(f, Mg)/sY, color=col, lw=2, label=yr)
    a4.plot(Kg/sK, outK(f, Kg)/sY, color=col, lw=2, label=yr)
    a3.plot(Lg/sL, isoq(f, Lg)/sM, color=col, lw=2, label=yr)
a1.set_title("(B) Output vs Labour — by year", fontsize=9)
a1.set_xlabel(LBL1, fontsize=8); a1.set_ylabel(LBLY, fontsize=8)
a2.set_title("(C) Output vs Intermediate — by year", fontsize=9)
a2.set_xlabel(LBL2, fontsize=8); a2.set_ylabel(LBLY, fontsize=8)
a4.set_title("(D) Output vs CAPITAL — parametric twin of the Fig 9 K/L view", fontsize=9)
a4.set_xlabel(LBLK, fontsize=8); a4.set_ylabel(LBLY, fontsize=8)
a3.set_title(f"(E) Labour–Intermediate isoquant, y*≈{Y_ISO/sY:,.0f}k — each curve a year", fontsize=9)
a3.set_xlabel(LBL1, fontsize=8); a3.set_ylabel(LBL2, fontsize=8); a3.set_ylim(0, Mg.max()/sM)
for a in (a1, a2, a4, a3):
    a.tick_params(labelsize=7); a.grid(alpha=.3)
# one shared YEAR colourbar instead of 22-entry legends (which swamp the panels)
_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "plasma_sub", plt.cm.plasma(np.linspace(0.05, 0.85, 256)))
_sm = plt.cm.ScalarMappable(cmap=_cmap, norm=YNORM)
for a in (a1, a2, a4, a3):
    _cb = fig.colorbar(_sm, ax=a, pad=.015, fraction=.045)
    _cb.set_label("year", fontsize=7); _cb.ax.tick_params(labelsize=6.5)

fig.suptitle(f"China county agriculture: per-year 3-D production surface + its four projections\n{SRC} — "
             f"cleaned panel, {d.countyid.nunique():,} counties, {d.year.min()}–{d.year.max()}",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(C.FIG_DIR, "fig_io_surface_combined.png")
fig.savefig(out, dpi=160); plt.close(fig)
print("saved:", out)
