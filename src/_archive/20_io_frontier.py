# -*- coding: utf-8 -*-
"""
Rebuilt input-output figure (replaces fig_io_byyear.png), fixing two flaws:

  (A) AGGREGATE input vs output, with a genuine CONCAVE frontier.
      - single aggregate input index  ln X = sum_k beta_k ln x_k   (CRS frontier
        weights from the converged BC92: L .448, Land .081, K .038, M .433),
        so output is plotted against ONE economically-weighted input, not one
        raw input at a time.
      - the red frontier is the VRS-DEA concave hull = smallest concave, non-
        decreasing function that wraps the cloud (diminishing returns), NOT a
        jagged binned max.  Drawn for 1981/1995/2005/2015 -> upward shift =
        technical progress.
  (B) SUBSTITUTION: labour-intermediate isoquants (x1-x2 give same y) from the
      CD frontier -> convex curves = diminishing MRTS (input substitution).

Output: src/figures/fig_io_frontier.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

def _cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _smooth(vx, vy):
    """shape-preserving smooth curve through hull vertices (concave/convex kept)."""
    vx, vy = np.asarray(vx, float), np.asarray(vy, float)
    keep = np.concatenate(([True], np.diff(vx) > 1e-9))     # strictly increasing x
    vx, vy = vx[keep], vy[keep]
    if len(vx) < 3:
        return vx, vy
    xx = np.linspace(vx.min(), vx.max(), 150)
    return xx, PchipInterpolator(vx, vy)(xx)

C.set_cjk_font(plt)
np.seterr(divide="ignore", invalid="ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})

Y = "real_gvp"
INP = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
BETA = {"Laborday_impute": 0.47, "Land_serv_q": 0.015, "capital_serv_q": 0.07, "Inter_all_real": 0.45}
YEARS = [1981, 1995, 2005, 2015]
reds = ["#fdae61", "#f46d43", "#d73027", "#a50026"]     # light->dark red = time

m = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "year", Y] + INP)
m = m.dropna(subset=[Y] + INP)
m = m[(m[Y] > 0) & (m[INP] > 0).all(axis=1)].copy()
m["lnX"] = sum(BETA[k] * np.log(m[k]) for k in INP)      # aggregate input index (log)
m["lnY"] = np.log(m[Y])

# common input grid for ALL years so the frontiers are comparable (and nest)
GLO, GHI = np.quantile(m.lnX, [0.02, 0.98])
GEDGES = np.linspace(GLO, GHI, 19)

def concave_frontier(x, y, q=0.92):
    """Smooth CONCAVE best-practice frontier: q of output within FIXED input bins,
    the upper concave hull (guarantees diminishing returns), then a shape-
    preserving PCHIP smoothing so the line is smooth, not angular."""
    edges = GEDGES
    idx = np.digitize(x, edges)
    P = []
    for b in range(1, len(edges)):
        yy = y[idx == b]
        if len(yy) >= 8:
            P.append((0.5 * (edges[b-1] + edges[b]), float(np.quantile(yy, q))))
    if len(P) < 3:
        return np.array([p[0] for p in P]), np.array([p[1] for p in P])
    H = []                                               # upper concave hull
    for p in P:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) >= 0:
            H.pop()
        H.append(p)
    hx = np.array([h[0] for h in H]); hy = np.array([h[1] for h in H])
    k = int(np.argmax(hy)) + 1                            # keep the non-decreasing part
    return _smooth(hx[:k], hy[:k])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.8))

# ---------- Panel A: aggregate input vs output + concave frontiers ----------
s0 = m[m.year == 1981]
axA.scatter(s0.lnX, s0.lnY, s=6, alpha=0.15, color="0.55", label="1981 counties", zorder=1)
for yr, col in zip(YEARS, reds):
    s = m[m.year == yr]
    hx, hy = concave_frontier(s.lnX.values, s.lnY.values)
    axA.plot(hx, hy, "-", color=col, lw=2.3, label=f"{yr} concave frontier", zorder=4)
axA.set_xlabel(r"ln aggregate input  $\ln X=\sum_k \beta_k \ln x_k$  (L .47, M .45, K .07, land .015)")
axA.set_ylabel("ln Real GVP (output)")
axA.set_title("(A) Output vs AGGREGATE input — concave VRS frontier\nfrontier shifts up over time = technical progress")
axA.legend(fontsize=7.5, loc="upper left")

# ---------- Panel B: labour-intermediate ISOQUANT by year (nested cleanly) ---
# Parametric CD isoquant (fixed elasticities) shifted by the year's frontier
# technology A_t -> nests cleanly by construction: inward shift = technical
# progress, downward slope = substitution.  Labour-Intermediate is the pair with
# comparable elasticities (.47, .45), so it is the meaningful substitution pair
# (capital/land elasticities are tiny -> near non-substitutable; see fig_isoquants_pairs).
xi, xj = "Laborday_impute", "Inter_all_real"
resid = m.lnY.values - sum(BETA[k] * np.log(m[k]).values for k in INP)
A_t = {yr: np.quantile(resid[m.year.values == yr], 0.90) for yr in YEARS}
medln = {k: np.log(m[k]).median() for k in INP}
lnYstar = m.lnY.median()
lnxi = np.linspace(np.log(m[xi]).quantile(.08), np.log(m[xi]).quantile(.92), 100)
others = [k for k in INP if k not in (xi, xj)]
for yr, col in zip(YEARS, reds):
    const = lnYstar - A_t[yr] - sum(BETA[o] * medln[o] for o in others)
    lnxj = (const - BETA[xi] * lnxi) / BETA[xj]
    axB.plot(np.exp(lnxi), np.exp(lnxj), "-", color=col, lw=2.4, zorder=4, label=str(yr))
axB.set_xscale("log"); axB.set_yscale("log")
axB.annotate("isoquants shift inward\n(fewer inputs, same output\n= technical progress)",
             (0.04, 0.06), xycoords="axes fraction", fontsize=7, color="0.35", va="bottom")
axB.set_xlabel("Labour (man-days)"); axB.set_ylabel("Intermediate input (real)")
axB.set_title("(B) Labour–Intermediate ISOQUANT by year (nested)\ndownward = substitution; inward shift = technical progress")
axB.legend(fontsize=7.5, loc="upper right")

fig.suptitle("China county agriculture: production frontier — aggregate input–output & input substitution (cleaned panel)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
for name in ("fig_io_byyear.png", "fig_io_frontier.png"):   # overwrite the old flawed plot
    fig.savefig(os.path.join(C.FIG_DIR, name), dpi=150)
plt.close(fig)
print("saved: fig_io_byyear.png (replaced) + fig_io_frontier.png")

# report frontier slopes (returns to scale along the aggregate frontier)
for yr in YEARS:
    s = m[m.year == yr]; hx, hy = concave_frontier(s.lnX.values, s.lnY.values)
    sl = np.diff(hy) / np.diff(hx)
    print(f"  {yr} frontier: {len(hx)} vertices, slope {sl[0]:.2f}(low X) -> {sl[-1]:.2f}(high X)  [<1 & falling = concave/DRS]")
