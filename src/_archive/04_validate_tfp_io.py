# -*- coding: utf-8 -*-
"""
STEP 4 — Validate the cleaning via production-economics checks.

Does the cleaned county panel behave like a sensible production technology, and
did cleaning remove spurious volatility?  Four diagnostics, RAW vs CLEANED:

  A. National aggregate index (output + 4 inputs, balanced counties, base year=100).
  B. Input-output relationships: ln Y vs ln X_k (year-demeaned), binned means +
     OLS slope = elasticity.  Production theory expects monotone-increasing,
     weakly concave responses and 0<elasticity<1 for each input.
  C. TFP by year: two-way (county+year) FE Cobb-Douglas elasticities, then the
     Solow residual ln TFP = ln Y - Σ β_k ln X_k; national mean index over time.
  D. TFP volatility: distribution of year-on-year Δln TFP.  Good cleaning should
     shrink the spurious tails without flattening the real trend.

Output -> src/figures/:  fig_io_aggregates.png, fig_io_elasticities.png,
          fig_tfp_byyear.png, fig_tfp_volatility.png   (+ console summary)
Run:  python src/04_validate_tfp_io.py
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
ALL = [Y] + INPUTS
m = pd.read_csv(C.CLEAN_PANEL)


def positive_logs(df, cols):
    """Return df with ln<col> columns, keeping only rows where ALL cols are > 0."""
    d = df.copy()
    ok = np.ones(len(d), bool)
    for c in cols:
        ok &= pd.to_numeric(d[c], errors="coerce").fillna(-1).values > 0
    d = d[ok].copy()
    for c in cols:
        d["l_" + c] = np.log(d[c].astype(float))
    return d


def twoway_within(d, ycol, xcols, iters=20):
    """Iterative county+year demeaning, then OLS (no intercept). Returns betas dict."""
    cols = [ycol] + xcols
    w = d[["countyid", "year"] + cols].copy()
    for _ in range(iters):
        w[cols] = w[cols] - w.groupby("countyid")[cols].transform("mean")
        w[cols] = w[cols] - w.groupby("year")[cols].transform("mean")
    A = w[xcols].values
    b = w[ycol].values
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return dict(zip(xcols, beta))


# ============================================================================
# A. National aggregate index (raw vs cleaned)
# ============================================================================
def balanced(d):
    ny = d["year"].nunique()
    g = d.groupby("countyid")["year"].nunique()
    return d[d["countyid"].isin(g[g == ny].index)]

labels = {Y: "Real GVP (output)", "Laborday_impute": "Labour",
          "Land_serv_q": "Land", "capital_serv_q": "Capital", "Inter_all_real": "Intermediate"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
base = int(m.year.min())
for ax, suffix, title in [(axes[0], "_raw", "RAW (pre-cleaning)"),
                          (axes[1], "", "CLEANED")]:
    cols = [v + suffix for v in ALL]
    d = m.dropna(subset=cols).copy()
    for v in ALL:
        d = d[d[v + suffix] > 0]
    d = balanced(d)
    tot = d.groupby("year")[cols].sum()
    if base not in tot.index:
        base = int(tot.index.min())
    idx = tot / tot.loc[base] * 100
    for v in ALL:
        ax.plot(idx.index, idx[v + suffix], lw=1.8, label=labels[v])
    ax.axhline(100, color="k", lw=.6, ls=":")
    ax.set_title(f"{title}\n({d.countyid.nunique()} balanced counties)")
    ax.set_ylabel(f"Index, {base} = 100")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("A. National agricultural aggregates — raw vs cleaned", fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "fig_io_aggregates.png")); plt.close(fig)

# ============================================================================
# B. Input-output relationships (cleaned, year-demeaned logs)
# ============================================================================
d = positive_logs(m, ALL)
for v in ALL:
    d["d_" + v] = d["l_" + v] - d.groupby("year")["l_" + v].transform("mean")
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
elas_binned = {}
for ax, v in zip(axes.ravel(), INPUTS):
    x = d["d_" + v].values; y = d["d_" + Y].values
    mstat, edges, _ = binned_statistic(x, y, statistic="mean", bins=25)
    cen = (edges[:-1] + edges[1:]) / 2
    ax.scatter(x, y, s=2, alpha=.03, color="gray")
    ax.plot(cen, mstat, "o-", color="C3", ms=4, label="binned mean")
    slope = np.polyfit(x, y, 1)[0]
    elas_binned[v] = slope
    ax.set_title(f"ln Y vs ln {labels[v]}\nslope (elasticity) ≈ {slope:.2f}")
    ax.set_xlabel(f"ln {labels[v]} (year-demeaned)"); ax.set_ylabel("ln Y (year-demeaned)")
    ax.legend(fontsize=8)
fig.suptitle("B. Input-output relationships — monotone & concave? (cleaned)", fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "fig_io_elasticities.png")); plt.close(fig)

# ============================================================================
# C. Production-function elasticities + TFP by year
# ============================================================================
beta = twoway_within(d, "l_" + Y, ["l_" + v for v in INPUTS])
beta = {v: beta["l_" + v] for v in INPUTS}
rts = sum(beta.values())

# Solow residual on cleaned and raw logs (same betas for comparability)
draw = positive_logs(m.rename(columns={v + "_raw": "RAW_" + v for v in ALL}),
                      ["RAW_" + v for v in ALL])
draw.columns = [c.replace("l_RAW_", "lraw_").replace("RAW_", "") for c in draw.columns]

d["lnTFP"] = d["l_" + Y] - sum(beta[v] * d["l_" + v] for v in INPUTS)
draw["lnTFP"] = draw["lraw_" + Y] - sum(beta[v] * draw["lraw_" + v] for v in INPUTS)

tfp_clean = d.groupby("year")["lnTFP"].mean()
tfp_raw = draw.groupby("year")["lnTFP"].mean()
b0 = int(min(tfp_clean.index.min(), tfp_raw.index.min()))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tfp_raw.index, (tfp_raw - tfp_raw.loc[b0]) * 100, color="0.6", lw=1.6, marker="o", ms=3, label="raw")
ax.plot(tfp_clean.index, (tfp_clean - tfp_clean.loc[b0]) * 100, color="C2", lw=2, marker="o", ms=3, label="cleaned")
ax.axhline(0, color="k", lw=.6, ls=":")
ax.set_title(f"C. Agricultural TFP index (Solow residual, {b0}=0)\n"
             f"two-way FE elasticities: " +
             ", ".join(f"{labels[v]} {beta[v]:.2f}" for v in INPUTS) +
             f"  |  RTS={rts:.2f}", fontsize=10)
ax.set_ylabel("ln TFP × 100 (≈ % vs base)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "fig_tfp_byyear.png")); plt.close(fig)

# ============================================================================
# D. TFP volatility (year-on-year Δln TFP), raw vs cleaned
# ============================================================================
def dln_tfp(frame):
    f = frame.sort_values(["countyid", "year"]).copy()
    f["d"] = f.groupby("countyid")["lnTFP"].diff()
    f["dyr"] = f.groupby("countyid")["year"].diff()
    return f.loc[f["dyr"] == 1, "d"].dropna().values

vr = dln_tfp(draw); vc = dln_tfp(d)
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-2, 2, 81)
ax.hist(vr, bins=bins, density=True, alpha=.5, color="0.6", label=f"raw (sd={vr.std():.2f})")
ax.hist(vc, bins=bins, density=True, alpha=.5, color="C2", label=f"cleaned (sd={vc.std():.2f})")
ax.set_xlim(-2, 2); ax.set_xlabel("Δln TFP (year-on-year)"); ax.set_ylabel("density")
ax.set_title("D. TFP year-on-year change distribution — cleaning shrinks spurious tails")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "fig_tfp_volatility.png")); plt.close(fig)

# ============================================================================
# Console summary
# ============================================================================
print("Production-function elasticities (two-way county+year FE, cleaned):")
for v in INPUTS:
    print(f"  {labels[v]:14s} {beta[v]:6.3f}")
print(f"  {'Returns to scale':14s} {rts:6.3f}")
print("\nBinned (year-demeaned) slopes:")
for v in INPUTS:
    print(f"  {labels[v]:14s} {elas_binned[v]:6.3f}")
def tail(x, t=1.0): return float((np.abs(x) > t).mean())
print(f"\nTFP year-on-year Δln volatility:  raw sd={vr.std():.3f}  cleaned sd={vc.std():.3f}  "
      f"({100*(1-vc.std()/vr.std()):.0f}% lower)")
print(f"Share of |Δln TFP|>1:             raw {tail(vr):.3f}      cleaned {tail(vc):.3f}")
print("\nFigures -> src/figures/: fig_io_aggregates.png, fig_io_elasticities.png, "
      "fig_tfp_byyear.png, fig_tfp_volatility.png")
