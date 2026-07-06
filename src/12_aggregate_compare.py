# -*- coding: utf-8 -*-
"""
Aggregate county-level TFP growth to a national annual series and compare
SFA (BC92, land-normalized, from 11_sfa_bc92_tfp.do) vs DEA (sequential NIRS
Malmquist, from 09_dea_vs_fe_tfp.py), under TWO aggregation weightings:

  simple   : equal weight over counties (mean of county dln)
  weighted : output-share Tornqvist  w_it = 0.5*(s_{i,t-1}+s_{i,t}),
             s = county share of national real GVP; the standard growth-
             accounting / sector-TFP aggregation (big farm counties count more).

Inputs  (src/clean/):
  sfa_bc92_county_year.csv   countyid year real_gvp ln_tfp_chen ...
  dea_malmquist_county.csv   year countyid lnM        (per-county Malmquist)
  county_panel_clean.csv     real_gvp for the GVP weights
Outputs (src/clean/ , src/figures/):
  tfp_sfa_annual.csv, tfp_dea_annual.csv, tfp_compare_annual.csv
  fig_tfp_sfa_vs_dea.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

CL = C.CLEAN_DIR

# ---- GVP weights from the cleaned panel -----------------------------------
panel = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "year", "real_gvp"])
panel = panel.dropna(subset=["real_gvp"])
panel = panel[panel.real_gvp > 0]
GVP = {(int(r.countyid), int(r.year)): float(r.real_gvp) for r in panel.itertuples()}

def aggregate(dln_df):
    """dln_df: columns countyid, year(=t, growth t-1->t), dln.
    returns per-year: n, simple mean, output-share Tornqvist weighted mean."""
    rows = []
    for t, g in dln_df.groupby("year"):
        c = g.countyid.to_numpy(); d = g.dln.to_numpy(float)
        gt  = np.array([GVP.get((int(x), int(t)),   np.nan) for x in c])
        gtm = np.array([GVP.get((int(x), int(t)-1), np.nan) for x in c])
        ok = np.isfinite(gt) & np.isfinite(gtm) & np.isfinite(d)
        c, d, gt, gtm = c[ok], d[ok], gt[ok], gtm[ok]
        if len(d) == 0:
            continue
        st = gt / gt.sum(); stm = gtm / gtm.sum()
        w = 0.5 * (st + stm); w = w / w.sum()
        rows.append((int(t), len(d), float(d.mean()), float((w * d).sum()),
                     float(d.std(ddof=1)) if len(d) > 1 else np.nan))
    out = pd.DataFrame(rows, columns=["year", "n", "dln_simple", "dln_weighted", "dln_sd"])
    out = out.sort_values("year").reset_index(drop=True)
    for k in ("simple", "weighted"):
        out[f"cum_{k}"]    = out[f"dln_{k}"].cumsum()
        out[f"growth_{k}"] = 100 * (np.exp(out[f"dln_{k}"]) - 1)   # % per year
    return out

# ---- SFA: per-county log growth of ln_tfp_chen ----------------------------
sfa = pd.read_csv(os.path.join(CL, "sfa_bc92_county_year.csv"))
sfa = sfa.sort_values(["countyid", "year"])
sfa["dln"] = sfa.groupby("countyid")["ln_tfp_chen"].diff()
sfa["gap"] = sfa.groupby("countyid")["year"].diff()
sfa_dln = sfa[(sfa.gap == 1) & sfa.dln.notna()][["countyid", "year", "dln"]]
sfa_ann = aggregate(sfa_dln)

# ---- DEA: per-county lnM already IS the t-1->t growth ---------------------
dea = pd.read_csv(os.path.join(CL, "dea_malmquist_county.csv")).rename(columns={"lnM": "dln"})
dea_ann = aggregate(dea[["countyid", "year", "dln"]])

sfa_ann.to_csv(os.path.join(CL, "tfp_sfa_annual.csv"), index=False)
dea_ann.to_csv(os.path.join(CL, "tfp_dea_annual.csv"), index=False)

# ---- merged comparison table ----------------------------------------------
comp = (sfa_ann[["year", "n", "growth_simple", "growth_weighted", "cum_simple", "cum_weighted"]]
        .rename(columns=lambda x: "sfa_" + x if x not in ("year",) else x)
        .merge(dea_ann[["year", "n", "growth_simple", "growth_weighted", "cum_simple", "cum_weighted"]]
               .rename(columns=lambda x: "dea_" + x if x not in ("year",) else x), on="year", how="outer")
        .sort_values("year"))
comp.to_csv(os.path.join(CL, "tfp_compare_annual.csv"), index=False)

def summ(a, tag):
    yr = a.year.iloc[-1] - a.year.iloc[0]
    for k in ("simple", "weighted"):
        cagr = 100 * a[f"cum_{k}"].iloc[-1] / yr
        gm, gs = a[f"growth_{k}"].mean(), a[f"growth_{k}"].std(ddof=1)
        print(f"  {tag:>4} {k:>8}: cum {a.year.iloc[0]}-{a.year.iloc[-1]} "
              f"{100*a[f'cum_{k}'].iloc[-1]:+6.1f}% log  |  {cagr:+.2f}%/yr  |  "
              f"annual mean {gm:+.2f}%  sd {gs:.2f}")
print("National agricultural TFP growth (cleaned panel):")
summ(sfa_ann, "SFA"); summ(dea_ann, "DEA")

# ---- figure: cumulative index (A) + annual growth (B) ---------------------
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
S = {"SFA weighted": (sfa_ann, "cum_weighted", "growth_weighted", "#1f77b4", "-"),
     "SFA simple":   (sfa_ann, "cum_simple",   "growth_simple",   "#1f77b4", "--"),
     "DEA weighted": (dea_ann, "cum_weighted", "growth_weighted", "#d62728", "-"),
     "DEA simple":   (dea_ann, "cum_simple",   "growth_simple",   "#d62728", "--")}
for lab, (a, ck, gk, col, ls) in S.items():
    yr = a.year.iloc[-1] - a.year.iloc[0]
    cagr = 100 * a[ck].iloc[-1] / yr
    ax1.plot(a.year, 100 * a[ck], ls, color=col, lw=1.8 if ls == "-" else 1.2,
             label=f"{lab} ({cagr:+.2f}%/yr)")
ax1.axhline(0, color="k", lw=.6, ls=":")
ax1.set_ylabel("Cumulative ln TFP x100 (~% vs base)")
ax1.set_title("China agricultural TFP: SFA (land-normalized CRS frontier) vs DEA (NIRS sequential)\n"
              "cleaned county panel — simple mean vs output-share (Tornqvist) weighting\n"
              "[SFA inefficiency ~0 (symmetric residuals) => SFA weighted=simple]")
ax1.legend(fontsize=8, ncol=2); ax1.grid(alpha=.3)
# Panel B: headline weighted annual growth
for lab, a, col in [("SFA weighted", sfa_ann, "#1f77b4"), ("DEA weighted", dea_ann, "#d62728")]:
    ax2.plot(a.year, a.growth_weighted, "-o", ms=3.5, color=col, label=lab)
ax2.axhline(0, color="k", lw=.6, ls=":")
ax2.set_ylabel("Annual TFP growth (%)"); ax2.set_xlabel("year")
ax2.set_title("Panel B: year-on-year growth (output-share weighted)")
ax2.legend(fontsize=8); ax2.grid(alpha=.3)
fig.tight_layout()
p = os.path.join(C.FIG_DIR, "fig_tfp_sfa_vs_dea.png")
fig.savefig(p, dpi=200); plt.close(fig)
print("saved:", p)
print("tables:", "tfp_sfa_annual.csv, tfp_dea_annual.csv, tfp_compare_annual.csv")
