# -*- coding: utf-8 -*-
"""
Diagnose whether the ~+-20% year-on-year national TFP swings are REAL national
volatility or an averaging artifact of mean-of-county growth.

Builds the AGGREGATE (national totals) Solow TFP index: sum output & each input
across counties by year, then lnTFP_t = lnY_t - sum(beta_k ln X_kt) with the SFA
CRS elasticities.  If the aggregate index is smooth (sd ~3-5%) while the
mean-of-county series is sd ~8-9%, the volatility is a heavy-tail/composition
artifact, not real national volatility (and not necessarily bad cleaning).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

iov = ["real_gvp", "Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
B   = {"Laborday_impute": 0.47, "Land_serv_q": 0.015, "capital_serv_q": 0.07, "Inter_all_real": 0.446}

d = pd.read_csv(C.CLEAN_PANEL)
d = d.dropna(subset=iov); d = d[(d[iov] > 0).all(axis=1)].copy()

def agg_solow(df):
    g = df.groupby("year")[iov].sum()
    lnTFP = np.log(g["real_gvp"]) - sum(B[k] * np.log(g[k]) for k in B)
    return lnTFP

def rep(dln, tag):
    x = dln.diff().dropna() * 100
    print(f"  {tag:24s}: mean={x.mean():+.2f}%/yr  sd={x.std():.2f}  min={x.min():+.1f} max={x.max():+.1f}")
    return x

print("AGGREGATE (national totals) Solow TFP, CRS elasticities:")
allln = agg_solow(d); rep(allln, "all valid obs")
nyr = d.year.nunique(); cnt = d.groupby("countyid").year.nunique()
bal = d[d.countyid.isin(cnt[cnt == nyr].index)]
print(f"  balanced counties: {bal.countyid.nunique()} of {d.countyid.nunique()}")
balln = agg_solow(bal); balx = rep(balln, "balanced panel")

print("\nMEAN-of-county growth (Panel B series):")
sfa = pd.read_csv(os.path.join(C.CLEAN_DIR, "tfp_sfa_annual.csv"))
dea = pd.read_csv(os.path.join(C.CLEAN_DIR, "tfp_dea_annual.csv"))
print(f"  SFA weighted: sd={sfa.growth_weighted.std():.2f}  min={sfa.growth_weighted.min():+.1f} max={sfa.growth_weighted.max():+.1f}")
print(f"  DEA weighted: sd={dea.growth_weighted.std():.2f}  min={dea.growth_weighted.min():+.1f} max={dea.growth_weighted.max():+.1f}")

dc = pd.read_csv(os.path.join(C.CLEAN_DIR, "dea_malmquist_county.csv"))
print("\nCounty-level DEA lnM tails (heavy tails inflate the mean):")
print(f"  |lnM|>0.5: {100*(dc.lnM.abs()>0.5).mean():.1f}%   |lnM|>1: {100*(dc.lnM.abs()>1).mean():.2f}%   county sd={dc.lnM.std():.3f}")
print(f"  trimmed(1-99pct) county lnM sd={dc.lnM.clip(dc.lnM.quantile(.01),dc.lnM.quantile(.99)).std():.3f}")

# figure: aggregate (smooth) vs mean-of-county (volatile)
aggidx = (balln - balln.iloc[0]) * 100
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
a1.plot(aggidx.index, aggidx.values, "-o", ms=3, color="k", label=f"AGGREGATE Solow (national totals), sd/yr={balx.std():.1f}")
a1.plot(sfa.year, 100*sfa.cum_weighted, "-", color="#1f77b4", label="SFA mean-of-county (weighted)")
a1.plot(dea.year, 100*dea.cum_weighted, "-", color="#d62728", label="DEA mean-of-county (weighted)")
a1.set_ylabel("cum ln TFP x100"); a1.legend(fontsize=8); a1.grid(alpha=.3)
a1.set_title("Aggregate (national-totals) TFP vs mean-of-county TFP")
a2.plot(balx.index, balx.values, "-o", ms=3, color="k", label=f"AGGREGATE Solow YoY (sd={balx.std():.1f})")
a2.plot(dea.year, dea.growth_weighted, "-", color="#d62728", alpha=.7, label=f"DEA mean-of-county YoY (sd={dea.growth_weighted.std():.1f})")
a2.axhline(0, color="gray", lw=.6); a2.set_ylabel("YoY growth %"); a2.set_xlabel("year")
a2.legend(fontsize=8); a2.grid(alpha=.3)
fig.tight_layout()
p = os.path.join(C.FIG_DIR, "fig_tfp_aggregate_vs_meancounty.png")
fig.savefig(p, dpi=200); plt.close(fig)
print("\nsaved:", p)
