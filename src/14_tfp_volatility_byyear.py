# -*- coding: utf-8 -*-
"""
(1) Decompose aggregate national ag TFP growth into components to locate the
    source of the ~+-20% year-on-year volatility:
       dln TFP = dln(sum GVP_real) - sum_k beta_k dln(sum X_k)
       dln(sum GVP_real) = dln(sum GVP_nominal) - dln(PPI deflator)
    If real-output growth is the volatile piece and nominal is smooth -> deflator.
(2) TFP VOLATILITY BY YEAR:
       - national year-on-year TFP growth (the level of each year's swing)
       - cross-county dispersion (sd across counties) of county TFP growth per year
       - 5-yr rolling sd of the national series
Outputs a table src/clean/tfp_volatility_byyear.csv and a figure.
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
# keep nominal GVP + PPI too for the deflator decomposition
need = iov + ["GVP_allagr_impute", "PPI_CCD_2005"]
d = d.dropna(subset=iov)
d = d[(d[iov] > 0).all(axis=1)].copy()

# balanced panel (clean common set, removes composition)
nyr = d.year.nunique(); cnt = d.groupby("countyid").year.nunique()
bal = d[d.countyid.isin(cnt[cnt == nyr].index)].copy()
print(f"balanced panel: {bal.countyid.nunique()} counties x {nyr} yrs")

g = bal.groupby("year")
agg = pd.DataFrame({
    "GVP_real":  g["real_gvp"].sum(),
    "GVP_nom":   g["GVP_allagr_impute"].sum(),
    "L": g["Laborday_impute"].sum(), "Land": g["Land_serv_q"].sum(),
    "K": g["capital_serv_q"].sum(),  "M": g["Inter_all_real"].sum(),
})
# implied aggregate deflator = nominal/real (should track PPI)
agg["defl"] = agg["GVP_nom"] / agg["GVP_real"]
agg["ln_input"] = (B["Laborday_impute"]*np.log(agg.L) + B["Land_serv_q"]*np.log(agg.Land)
                   + B["capital_serv_q"]*np.log(agg.K) + B["Inter_all_real"]*np.log(agg.M))
agg["ln_tfp"] = np.log(agg.GVP_real) - agg.ln_input

dd = pd.DataFrame({
    "d_nomGVP": np.log(agg.GVP_nom).diff()*100,
    "d_defl":   np.log(agg.defl).diff()*100,
    "d_realGVP":np.log(agg.GVP_real).diff()*100,
    "d_input":  agg.ln_input.diff()*100,
    "d_tfp":    agg.ln_tfp.diff()*100,
}).dropna()
print("\nAggregate growth decomposition (%/yr) sd of each piece:")
for c in dd.columns:
    print(f"  {c:10s}: mean={dd[c].mean():+6.2f}  sd={dd[c].std():5.2f}  min={dd[c].min():+6.1f} max={dd[c].max():+6.1f}")
print("\n  corr(d_realGVP, d_tfp) =", round(dd.d_realGVP.corr(dd.d_tfp),3),
      "  corr(d_input,d_tfp) =", round(dd.d_input.corr(dd.d_tfp),3))
print("  => if sd(d_realGVP) ~ sd(d_tfp) and sd(d_input) small, output side drives volatility")

# ---- TFP volatility BY YEAR ------------------------------------------------
# county-level TFP growth (Solow, CRS) for cross-sectional dispersion
b2 = bal.sort_values(["countyid", "year"]).copy()
b2["ln_tfp_i"] = np.log(b2.real_gvp) - (B["Laborday_impute"]*np.log(b2.Laborday_impute)
                 + B["Land_serv_q"]*np.log(b2.Land_serv_q) + B["capital_serv_q"]*np.log(b2.capital_serv_q)
                 + B["Inter_all_real"]*np.log(b2.Inter_all_real))
b2["d_tfp_i"] = b2.groupby("countyid")["ln_tfp_i"].diff()*100
b2["gap"] = b2.groupby("countyid")["year"].diff()
b2 = b2[b2.gap == 1]
by = b2.groupby("year")["d_tfp_i"].agg(nat_mean="mean", nat_median="median",
                                       xsec_sd="std",
                                       p10=lambda s: s.quantile(.10),
                                       p90=lambda s: s.quantile(.90),
                                       n="count").reset_index()
# merge SFA & DEA national series
sfa = pd.read_csv(os.path.join(C.CLEAN_DIR, "tfp_sfa_annual.csv"))[["year","growth_weighted"]].rename(columns={"growth_weighted":"sfa_grw"})
dea = pd.read_csv(os.path.join(C.CLEAN_DIR, "tfp_dea_annual.csv"))[["year","growth_weighted"]].rename(columns={"growth_weighted":"dea_grw"})
by = by.merge(sfa,on="year",how="left").merge(dea,on="year",how="left")
by["roll5_sd_natmean"] = by["nat_mean"].rolling(5, center=True, min_periods=3).std()
by.to_csv(os.path.join(C.CLEAN_DIR, "tfp_volatility_byyear.csv"), index=False)
pd.set_option("display.width",200)
print("\nTFP volatility BY YEAR (balanced-panel Solow):")
print(by[["year","n","nat_mean","nat_median","xsec_sd","p10","p90","sfa_grw","dea_grw","roll5_sd_natmean"]]
      .round(2).to_string(index=False))
print("\nbiggest swing years (|nat_mean|):")
print(by.reindex(by.nat_mean.abs().sort_values(ascending=False).index).head(6)[["year","nat_mean","xsec_sd"]].round(1).to_string(index=False))

# ---- figure ---------------------------------------------------------------
fig,(a1,a2)=plt.subplots(2,1,figsize=(10,8),sharex=True)
a1.fill_between(by.year, by.p10, by.p90, color="0.8", label="county p10-p90")
a1.plot(by.year, by.nat_mean,"-o",ms=3,color="k",label="national mean TFP growth")
a1.plot(by.year, by.nat_median,"-",color="#2ca02c",label="national median")
a1.axhline(0,color="gray",lw=.6); a1.set_ylabel("TFP growth %/yr"); a1.legend(fontsize=8); a1.grid(alpha=.3)
a1.set_title("National ag TFP growth by year + cross-county dispersion (balanced panel, CRS Solow)")
a2.plot(by.year, by.xsec_sd,"-o",ms=3,color="#d62728",label="cross-county sd (dispersion each year)")
a2.plot(by.year, by.roll5_sd_natmean,"-",color="navy",label="5-yr rolling sd of national growth")
a2.set_ylabel("volatility (sd, %)"); a2.set_xlabel("year"); a2.legend(fontsize=8); a2.grid(alpha=.3)
a2.set_title("TFP volatility by year")
fig.tight_layout()
p=os.path.join(C.FIG_DIR,"fig_tfp_volatility_byyear.png")
fig.savefig(p,dpi=200); plt.close(fig)
print("\nsaved:",p,"\ntable: src/clean/tfp_volatility_byyear.csv")
