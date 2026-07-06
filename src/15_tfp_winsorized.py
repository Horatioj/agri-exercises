# -*- coding: utf-8 -*-
"""
Winsorized national agricultural TFP (per user's fix #2).

For each method's COUNTY-level year-on-year TFP growth:
  1. winsorize the growth cross-section WITHIN each year at 1/99 (same as SFA.do's
     `winsor2 tfp_growth_pct, cuts(1 99)`, but by year) -> trims heavy tails
  2. aggregate to national: simple mean AND output-share Tornqvist weight
     w_it = 0.5*(s_{i,t-1}+s_{i,t}), s = county GVP share
  3. cumulative index, annual growth %, and 5-STAGE averages.

Methods:
  DEA  = sequential-NIRS Malmquist county lnM      (efficiency-inclusive, frontier)
  SOL  = CRS Solow residual, county-level          (transparent cross-check)
Compares raw vs winsorized volatility; maps to the five reform stages.

Outputs: src/clean/tfp_winsorized_annual.csv, src/clean/tfp_stage_summary.csv,
         src/figures/fig_tfp_winsorized_stages.png
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

# five reform stages (start of data 1981)
STAGES = [("1 Reform take-off", 1981, 1984),
          ("2 Stagnation",      1985, 1988),
          ("3 Recovery",        1989, 1996),
          ("4 Adjustment",      1997, 2003),
          ("5 Subsidy era",     2004, 2016)]

panel = pd.read_csv(C.CLEAN_PANEL)
panel = panel.dropna(subset=iov); panel = panel[(panel[iov] > 0).all(axis=1)].copy()
GVP = {(int(r.countyid), int(r.year)): float(r.real_gvp) for r in panel.itertuples()}

# ---- county-level dln for each method -------------------------------------
# SOL: CRS Solow residual
p = panel.sort_values(["countyid", "year"]).copy()
p["ln_tfp"] = np.log(p.real_gvp) - sum(B[k]*np.log(p[k]) for k in B)
p["dln"] = p.groupby("countyid")["ln_tfp"].diff()
p["gap"] = p.groupby("countyid")["year"].diff()
sol = p[(p.gap == 1) & p.dln.notna()][["countyid", "year", "dln"]].copy()
# DEA: county Malmquist lnM already IS the growth
dea = pd.read_csv(os.path.join(C.CLEAN_DIR, "dea_malmquist_county.csv")).rename(columns={"lnM": "dln"})[["countyid","year","dln"]]

def winsor_by_year(df, lo=0.01, hi=0.99):
    out = df.copy()
    out["dln_w"] = out.groupby("year")["dln"].transform(
        lambda s: s.clip(s.quantile(lo), s.quantile(hi)))
    return out

def aggregate(df, col):
    rows = []
    for t, g in df.groupby("year"):
        c = g.countyid.to_numpy(); d = g[col].to_numpy(float)
        gt  = np.array([GVP.get((int(x), int(t)),   np.nan) for x in c])
        gtm = np.array([GVP.get((int(x), int(t)-1), np.nan) for x in c])
        ok = np.isfinite(gt) & np.isfinite(gtm) & np.isfinite(d)
        c, d, gt, gtm = c[ok], d[ok], gt[ok], gtm[ok]
        if len(d) == 0: continue
        w = 0.5*(gt/gt.sum() + gtm/gtm.sum()); w /= w.sum()
        rows.append((int(t), len(d), float(d.mean()), float((w*d).sum())))
    o = pd.DataFrame(rows, columns=["year","n","simple","weighted"]).sort_values("year")
    o["cum_w"] = o["weighted"].cumsum()
    o["grw_w"] = 100*(np.exp(o["weighted"])-1)
    o["grw_s"] = 100*(np.exp(o["simple"])-1)
    return o

res = {}
for tag, df in [("DEA", dea), ("SOL", sol)]:
    w = winsor_by_year(df)
    raw = aggregate(w, "dln"); win = aggregate(w, "dln_w")
    res[tag] = dict(raw=raw, win=win)
    print(f"[{tag}] national TFP growth %/yr and volatility:")
    for lab, a in [("raw ", raw), ("wins", win)]:
        yr = a.year.iloc[-1]-a.year.iloc[0]
        print(f"   {lab}: weighted mean/yr={100*a.cum_w.iloc[-1]/yr:+.2f}%  "
              f"annual sd(weighted)={a.grw_w.std():.2f}  (raw-annual-mean {a.grw_w.mean():+.2f})")

# ---- assemble annual table (winsorized, weighted) -------------------------
ann = res["DEA"]["win"][["year","n","grw_w","cum_w"]].rename(columns={"grw_w":"dea_grw_wins","cum_w":"dea_cum_wins"})
ann = ann.merge(res["SOL"]["win"][["year","grw_w","cum_w"]].rename(columns={"grw_w":"sol_grw_wins","cum_w":"sol_cum_wins"}), on="year", how="outer")
ann = ann.merge(res["DEA"]["raw"][["year","grw_w"]].rename(columns={"grw_w":"dea_grw_raw"}), on="year", how="outer").sort_values("year")
ann.to_csv(os.path.join(C.CLEAN_DIR, "tfp_winsorized_annual.csv"), index=False)

# ---- five-stage summary (winsorized weighted) -----------------------------
def stage_of(y):
    for name, a, b in STAGES:
        if a <= y <= b: return name
    return "NA"
srows = []
for tag in ("DEA","SOL"):
    a = res[tag]["win"].copy(); a["stage"] = a.year.map(stage_of)
    for name,_,_ in STAGES:
        s = a[a.stage==name]
        if len(s)==0: continue
        srows.append(dict(method=tag, stage=name, yrs=f"{s.year.min()}-{s.year.max()}",
                          mean_grw_pct=round(100*(np.exp(s.weighted.mean())-1),2),
                          annual_sd=round(s.grw_w.std(),2)))
stage = pd.DataFrame(srows)
stage.to_csv(os.path.join(C.CLEAN_DIR, "tfp_stage_summary.csv"), index=False)
print("\nFIVE-STAGE mean TFP growth (winsorized, output-weighted):")
piv = stage[stage.method=="DEA"][["stage","yrs","mean_grw_pct","annual_sd"]]
print(piv.to_string(index=False))
print("\n(SOL cross-check):")
print(stage[stage.method=="SOL"][["stage","mean_grw_pct","annual_sd"]].to_string(index=False))

# ---- figure ---------------------------------------------------------------
colors = ["#eef7ff","#fff3f0","#eef7ff","#fff3f0","#eef7ff"]
fig,(a1,a2)=plt.subplots(2,1,figsize=(11,8),sharex=True)
for (name,y0,y1),cc in zip(STAGES,colors):
    for ax in (a1,a2): ax.axvspan(y0-0.5,y1+0.5,color=cc,zorder=0)
    a1.text((y0+y1)/2, 1.02, name.split(" ",1)[1], ha="center", va="bottom",
            transform=a1.get_xaxis_transform(), fontsize=7.5, color="0.35")
d=res["DEA"]; s=res["SOL"]
a1.plot(d["raw"].year, 100*d["raw"].cum_w, "--", color="0.6", lw=1, label="DEA raw (cum)")
a1.plot(d["win"].year, 100*d["win"].cum_w, "-", color="#d62728", lw=2, label="DEA winsorized (cum)")
a1.plot(s["win"].year, 100*s["win"].cum_w, "-", color="#1f77b4", lw=1.6, label="Solow-CRS winsorized (cum)")
a1.set_ylabel("cumulative ln TFP x100"); a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=.25)
a1.set_title("China agricultural TFP by five reform stages (winsorized 1/99 by year, output-weighted)")
a2.plot(d["raw"].year, d["raw"].grw_w, "--", color="0.6", lw=1, label=f"DEA raw YoY (sd {d['raw'].grw_w.std():.1f})")
a2.plot(d["win"].year, d["win"].grw_w, "-o", ms=3, color="#d62728", label=f"DEA winsorized YoY (sd {d['win'].grw_w.std():.1f})")
a2.axhline(0,color="gray",lw=.6); a2.set_ylabel("annual TFP growth %"); a2.set_xlabel("year")
a2.legend(fontsize=8); a2.grid(alpha=.25)
fig.tight_layout()
pth=os.path.join(C.FIG_DIR,"fig_tfp_winsorized_stages.png")
fig.savefig(pth,dpi=200); plt.close(fig)
print("\nsaved:",pth,"\ntables: tfp_winsorized_annual.csv, tfp_stage_summary.csv")
