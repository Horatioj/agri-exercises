# -*- coding: utf-8 -*-
"""
Build national ag TFP from the CONVERGED BC92-tnormal SFA on the 1000-county
subsample, and compare (same counties, winsorized 1/99 by year, output-weighted)
against DEA Malmquist and Solow-CRS, cut by the five reform stages.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

B = {"Laborday_impute": 0.448, "Land_serv_q": 0.081, "capital_serv_q": 0.038, "Inter_all_real": 0.433}
STAGES = [("1 Reform take-off",1981,1984),("2 Stagnation",1985,1988),("3 Recovery",1989,1996),
          ("4 Adjustment",1997,2003),("5 Subsidy era",2004,2016)]

sfa = pd.read_csv(os.path.join(C.CLEAN_DIR, "sfa_sub1000_county_year.csv"))
subset = set(sfa.countyid.unique())
print(f"subsample counties={len(subset)}  obs={len(sfa)}")
print(f"BC92 mean technical efficiency te_jlms = {sfa.te_jlms.mean():.3f}  (median {sfa.te_jlms.median():.3f}) "
      f"-> LEVELS implausibly low (heterogeneity->inefficiency; no FE control)")

# GVP weights from the same subsample rows
GVP = {(int(r.countyid), int(r.year)): float(r.real_gvp) for r in sfa.itertuples()}

def county_dln_sfa():
    s = sfa.sort_values(["countyid","year"]).copy()
    s["dln"] = s.groupby("countyid")["ln_tfp_chen"].diff()
    s["gap"] = s.groupby("countyid")["year"].diff()
    return s[(s.gap==1)&s.dln.notna()][["countyid","year","dln"]]

def county_dln_solow():
    p = C.pd_read = pd.read_csv(C.CLEAN_PANEL)
    iov=list(B); p=p.dropna(subset=iov+["real_gvp"]); p=p[(p[iov]>0).all(axis=1)]
    p=p[p.countyid.isin(subset)].sort_values(["countyid","year"]).copy()
    p["ln_tfp"]=np.log(p.real_gvp)-sum(B[k]*np.log(p[k]) for k in B)
    p["dln"]=p.groupby("countyid")["ln_tfp"].diff(); p["gap"]=p.groupby("countyid")["year"].diff()
    return p[(p.gap==1)&p.dln.notna()][["countyid","year","dln"]]

def county_dln_dea():
    d=pd.read_csv(os.path.join(C.CLEAN_DIR,"dea_malmquist_county.csv")).rename(columns={"lnM":"dln"})
    return d[d.countyid.isin(subset)][["countyid","year","dln"]]

def winsor_agg(df):
    df=df.copy()
    df["dln"]=df.groupby("year")["dln"].transform(lambda s:s.clip(s.quantile(.01),s.quantile(.99)))
    rows=[]
    for t,g in df.groupby("year"):
        c=g.countyid.to_numpy(); d=g.dln.to_numpy(float)
        gt=np.array([GVP.get((int(x),int(t)),np.nan) for x in c])
        gtm=np.array([GVP.get((int(x),int(t)-1),np.nan) for x in c])
        ok=np.isfinite(gt)&np.isfinite(gtm)&np.isfinite(d); c,d,gt,gtm=c[ok],d[ok],gt[ok],gtm[ok]
        if len(d)==0: continue
        w=0.5*(gt/gt.sum()+gtm/gtm.sum()); w/=w.sum()
        rows.append((int(t),len(d),float((w*d).sum())))
    o=pd.DataFrame(rows,columns=["year","n","dln"]).sort_values("year")
    o["cum"]=o.dln.cumsum(); o["grw"]=100*(np.exp(o.dln)-1); return o

M={"SFA(BC92)":winsor_agg(county_dln_sfa()),"DEA":winsor_agg(county_dln_dea()),"Solow":winsor_agg(county_dln_solow())}
def stage_of(y):
    for n,a,b in STAGES:
        if a<=y<=b: return n
    return "NA"

print("\nFIVE-STAGE mean TFP growth %/yr (winsorized, output-weighted, same 1000 counties):")
hdr=f"{'stage':18s}"+"".join(f"{k:>12s}" for k in M); print(hdr)
tbl={}
for n,a,b in STAGES:
    line=f"{n:18s}"
    for k,df in M.items():
        s=df[(df.year>=a)&(df.year<=b)]
        v=100*(np.exp(s.dln.mean())-1) if len(s) else np.nan
        tbl[(n,k)]=v; line+=f"{v:>+12.2f}"
    print(line)
ov=f"{'overall %/yr':18s}"
for k,df in M.items():
    yr=df.year.iloc[-1]-df.year.iloc[0]; ov+=f"{100*df.cum.iloc[-1]/yr:>+12.2f}"
print(ov)
sd=f"{'annual sd':18s}"
for k,df in M.items(): sd+=f"{df.grw.std():>12.2f}"
print(sd)

# save + figure
out=M["SFA(BC92)"][["year","grw","cum"]].rename(columns={"grw":"sfa_grw","cum":"sfa_cum"})
for k in ("DEA","Solow"):
    out=out.merge(M[k][["year","grw","cum"]].rename(columns={"grw":k.lower()+"_grw","cum":k.lower()+"_cum"}),on="year",how="outer")
out.sort_values("year").to_csv(os.path.join(C.CLEAN_DIR,"tfp_sub1000_annual.csv"),index=False)

fig,(a1,a2)=plt.subplots(2,1,figsize=(11,8),sharex=True)
cols={"SFA(BC92)":"#7b3294","DEA":"#d62728","Solow":"#1f77b4"}
for (n,y0,y1),cc in zip(STAGES,["#eef7ff","#fff3f0","#eef7ff","#fff3f0","#eef7ff"]):
    for ax in (a1,a2): ax.axvspan(y0-.5,y1+.5,color=cc,zorder=0)
    a1.text((y0+y1)/2,1.02,n.split(" ",1)[1],ha="center",va="bottom",transform=a1.get_xaxis_transform(),fontsize=7.5,color="0.35")
for k,df in M.items():
    yr=df.year.iloc[-1]-df.year.iloc[0]
    a1.plot(df.year,100*df.cum,"-",color=cols[k],lw=1.8,label=f"{k} ({100*df.cum.iloc[-1]/yr:+.2f}%/yr)")
    a2.plot(df.year,df.grw,"-o",ms=2.5,color=cols[k],lw=1,label=k)
a1.set_ylabel("cum lnTFP x100"); a1.legend(fontsize=8,loc="upper left"); a1.grid(alpha=.25)
a1.set_title("1000-county sample (IM/Tibet/Qinghai/Xinjiang excluded): converged BC92-tnormal SFA vs DEA vs Solow\nwinsorized 1/99 by year, output-weighted, five reform stages")
a2.axhline(0,color="gray",lw=.6); a2.set_ylabel("annual TFP growth %"); a2.set_xlabel("year"); a2.legend(fontsize=8); a2.grid(alpha=.25)
fig.tight_layout(); pth=os.path.join(C.FIG_DIR,"fig_tfp_sub1000_stages.png"); fig.savefig(pth,dpi=200); plt.close(fig)
print("\nsaved:",pth,"\ntable: src/clean/tfp_sub1000_annual.csv")
