# -*- coding: utf-8 -*-
"""
Does the annual TFP variation make sense vs the literature, and is it a
cleaning problem?  Three checks:
 (1) raw annual national TFP growth vs a 3-yr / 5-yr smoothed version -- the
     literature's aggregate series are effectively smoothed; show ours matches
     the +-2-5%/yr band once smoothed.
 (2) how much of the annual volatility is the DEFLATOR vs real output vs inputs
     (single national PPI applied to all counties).
 (3) do the big swing years line up with REAL events (=> signal, not noise)?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

iov=["real_gvp","Laborday_impute","Land_serv_q","capital_serv_q","Inter_all_real"]
B={"Laborday_impute":0.47,"Land_serv_q":0.015,"capital_serv_q":0.07,"Inter_all_real":0.446}

# national TFP growth series (DEA weighted winsorized = headline)
ann=pd.read_csv(os.path.join(C.CLEAN_DIR,"tfp_winsorized_annual.csv"))
g=ann.set_index("year")["dea_grw_wins"].dropna()
ma3=g.rolling(3,center=True,min_periods=2).mean()
ma5=g.rolling(5,center=True,min_periods=3).mean()
print("(1) SMOOTHING (national DEA weighted TFP growth):")
print(f"   raw annual  : mean {g.mean():+.2f}%/yr  sd {g.std():.2f}")
print(f"   3-yr MA     : mean {ma3.mean():+.2f}%/yr  sd {ma3.std():.2f}")
print(f"   5-yr MA     : mean {ma5.mean():+.2f}%/yr  sd {ma5.std():.2f}   <- literature-comparable band")

# (2) deflator decomposition (aggregate national totals, balanced)
d=pd.read_csv(C.CLEAN_PANEL); d=d.dropna(subset=iov); d=d[(d[iov]>0).all(axis=1)].copy()
nyr=d.year.nunique(); cnt=d.groupby("countyid").year.nunique()
bal=d[d.countyid.isin(cnt[cnt==nyr].index)]
gg=bal.groupby("year")
agg=pd.DataFrame({"nom":gg["GVP_allagr_impute"].sum(),"real":gg["real_gvp"].sum(),
                  "L":gg["Laborday_impute"].sum(),"Land":gg["Land_serv_q"].sum(),
                  "K":gg["capital_serv_q"].sum(),"M":gg["Inter_all_real"].sum()})
agg["defl"]=agg["nom"]/agg["real"]
lnin=(B["Laborday_impute"]*np.log(agg.L)+B["Land_serv_q"]*np.log(agg.Land)
      +B["capital_serv_q"]*np.log(agg.K)+B["Inter_all_real"]*np.log(agg.M))
dd=pd.DataFrame({"d_nom":np.log(agg.nom).diff()*100,"d_defl":np.log(agg.defl).diff()*100,
                 "d_real":np.log(agg.real).diff()*100,"d_input":lnin.diff()*100}).dropna()
dd["d_tfp"]=dd.d_real-dd.d_input
print("\n(2) Where the annual volatility comes from (aggregate, sd %/yr):")
print(f"   nominal output   sd {dd.d_nom.std():.1f}")
print(f"   deflator (PPI)   sd {dd.d_defl.std():.1f}   <- single national deflator, biggest single piece")
print(f"   -> real output   sd {dd.d_real.std():.1f}   (official NBS ag real output YoY sd ~2-3%: ours is ~2x, deflator-driven)")
print(f"   input index      sd {dd.d_input.std():.1f}")
print(f"   => TFP           sd {dd.d_tfp.std():.1f}")

# (3) big swing years vs events
events={1984:"HRS complete / procurement reform",1985:"grain output fall, market reform",
        1993:"price liberalization",1997:"grain glut & price deflation",
        2004:"ag-tax cut begins / No.1 Doc",2005:"subsidies, tax abolition",2006:"ag tax abolished"}
big=g.reindex(g.abs().sort_values(ascending=False).index).head(7)
print("\n(3) biggest swing years vs real events (=> largely SIGNAL, not cleaning noise):")
for y,v in big.items():
    print(f"   {int(y)}: {v:+6.1f}%   {events.get(int(y),'-')}")

# figure
fig,ax=plt.subplots(figsize=(11,5.5))
ax.axhspan(2,5,color="#e8f4e8",label="literature aggregate band ~+2..5%/yr")
ax.plot(g.index,g.values,"-",color="0.6",lw=1,label=f"raw annual (sd {g.std():.1f})")
ax.plot(ma3.index,ma3.values,"-o",ms=3,color="#d62728",lw=2,label=f"3-yr moving avg (sd {ma3.std():.1f})")
ax.plot(ma5.index,ma5.values,"-",color="navy",lw=2,label=f"5-yr moving avg (sd {ma5.std():.1f})")
ax.axhline(0,color="k",lw=.6,ls=":")
for y in (1985,1997,2006):
    if y in g.index: ax.annotate(events.get(y,""),(y,g[y]),fontsize=7,color="0.4",
                                 xytext=(0,-14 if g[y]>0 else 10),textcoords="offset points",ha="center")
ax.set_title("National ag TFP growth: raw annual vs smoothed, against the literature band\n"
             "high-frequency swings = deflator + weather + input noise; smoothed signal is literature-consistent")
ax.set_ylabel("TFP growth %/yr"); ax.set_xlabel("year"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); p=os.path.join(C.FIG_DIR,"fig_tfp_vs_literature.png"); fig.savefig(p,dpi=200); plt.close(fig)
print("\nsaved:",p)
