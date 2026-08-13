# -*- coding: utf-8 -*-
"""
Plot the DEA Malmquist decomposition produced by 23_dea_sfa_framework.do:
  (A) cumulative TFP = efficiency change x technical change, by year
  (B) stage bar chart: how much of each stage's TFP growth was frontier shift
      (innovation) vs catch-up (efficiency change)
Inputs : src/clean/dea_decomp_byyear.csv, dea_decomp_bystage.csv
Output : src/figures/fig_dea_decomposition.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

yr = pd.read_csv(os.path.join(C.CLEAN_DIR, "dea_decomp_byyear.csv"))
st = pd.read_csv(os.path.join(C.CLEAN_DIR, "dea_decomp_bystage.csv"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))

# ---- (A) cumulative decomposition ----------------------------------------
a1.plot(yr.year, 100*yr.cum_tfp, "-", color="#111111", lw=2.6, label="TFP (Malmquist)")
a1.plot(yr.year, 100*yr.cum_tc, "-", color="#1f77b4", lw=2.0, label="Technical change (frontier shift)")
a1.plot(yr.year, 100*yr.cum_ec, "-", color="#d62728", lw=2.0, label="Efficiency change (catch-up)")
a1.axhline(0, color="0.5", lw=.7, ls=":")
a1.set_ylabel("cumulative ln index ×100"); a1.set_xlabel("year")
a1.set_title("(A) DEA Malmquist decomposition, cumulative\nTFP = technical change + efficiency change")
a1.legend(fontsize=8); a1.grid(alpha=.3)

# ---- (B) stage bars -------------------------------------------------------
lab = [s.split(" ", 1)[1] if " " in s else s for s in st.stage.astype(str)]
x = np.arange(len(st)); w = 0.36
a2.bar(x - w/2, st.tech_pct, w, color="#1f77b4", label="Technical change")
a2.bar(x + w/2, st.effch_pct, w, color="#d62728", label="Efficiency change")
a2.plot(x, st.tfp_pct, "o-", color="#111111", lw=2, ms=7, label="Net TFP growth")
a2.axhline(0, color="0.4", lw=.8)
a2.set_xticks(x); a2.set_xticklabels(lab, fontsize=8, rotation=12)
a2.set_ylabel("%/yr"); a2.set_title("(B) By reform stage: innovation vs catch-up")
a2.legend(fontsize=8); a2.grid(alpha=.3, axis="y")
for i, v in enumerate(st.tfp_pct):
    a2.annotate(f"{v:+.1f}", (i, v), fontsize=7.5, fontweight="bold",
                xytext=(0, 9), textcoords="offset points", ha="center")

fig.suptitle("China county agriculture: DEA Malmquist decomposition (sequential NIRS, output-weighted)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(C.FIG_DIR, "fig_dea_decomposition.png")
fig.savefig(out, dpi=160); plt.close(fig)
print("saved:", out)
print(st[["stage", "tfp_pct", "effch_pct", "tech_pct"]].to_string(index=False))
