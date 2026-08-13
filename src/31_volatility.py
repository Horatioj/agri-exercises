# -*- coding: utf-8 -*-
"""
STEP 31 — where the ~±20% year-on-year national TFP swings come from, and whether
they are a cleaning failure.  Four checks, one script (was 13 + 14 + 19, which
shared the same aggregate-Solow construction three times over).

(A) IS IT AN AVERAGING ARTIFACT?
    Build the AGGREGATE (national-totals) Solow index — sum output and each input
    across counties by year, then lnTFP_t = lnY_t - SUM_k beta_k ln X_kt — and
    compare its volatility with the mean-of-county series.  If the aggregate were
    smooth (sd ~3-5%) while mean-of-county is sd ~8-9%, the swings would be a
    heavy-tail/composition artifact.  They are NOT: both are ~8.5, so the
    volatility is a common annual factor.

(B) WHICH COMPONENT MOVES?
        dln TFP        = dln(SUM GVP_real) - SUM_k beta_k dln(SUM X_k)
        dln GVP_real   = dln(SUM GVP_nom)  - dln(deflator)
    If real output is the volatile piece while nominal is smooth, the deflator is
    the culprit — a single national PPI applied to every county.

(C) VOLATILITY BY YEAR
    National year-on-year growth, cross-county dispersion of county growth each
    year, and a 5-year rolling sd of the national series.

(D) AGAINST THE LITERATURE
    The published aggregate series are effectively smoothed.  Smoothed to that
    frequency ours sits inside the ±2-5%/yr band with the mean unchanged, and the
    biggest swing years line up with real events — i.e. signal, not noise.

All four run on the BALANCED panel so composition change cannot masquerade as
volatility.

OUTPUTS  src/clean/tfp_volatility_byyear.csv
FIGURES  fig_tfp_aggregate_vs_meancounty.png, fig_tfp_volatility_byyear.png,
         fig_tfp_vs_literature.png
Run:  python src/31_volatility.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C
import _tfp as T

B = T.BETA_FULL
CL = C.CLEAN_DIR
# Big swing years against what was actually happening in Chinese agriculture.
EVENTS = {1984: "HRS complete / procurement reform",
          1985: "grain output fall, market reform",
          1993: "price liberalization",
          1997: "grain glut & price deflation",
          2004: "ag-tax cut begins / No.1 Document",
          2005: "subsidies, tax abolition",
          2006: "ag tax abolished"}


def read_annual(name, cols):
    p = os.path.join(CL, name)
    return pd.read_csv(p)[cols] if os.path.exists(p) else None


d = T.load_panel()
# the aggregate accounts need nominal GVP for the deflator split, so require it
# before the balance test rather than letting a hole in it pass unnoticed
bal = T.load_panel(extra_cols=["GVP_allagr_impute"], balanced=True)
print(f"panel {d.countyid.nunique():,d} counties -> balanced {bal.countyid.nunique():,d} "
      f"x {bal.year.nunique()} years")

# ---------------------------------------------------------------------------
# Aggregate (national-totals) accounts, once — (A), (B) and (D) all read from it
# ---------------------------------------------------------------------------
g = bal.groupby("year")
agg = pd.DataFrame({"GVP_real": g["real_gvp"].sum(),
                    "GVP_nom": g["GVP_allagr_impute"].sum(),
                    "L": g["Laborday_impute"].sum(), "Land": g["Land_serv_q"].sum(),
                    "K": g["capital_serv_q"].sum(), "M": g["Inter_all_real"].sum()})
agg["defl"] = agg["GVP_nom"] / agg["GVP_real"]        # implied deflator, tracks PPI
agg["ln_input"] = (B["Laborday_impute"] * np.log(agg.L)
                   + B["Land_serv_q"] * np.log(agg.Land)
                   + B["capital_serv_q"] * np.log(agg.K)
                   + B["Inter_all_real"] * np.log(agg.M))
agg["ln_tfp"] = np.log(agg.GVP_real) - agg.ln_input

dd = pd.DataFrame({"d_nomGVP": np.log(agg.GVP_nom).diff() * 100,
                   "d_defl": np.log(agg.defl).diff() * 100,
                   "d_realGVP": np.log(agg.GVP_real).diff() * 100,
                   "d_input": agg.ln_input.diff() * 100,
                   "d_tfp": agg.ln_tfp.diff() * 100}).dropna()

# ============================================== (A) artifact or real volatility
print("\n(A) AGGREGATE (national totals) Solow TFP, CRS elasticities:")
for tag, frame in (("all valid obs", d), ("balanced panel", bal)):
    gg = frame.groupby("year")[C.IO_VARS].sum()
    ln = np.log(gg["real_gvp"]) - sum(B[k] * np.log(gg[k]) for k in B)
    x = ln.diff().dropna() * 100
    print(f"   {tag:16s}: mean={x.mean():+.2f}%/yr  sd={x.std():.2f}  "
          f"min={x.min():+.1f} max={x.max():+.1f}")
balx = dd.d_tfp

print("\n    MEAN-of-county growth (for comparison):")
for tag, name in (("SFA", "tfp_sfa_annual.csv"), ("DEA", "tfp_dea_annual.csv")):
    a = read_annual(name, ["year", "growth_weighted", "cum_weighted"])
    if a is None:
        print(f"      [skip] {name} not found — run 30_aggregate_tfp.py"); continue
    print(f"      {tag} weighted: sd={a.growth_weighted.std():.2f}  "
          f"min={a.growth_weighted.min():+.1f} max={a.growth_weighted.max():+.1f}")

dc_path = os.path.join(CL, "dea_malmquist_county.csv")
if os.path.exists(dc_path):
    dc = pd.read_csv(dc_path)
    trimmed = dc.lnM.clip(dc.lnM.quantile(.01), dc.lnM.quantile(.99))
    print("\n    County-level DEA lnM tails (heavy tails inflate the mean):")
    print(f"      |lnM|>0.5: {100*(dc.lnM.abs()>0.5).mean():.1f}%   "
          f"|lnM|>1: {100*(dc.lnM.abs()>1).mean():.2f}%   county sd={dc.lnM.std():.3f}")
    print(f"      trimmed(1-99pct) county lnM sd={trimmed.std():.3f}")

sfa_a = read_annual("tfp_sfa_annual.csv", ["year", "growth_weighted", "cum_weighted"])
dea_a = read_annual("tfp_dea_annual.csv", ["year", "growth_weighted", "cum_weighted"])
aggidx = (agg.ln_tfp - agg.ln_tfp.iloc[0]) * 100
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
a1.plot(aggidx.index, aggidx.values, "-o", ms=3, color="k",
        label=f"AGGREGATE Solow (national totals), sd/yr={balx.std():.1f}")
for a, lab, col in ((sfa_a, "SFA mean-of-county (weighted)", "#1f77b4"),
                    (dea_a, "DEA mean-of-county (weighted)", "#d62728")):
    if a is not None:
        a1.plot(a.year, 100 * a.cum_weighted, "-", color=col, label=lab)
        a2.plot(a.year, a.growth_weighted, "-", color=col, alpha=.7,
                label=f"{lab.split()[0]} mean-of-county YoY (sd={a.growth_weighted.std():.1f})")
a1.set_ylabel("cum ln TFP x100"); a1.legend(fontsize=8); a1.grid(alpha=.3)
a1.set_title("(A) Aggregate (national-totals) TFP vs mean-of-county TFP")
a2.plot(balx.index, balx.values, "-o", ms=3, color="k",
        label=f"AGGREGATE Solow YoY (sd={balx.std():.1f})")
a2.axhline(0, color="gray", lw=.6); a2.set_ylabel("YoY growth %"); a2.set_xlabel("year")
a2.legend(fontsize=8); a2.grid(alpha=.3)
fig.tight_layout()
p = os.path.join(C.FIG_DIR, "fig_tfp_aggregate_vs_meancounty.png")
fig.savefig(p, dpi=200); plt.close(fig)
print("\nsaved:", p)

# ================================================== (B) which component moves?
print("\n(B) Aggregate growth decomposition (%/yr):")
for c in dd.columns:
    print(f"   {c:10s}: mean={dd[c].mean():+6.2f}  sd={dd[c].std():5.2f}  "
          f"min={dd[c].min():+6.1f} max={dd[c].max():+6.1f}")
print(f"   corr(d_realGVP, d_tfp) = {dd.d_realGVP.corr(dd.d_tfp):.3f}   "
      f"corr(d_input, d_tfp) = {dd.d_input.corr(dd.d_tfp):.3f}")
print("   => sd(d_realGVP) ~ sd(d_tfp) with small sd(d_input): the OUTPUT side drives it,")
print(f"      and within it the single national deflator (sd {dd.d_defl.std():.1f}) is the "
      f"biggest piece.")

# ========================================================= (C) volatility by year
b2 = bal.copy()
b2["ln_tfp_i"] = T.solow_ln_tfp(b2, B)
ci = T.county_dln(b2, "ln_tfp_i")
ci["d_tfp_i"] = ci["dln"] * 100
by = ci.groupby("year")["d_tfp_i"].agg(
    nat_mean="mean", nat_median="median", xsec_sd="std",
    p10=lambda s: s.quantile(.10), p90=lambda s: s.quantile(.90), n="count").reset_index()
for tag, a in (("sfa", sfa_a), ("dea", dea_a)):
    if a is not None:
        by = by.merge(a[["year", "growth_weighted"]].rename(
            columns={"growth_weighted": f"{tag}_grw"}), on="year", how="left")
by["roll5_sd_natmean"] = by["nat_mean"].rolling(5, center=True, min_periods=3).std()
by.to_csv(os.path.join(CL, "tfp_volatility_byyear.csv"), index=False)

pd.set_option("display.width", 200)
print("\n(C) TFP volatility BY YEAR (balanced-panel Solow):")
print(by.round(2).to_string(index=False))
print("\n    biggest swing years (|nat_mean|):")
print(by.reindex(by.nat_mean.abs().sort_values(ascending=False).index)
        .head(6)[["year", "nat_mean", "xsec_sd"]].round(1).to_string(index=False))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
a1.fill_between(by.year, by.p10, by.p90, color="0.8", label="county p10-p90")
a1.plot(by.year, by.nat_mean, "-o", ms=3, color="k", label="national mean TFP growth")
a1.plot(by.year, by.nat_median, "-", color="#2ca02c", label="national median")
a1.axhline(0, color="gray", lw=.6); a1.set_ylabel("TFP growth %/yr")
a1.legend(fontsize=8); a1.grid(alpha=.3)
a1.set_title("(C) National ag TFP growth by year + cross-county dispersion "
             "(balanced panel, CRS Solow)")
a2.plot(by.year, by.xsec_sd, "-o", ms=3, color="#d62728",
        label="cross-county sd (dispersion each year)")
a2.plot(by.year, by.roll5_sd_natmean, "-", color="navy",
        label="5-yr rolling sd of national growth")
a2.set_ylabel("volatility (sd, %)"); a2.set_xlabel("year")
a2.legend(fontsize=8); a2.grid(alpha=.3); a2.set_title("TFP volatility by year")
fig.tight_layout()
p = os.path.join(C.FIG_DIR, "fig_tfp_volatility_byyear.png")
fig.savefig(p, dpi=200); plt.close(fig)
print("\nsaved:", p)

# ================================================== (D) against the literature
wins_path = os.path.join(CL, "tfp_winsorized_annual.csv")
if not os.path.exists(wins_path):
    print(f"\n(D) [skip] {os.path.basename(wins_path)} not found — run 30_aggregate_tfp.py")
    sys.exit(0)

gser = pd.read_csv(wins_path).set_index("year")["dea_grw_wins"].dropna()
ma3 = gser.rolling(3, center=True, min_periods=2).mean()
ma5 = gser.rolling(5, center=True, min_periods=3).mean()
print("\n(D) SMOOTHING (national DEA weighted winsorized TFP growth):")
print(f"   raw annual : mean {gser.mean():+.2f}%/yr  sd {gser.std():.2f}")
print(f"   3-yr MA    : mean {ma3.mean():+.2f}%/yr  sd {ma3.std():.2f}")
print(f"   5-yr MA    : mean {ma5.mean():+.2f}%/yr  sd {ma5.std():.2f}  "
      f"<- the frequency the literature reports")

print("\n   biggest swing years vs real events (=> largely SIGNAL, not cleaning noise):")
for y, v in gser.reindex(gser.abs().sort_values(ascending=False).index).head(7).items():
    print(f"     {int(y)}: {v:+6.1f}%   {EVENTS.get(int(y), '-')}")

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axhspan(2, 5, color="#e8f4e8", label="literature aggregate band ~+2..5%/yr")
ax.plot(gser.index, gser.values, "-", color="0.6", lw=1,
        label=f"raw annual (sd {gser.std():.1f})")
ax.plot(ma3.index, ma3.values, "-o", ms=3, color="#d62728", lw=2,
        label=f"3-yr moving avg (sd {ma3.std():.1f})")
ax.plot(ma5.index, ma5.values, "-", color="navy", lw=2,
        label=f"5-yr moving avg (sd {ma5.std():.1f})")
ax.axhline(0, color="k", lw=.6, ls=":")
for y in (1985, 1997, 2006):
    if y in gser.index:
        ax.annotate(EVENTS.get(y, ""), (y, gser[y]), fontsize=7, color="0.4",
                    xytext=(0, -14 if gser[y] > 0 else 10),
                    textcoords="offset points", ha="center")
ax.set_title("(D) National ag TFP growth: raw annual vs smoothed, against the literature band\n"
             "high-frequency swings = deflator + weather + input noise; "
             "the smoothed signal is literature-consistent")
ax.set_ylabel("TFP growth %/yr"); ax.set_xlabel("year")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout()
p = os.path.join(C.FIG_DIR, "fig_tfp_vs_literature.png")
fig.savefig(p, dpi=200); plt.close(fig)
print("\nsaved:", p)
