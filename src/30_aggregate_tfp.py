# -*- coding: utf-8 -*-
"""
STEP 30 — county TFP growth -> national annual series, for every estimator, under
both weightings, raw and winsorized, cut by the five reform stages.

This is the single aggregation step.  It replaces three scripts that each carried
their own copy of the Tornqvist aggregator (12_aggregate_compare, 15_tfp_winsorized,
18_sub1000_compare); the aggregator now lives once in _tfp.py, so a change to the
weighting rule cannot silently apply to one panel of the paper and not another.

ESTIMATORS
  DEA    sequential-NIRS Malmquist, county lnM        (20_dea_vs_fe_tfp.py)
  SFA    BC92 land-normalized, d ln_tfp_chen          (21_sfa_bc92.do, full panel)
  SOL    CRS Solow residual, county-level             (transparent cross-check)

WEIGHTING
  simple   : equal weight per county
  weighted : output-share Tornqvist, w_it = 0.5*(s_{i,t-1} + s_{i,t})
             -- the standard growth-accounting aggregator.  Equal weighting would
             give a tiny Tibetan county the same weight as a large Henan grain
             producer, so `weighted` is the headline.

WINSORIZING
  County growth is winsorized 1/99 WITHIN year before aggregating (SFA.do's
  winsor2, applied by year).  It barely moves the national volatility
  (8.82 -> 8.83 sd), which is itself the finding: the swings are a common annual
  factor, not a handful of extreme counties.  Both versions are reported.

SUBSAMPLE COMPARISON
  The SFA1k block re-aggregates DEA and Solow on the SAME 1,000 counties, so the
  BC92-vs-DEA gap is not contaminated by sample composition.

OUTPUTS (src/clean/)
  tfp_<dea|sfa|sol>_annual.csv  per-estimator national annual series
  tfp_compare_annual.csv        merged DEA vs SFA table
  tfp_winsorized_annual.csv     raw vs winsorized, weighted
  tfp_stage_summary.csv         five-stage means and volatility
  tfp_sub1000_annual.csv        1,000-county comparison (if that SFA run exists)
FIGURES (src/figures/)
  fig_tfp_sfa_vs_dea.png  fig_tfp_winsorized_stages.png  fig_tfp_sub1000_stages.png
Run:  python src/30_aggregate_tfp.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C
import _tfp as T

CL = C.CLEAN_DIR
COL = {"DEA": "#d62728", "SFA": "#1f77b4", "SOL": "#2ca02c", "SFA1k": "#7b3294"}


# ---------------------------------------------------------------- county dln
def dln_dea(counties=None):
    """The Malmquist lnM already IS the t-1 -> t growth, so no differencing."""
    p = os.path.join(CL, "dea_malmquist_county.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).rename(columns={"lnM": "dln"})[["countyid", "year", "dln"]]
    return d if counties is None else d[d.countyid.isin(counties)].copy()


def dln_sfa(csv, counties=None):
    p = os.path.join(CL, csv)
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p)
    if counties is not None:
        s = s[s.countyid.isin(counties)]
    return T.county_dln(s, "ln_tfp_chen")


def dln_solow(panel, beta=T.BETA_FULL, counties=None):
    p = panel if counties is None else panel[panel.countyid.isin(counties)]
    p = p.copy()
    p["ln_tfp"] = T.solow_ln_tfp(p, beta)
    return T.county_dln(p, "ln_tfp")


# --------------------------------------------------------------------- setup
panel = T.load_panel()
GVP = T.gvp_weights(panel)
print(f"panel: {panel.countyid.nunique():,d} counties x "
      f"{panel.year.min()}-{panel.year.max()}, {len(panel):,d} county-years")

series = {}
for tag, d in (("DEA", dln_dea()),
               ("SFA", dln_sfa("sfa_bc92_county_year.csv")),
               ("SOL", dln_solow(panel))):
    if d is None or d.empty:
        print(f"  [skip] {tag}: input not found")
        continue
    series[tag] = T.tornqvist_aggregate(d, GVP)

print("\nNational agricultural TFP growth (cleaned panel):")
for tag, a in series.items():
    T.summarise(a, tag)

# ------------------------------------------------------- A. per-estimator CSVs
for tag, a in series.items():
    a.to_csv(os.path.join(CL, f"tfp_{tag.lower()}_annual.csv"), index=False)

KEEP = ["year", "n", "growth_simple", "growth_weighted", "cum_simple", "cum_weighted"]
if {"SFA", "DEA"} <= series.keys():
    comp = None
    for tag in ("SFA", "DEA"):
        t = series[tag][KEEP].rename(
            columns={c: f"{tag.lower()}_{c}" for c in KEEP if c != "year"})
        comp = t if comp is None else comp.merge(t, on="year", how="outer")
    comp.sort_values("year").to_csv(os.path.join(CL, "tfp_compare_annual.csv"), index=False)

    # figure: cumulative index (A) + annual growth (B)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for tag in ("SFA", "DEA"):
        a = series[tag]
        for kind, ls, lw in (("weighted", "-", 1.8), ("simple", "--", 1.2)):
            ax1.plot(a.year, 100 * a[f"cum_{kind}"], ls, color=COL[tag], lw=lw,
                     label=f"{tag} {kind} ({T.cagr_pct(a, kind):+.2f}%/yr)")
        ax2.plot(a.year, a.growth_weighted, "-o", ms=3.5, color=COL[tag],
                 label=f"{tag} weighted")
    ax1.axhline(0, color="k", lw=.6, ls=":")
    ax1.set_ylabel("Cumulative ln TFP x100 (~% vs base)")
    ax1.set_title("China agricultural TFP: SFA (land-normalized CRS frontier) vs DEA "
                  "(NIRS sequential)\ncleaned county panel — simple mean vs output-share "
                  "(Tornqvist) weighting\n[SFA inefficiency ~0 (symmetric residuals) "
                  "=> SFA weighted=simple]")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(alpha=.3)
    ax2.axhline(0, color="k", lw=.6, ls=":")
    ax2.set_ylabel("Annual TFP growth (%)"); ax2.set_xlabel("year")
    ax2.set_title("Panel B: year-on-year growth (output-share weighted)")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig_tfp_sfa_vs_dea.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("saved:", p)

# ------------------------------------------- B. winsorized + five-stage summary
raw_dln = {"DEA": dln_dea(), "SOL": dln_solow(panel)}
raw_dln = {k: v for k, v in raw_dln.items() if v is not None and not v.empty}
wins, stages = {}, []
print("\nWinsorized 1/99 within year (weighted):")
for tag, d in raw_dln.items():
    w = T.winsorize_by_year(d)
    raw = T.tornqvist_aggregate(w, GVP, "dln")
    win = T.tornqvist_aggregate(w, GVP, "dln_w")
    wins[tag] = dict(raw=raw, win=win)
    print(f"  [{tag}] raw : {T.cagr_pct(raw):+.2f}%/yr  annual sd {raw.growth_weighted.std():.2f}")
    print(f"  [{tag}] wins: {T.cagr_pct(win):+.2f}%/yr  annual sd {win.growth_weighted.std():.2f}")
    stages.append(T.stage_table(win, tag))

if wins:
    ann = None
    for tag, r in wins.items():
        t = r["win"][["year", "n", "growth_weighted", "cum_weighted"]].rename(
            columns={"growth_weighted": f"{tag.lower()}_grw_wins",
                     "cum_weighted": f"{tag.lower()}_cum_wins"})
        t = t.merge(r["raw"][["year", "growth_weighted"]].rename(
            columns={"growth_weighted": f"{tag.lower()}_grw_raw"}), on="year", how="outer")
        if ann is None:
            ann = t
        else:
            ann = ann.merge(t.drop(columns=["n"], errors="ignore"), on="year", how="outer")
    ann.sort_values("year").to_csv(os.path.join(CL, "tfp_winsorized_annual.csv"), index=False)

    stage = pd.concat(stages, ignore_index=True)
    stage.to_csv(os.path.join(CL, "tfp_stage_summary.csv"), index=False)
    print("\nFIVE-STAGE mean TFP growth (winsorized, output-weighted):")
    print(stage[stage.method == "DEA"][["stage", "yrs", "mean_grw_pct", "annual_sd"]]
          .to_string(index=False))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for (name, y0, y1), cc in zip(T.STAGES, ["#eef7ff", "#fff3f0"] * 3):
        for ax in (a1, a2):
            ax.axvspan(y0 - .5, y1 + .5, color=cc, zorder=0)
        a1.text((y0 + y1) / 2, 1.02, name.split(" ", 1)[1], ha="center", va="bottom",
                transform=a1.get_xaxis_transform(), fontsize=7.5, color="0.35")
    for tag, r in wins.items():
        w = r["win"]
        a1.plot(w.year, 100 * w.cum_weighted, "-", color=COL[tag], lw=1.8,
                label=f"{tag} winsorized ({T.cagr_pct(w):+.2f}%/yr)")
        a1.plot(r["raw"].year, 100 * r["raw"].cum_weighted, ":", color=COL[tag], lw=1.1,
                label=f"{tag} raw")
        a2.plot(w.year, w.growth_weighted, "-o", ms=2.5, color=COL[tag], lw=1, label=tag)
    a1.set_ylabel("cum ln TFP x100"); a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=.25)
    a1.set_title("National ag TFP, winsorized 1/99 by year, output-weighted, five reform stages")
    a2.axhline(0, color="gray", lw=.6); a2.set_ylabel("annual TFP growth %")
    a2.set_xlabel("year"); a2.legend(fontsize=8); a2.grid(alpha=.25)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig_tfp_winsorized_stages.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("saved:", p)

# ------------------------------------- C. 1,000-county converged-BC92 comparison
sub_csv = os.path.join(CL, "sfa_sub1000_county_year.csv")
if os.path.exists(sub_csv):
    sub = pd.read_csv(sub_csv)
    counties = set(sub.countyid.unique())
    print(f"\n1,000-county subsample: {len(counties):,d} counties, {len(sub):,d} obs")
    print(f"  BC92 mean technical efficiency te_jlms = {sub.te_jlms.mean():.3f} "
          f"(median {sub.te_jlms.median():.3f})\n"
          f"  -> LEVELS implausibly low: without a heterogeneity term, persistent county\n"
          f"     differences are mislabelled as inefficiency.  Use BC92 for TFP GROWTH,\n"
          f"     not for efficiency levels (METHODOLOGY.md 4.3).")
    # weights from the same rows, so composition is identical across estimators
    gvp_sub = {(int(r.countyid), int(r.year)): float(r.real_gvp) for r in sub.itertuples()}
    M = {}
    for tag, d in (("SFA1k", T.county_dln(sub, "ln_tfp_chen")),
                   ("DEA", dln_dea(counties)),
                   ("SOL", dln_solow(panel, T.BETA_SUB1000, counties))):
        if d is None or d.empty:
            continue
        M[tag] = T.tornqvist_aggregate(T.winsorize_by_year(d), gvp_sub, "dln_w")

    print("\nFIVE-STAGE mean TFP growth %/yr (winsorized, output-weighted, same counties):")
    print(f"{'stage':20s}" + "".join(f"{k:>10s}" for k in M))
    for name, y0, y1 in T.STAGES:
        line = f"{name:20s}"
        for a in M.values():
            s = a[(a.year >= y0) & (a.year <= y1)]
            line += f"{100*(np.exp(s.dln_weighted.mean())-1):>+10.2f}" if len(s) else f"{'':>10s}"
        print(line)
    print(f"{'overall %/yr':20s}" + "".join(f"{T.cagr_pct(a):>+10.2f}" for a in M.values()))
    print(f"{'annual sd':20s}" + "".join(f"{a.growth_weighted.std():>10.2f}" for a in M.values()))

    out = None
    for tag, a in M.items():
        t = a[["year", "growth_weighted", "cum_weighted"]].rename(
            columns={"growth_weighted": f"{tag.lower()}_grw", "cum_weighted": f"{tag.lower()}_cum"})
        out = t if out is None else out.merge(t, on="year", how="outer")
    out.sort_values("year").to_csv(os.path.join(CL, "tfp_sub1000_annual.csv"), index=False)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for (name, y0, y1), cc in zip(T.STAGES, ["#eef7ff", "#fff3f0"] * 3):
        for ax in (a1, a2):
            ax.axvspan(y0 - .5, y1 + .5, color=cc, zorder=0)
        a1.text((y0 + y1) / 2, 1.02, name.split(" ", 1)[1], ha="center", va="bottom",
                transform=a1.get_xaxis_transform(), fontsize=7.5, color="0.35")
    for tag, a in M.items():
        a1.plot(a.year, 100 * a.cum_weighted, "-", color=COL[tag], lw=1.8,
                label=f"{tag} ({T.cagr_pct(a):+.2f}%/yr)")
        a2.plot(a.year, a.growth_weighted, "-o", ms=2.5, color=COL[tag], lw=1, label=tag)
    a1.set_ylabel("cum ln TFP x100"); a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=.25)
    a1.set_title("1,000-county sample (IM/Tibet/Qinghai/Xinjiang excluded): converged "
                 "BC92-tnormal SFA vs DEA vs Solow\nwinsorized 1/99 by year, "
                 "output-weighted, five reform stages")
    a2.axhline(0, color="gray", lw=.6); a2.set_ylabel("annual TFP growth %")
    a2.set_xlabel("year"); a2.legend(fontsize=8); a2.grid(alpha=.25)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig_tfp_sub1000_stages.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("saved:", p)
else:
    print(f"\n[skip] {os.path.basename(sub_csv)} not found — run "
          f"`do src/21_sfa_bc92.do sub1000 tnormal` for the subsample comparison")
