# -*- coding: utf-8 -*-
"""
STEP 3 — Province statistical-caliber / definition-break diagnostics.

A "caliber break" is a year where MANY counties in a province jump simultaneously
(a definitional/statistical change, not a real production shock). These are held
(not pointwise-imputed) in step 1; this script visualises them so you can decide
how to segment or trim each province.

Per province -> src/figures/provinces/{SID:02d}_{state}.png :
  - top heatmap: variable x year, colour = share of counties with |Δln|>0.3 that
    year (a bright vertical stripe = a candidate caliber-change year)
  - 5 line panels (one per I-O variable): every county overlaid (ln, thin grey),
    province median bold; strong caliber years marked with dashed lines.
Plus src/figures/national_caliber_overview.png (province x year, real_gvp jumps).
Run:  python src/03_caliber_diagnostics.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

C.set_cjk_font(plt)
np.seterr(divide="ignore", invalid="ignore")
PROV_DIR = os.path.join(C.FIG_DIR, "provinces")
os.makedirs(PROV_DIR, exist_ok=True)

m = pd.read_csv(C.CLEAN_PANEL)
cal = pd.read_csv(C.CALIBER_CSV) if os.path.exists(C.CALIBER_CSV) else pd.DataFrame()
THR = 0.30
years = np.arange(int(m.year.min())+1, int(m.year.max())+1)


def jump_share(df, col):
    d = df[["countyid", "year", col]].copy()
    d = d[d[col] > 0]
    d["ldiff"] = d.groupby("countyid")[col].transform(lambda s: np.log(s).diff())
    d["dyr"]   = d.groupby("countyid")["year"].diff()
    d.loc[d["dyr"] != 1, "ldiff"] = np.nan
    g = d.dropna(subset=["ldiff"]).groupby("year")["ldiff"]
    return g.apply(lambda s: (s.abs() > THR).mean()).reindex(years)


prov = m[["SID", "state"]].drop_duplicates().sort_values("SID")
for _, pr in prov.iterrows():
    sid, name = int(pr.SID), pr.state
    dp = m[m.SID == sid]
    ncty = dp.countyid.nunique()
    shares = {v: jump_share(dp, v) for v in C.IO_VARS}
    H = np.vstack([shares[v].values for v in C.IO_VARS])
    strong = sorted(cal[(cal.SID == sid) & (cal.strong)]["year"].tolist()) if len(cal) else []

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1], hspace=0.42, wspace=0.22)
    axh = fig.add_subplot(gs[0, :])
    im = axh.imshow(H, aspect="auto", cmap="inferno", vmin=0, vmax=0.6,
                    extent=[years[0]-0.5, years[-1]+0.5, len(C.IO_VARS)-0.5, -0.5])
    axh.set_yticks(range(len(C.IO_VARS))); axh.set_yticklabels([C.VLAB[v] for v in C.IO_VARS])
    axh.set_xticks(years[::2]); axh.set_xticklabels(years[::2], rotation=90, fontsize=7)
    axh.set_title(f"{sid} {name}  ({ncty} counties)  —  share of counties with |Δln|>{THR} per year "
                  f"(bright vertical stripe = candidate caliber-change year)", fontsize=11)
    fig.colorbar(im, ax=axh, fraction=0.025, pad=0.01)

    for k, v in enumerate(C.IO_VARS):
        ax = fig.add_subplot(gs[1 + k//3, k % 3])
        for cid, g in dp.groupby("countyid"):
            g = g[g[v] > 0]
            if len(g) > 1:
                ax.plot(g["year"], np.log(g[v]), color="0.6", lw=0.4, alpha=0.4)
        med = dp[dp[v] > 0].groupby("year")[v].median()
        ax.plot(med.index, np.log(med), color="C3", lw=2, label="median")
        for y in strong:
            ax.axvline(y, color="C0", ls="--", lw=1)
        ax.set_title(C.VLAB[v] + " (ln)", fontsize=10); ax.tick_params(labelsize=7)
    note = "strong caliber yrs (>=2 vars): " + (", ".join(map(str, strong)) if strong else "none")
    fig.suptitle(note, y=0.065, fontsize=10, color="C0")
    fig.savefig(os.path.join(PROV_DIR, f"{sid:02d}_{name}.png"), dpi=92, bbox_inches="tight")
    plt.close(fig)
    print(f"  {sid:>2} {name:<16} {ncty:>3} counties | strong caliber yrs: {strong}")

# national overview (real_gvp)
Nat = np.vstack([jump_share(m[m.SID == int(pr.SID)], "real_gvp").values for _, pr in prov.iterrows()])
fig, ax = plt.subplots(figsize=(15, 9))
im = ax.imshow(Nat, aspect="auto", cmap="inferno", vmin=0, vmax=0.6,
               extent=[years[0]-0.5, years[-1]+0.5, len(prov)-0.5, -0.5])
ax.set_yticks(range(len(prov))); ax.set_yticklabels([f"{int(s)} {n}" for s, n in zip(prov.SID, prov.state)], fontsize=8)
ax.set_xticks(years[::1]); ax.set_xticklabels(years, rotation=90, fontsize=7)
ax.set_title(f"NATIONAL: share of counties with |Δln real_GVP|>{THR} (province x year)\n"
             "vertical stripe = nationwide caliber change; single cell = province-specific", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.02)
fig.savefig(os.path.join(C.FIG_DIR, "national_caliber_overview.png"), dpi=110, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved {len(prov)} province figures -> {os.path.relpath(PROV_DIR, C.ROOT)}/")
print(f"Saved national overview -> {os.path.relpath(os.path.join(C.FIG_DIR,'national_caliber_overview.png'), C.ROOT)}")
if len(cal):
    print("\nMost common strong caliber years across provinces:")
    print(cal[cal.strong].year.value_counts().sort_index().to_string())
