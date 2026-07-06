# -*- coding: utf-8 -*-
"""
STEP 2 — County review figures for one-by-one manual checking.

For EVERY province and EVERY input-output variable, juxtapose all of that
province's counties in a small-multiples grid (one panel per county) showing the
log series across 1981-2016:
    grey line  = raw value (pre-cleaning)
    green line = cleaned value (after ÷10,000 fix + spike imputation)
    red  x     = flagged point that was IMPUTED (plotted at its original value)
    orange x   = flagged point that was HELD (endpoint / caliber regime break)
Each panel title is "<countyid> <county name>".

This lets you scan whether each county's variable moves consistently from the
1980s to the 2010s, and spot residual spikes that need manual judgement.

Output (vector SVGs, text rendered as paths so Chinese always shows):
    src/figures/county_review/{SID:02d}_{state}_{var}.svg    (30 provinces x 5 vars)
Run:  python src/02_county_review.py            # resume (skip existing)
      python src/02_county_review.py --overwrite # full rebuild
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
# keep rendering cheap on the big (100+ subplot) provinces; svg.fonttype='path'
# renders text (incl. Chinese) as vector outlines, so CJK never drops in the file.
plt.rcParams.update({"axes.grid": False, "path.simplify": True,
                     "path.simplify_threshold": 1.0, "svg.fonttype": "path"})
EXT = "svg"   # vector output (set to "png" + a dpi in savefig for raster)

# --overwrite clears the folder; default resumes (skips provinces already done)
OVERWRITE = "--overwrite" in sys.argv
REVIEW_DIR = os.path.join(C.FIG_DIR, "county_review")
os.makedirs(REVIEW_DIR, exist_ok=True)
if OVERWRITE:
    locked = 0
    for f in os.listdir(REVIEW_DIR):
        try:
            os.remove(os.path.join(REVIEW_DIR, f))
        except OSError:
            locked += 1            # skip files momentarily locked by a viewer
    if locked:
        print(f"(overwrite: {locked} file(s) were locked and left in place)")

m    = pd.read_csv(C.CLEAN_PANEL)
alog = pd.read_csv(C.ANOMALY_LOG) if os.path.exists(C.ANOMALY_LOG) else pd.DataFrame()
NCOL = 8

# Provincial MEDIAN ln-trend per variable (robust reference). Overlaid on every
# county panel (aligned to the county's own level) so multi-year stretches where a
# county DIVERGES from its province stand out -> mark those in data/manual_impute_list.csv.
PROVMED = {}
for v in C.IO_VARS:
    lnv = np.log(pd.to_numeric(m[v], errors="coerce").where(lambda s: s > 0))
    PROVMED[v] = lnv.groupby([m["SID"], m["year"]]).median()


def ln(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s.where(s > 0))


def render(sid, var):
    dp = m[m.SID == sid].sort_values(["countyid", "year"])
    if dp.empty:
        return None
    name = dp["state"].iloc[0]
    cids = sorted(dp.countyid.unique())
    nrow = int(np.ceil(len(cids) / NCOL))
    figh = max(nrow * 1.95, 3.2)            # min height so small provinces aren't squished
    fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL*2.15, figh), squeeze=False)
    axes = axes.ravel()
    al = alog[(alog.SID == sid) & (alog["var"] == var)] if len(alog) else alog
    Rprov = PROVMED[var].loc[sid] if sid in PROVMED[var].index.get_level_values(0) else None

    for ax, cid in zip(axes, cids):
        g = dp[dp.countyid == cid]
        lnC = ln(g[var])
        # connect-dot: markers on every observation so long gaps (e.g. 440824
        # 龙门县, data ~1990s then 2016) are legible instead of one long line.
        ax.plot(g.year, ln(g[var + "_raw"]), color="0.6", lw=0.7, marker=".", ms=2.5, zorder=1)
        ax.plot(g.year, lnC,                 color="C2", lw=0.8, marker=".", ms=2.5, zorder=2)
        # provincial-median trend, shifted to this county's level: county should
        # track its SHAPE; a stretch peeling away then returning = a candidate to impute.
        if Rprov is not None:
            R = Rprov.reindex(g.year.values)
            mask = lnC.to_numpy() == lnC.to_numpy()  # finite
            ok = np.isfinite(lnC.to_numpy()) & np.isfinite(R.to_numpy())
            if ok.sum() >= 3:
                shift = np.median(lnC.to_numpy()[ok]) - np.median(R.to_numpy()[ok])
                ax.plot(g.year, R.to_numpy() + shift, color="#9467bd", lw=0.8, ls="--", zorder=1.5)
        # imputed years (auto spike-and-revert + manual) -> red dot on the cleaned value;
        # the grey original at that year shows what was replaced (a spike, or a gap).
        imp_mask = (g[var + "_imp"] == True).to_numpy()
        if imp_mask.any():
            ax.plot(g.year.to_numpy()[imp_mask], lnC.to_numpy()[imp_mask],
                    "o", color="red", ms=3.6, zorder=4)
        # held spikes (detected but kept, e.g. caliber break / unanchored) -> orange x
        if len(al):
            held = al[(al.countyid == cid) & (al["action"] == "hold")]
            for _, r in held.iterrows():
                ov = r.get("orig_value")
                if pd.notna(ov) and ov > 0:
                    ax.plot(r.year, np.log(ov), "x", color="orange", ms=5, mew=1.2, zorder=3)
        ax.set_xlim(1981, 2016)               # same x-range for every subplot
        ax.set_title(f"{int(cid)} {g['county_name'].iloc[0]}", fontsize=6, pad=3)
        ax.tick_params(labelsize=5)
    for ax in axes[len(cids):]:
        ax.axis("off")

    nimp  = int(dp[var + "_imp"].sum())
    nheld = int((al["action"] == "hold").sum()) if len(al) else 0
    fig.suptitle(f"{sid} {name}  —  {C.VLAB[var]}  (ln scale, {len(cids)} counties)\n"
                 f"grey = original (raw), green = cleaned, purple dash = provincial median;  "
                 f"red ● = imputed ({nimp}), orange x = held spike ({nheld})",
                 fontsize=11, y=1 - 0.34/figh)
    # manual layout; fixed ~0.95in band for the title + roomy hspace so per-panel
    # titles don't collide, and small provinces (Tianjin/Shanghai/Ningxia) fit fully.
    fig.subplots_adjust(left=0.03, right=0.995, bottom=min(0.06, 0.32/figh),
                        top=1 - 0.95/figh, wspace=0.30, hspace=0.80)
    out = os.path.join(REVIEW_DIR, f"{sid:02d}_{name}_{var}.{EXT}")
    fig.savefig(out)              # vector (SVG) -> infinite resolution
    plt.close(fig)
    return out


sids = sorted(m.SID.dropna().unique().astype(int))
n = skipped = 0
for sid in sids:
    name = m[m.SID == sid]["state"].iloc[0]
    todo = [v for v in C.IO_VARS
            if not os.path.exists(os.path.join(REVIEW_DIR, f"{sid:02d}_{name}_{v}.{EXT}"))]
    if not todo:
        skipped += len(C.IO_VARS)
        print(f"  {sid:>2} {name:<16} {m[m.SID==sid].countyid.nunique():>3} counties -> already done (skip)")
        continue
    for var in todo:
        if render(sid, var) is not None:
            n += 1
    print(f"  {sid:>2} {name:<16} {m[m.SID==sid].countyid.nunique():>3} counties -> {len(todo)} figures")

print(f"\nRendered {n} new {EXT.upper()} figures ({skipped} skipped) "
      f"-> {os.path.relpath(REVIEW_DIR, C.ROOT)}")
