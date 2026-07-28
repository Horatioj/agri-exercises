# -*- coding: utf-8 -*-
"""
Map county-level DEA-Malmquist TFP on the 2010 county shapefile.

  (A) mean annual TFP growth rate, %/yr        (county-level Malmquist)
  (B) volatility: sd of annual TFP growth, pp  (year-to-year variation)

County join is deliberately layered because production codes/names and the 2010
census boundaries are NOT perfectly aligned:
   1. exact county CODE
   2. exact NAME within the same province
   3. FORMER name 曾用名 within the same province   (撤县设市/设区 renames)
   4. NAME STEM (suffix 市/县/区/旗... stripped) within the same province
Unmatched counties are drawn in light grey and reported explicitly.

National aggregate (printed) uses output-share (real GVP) Tornqvist weights.
Inputs : src/clean/dea_malmquist_county.csv, src/clean/county_panel_clean.csv
Shape  : E:/Poverty_Data/2010/县级/T2010年初县级.shp
Output : src/figures/fig_map_tfp.png  + src/clean/map_county_tfp.csv
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import _common as C

SHP = r"E:/Poverty_Data/2010/县级/T2010年初县级.shp"
ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")     # China Albers equal-area
SUF = re.compile(r"(自治县|自治旗|特区|林区|地区|盟|市|县|区|旗)$")

# manual fixes for names that neither code, name, 曾用名 nor stem can resolve
MANUAL = {"满州里市": "满洲里市", "霍县": "霍州市", "布特哈旗": "扎兰屯市",
          "江浦县": "浦口区", "共青城市": "德安县"}

# ---------------------------------------------------------------- county TFP
dea = pd.read_csv(os.path.join(C.CLEAN_DIR, "dea_malmquist_county.csv"))
panel = pd.read_csv(C.CLEAN_PANEL, usecols=["countyid", "county_name", "year", "real_gvp"])

cty = (dea.groupby("countyid")["lnM"]
          .agg(g_mean="mean", g_sd="std", n="count").reset_index())
cty = cty[cty.n >= 10]                                  # need a real time series
cty["growth_pct"] = 100 * (np.exp(cty.g_mean) - 1)      # mean annual TFP growth %
cty["vol_pp"] = 100 * cty.g_sd                          # sd of annual log growth (pp)

# national aggregate, output-share (real GVP) Tornqvist weights -- for reporting
gvp = panel.dropna(subset=["real_gvp"]).query("real_gvp>0")
G = {(int(r.countyid), int(r.year)): float(r.real_gvp) for r in gvp.itertuples()}
rows = []
for t, gg in dea.groupby("year"):
    c = gg.countyid.to_numpy(); d = gg.lnM.to_numpy(float)
    a = np.array([G.get((int(x), int(t)), np.nan) for x in c])
    b = np.array([G.get((int(x), int(t) - 1), np.nan) for x in c])
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(d)
    if ok.sum() == 0: continue
    c, d, a, b = c[ok], d[ok], a[ok], b[ok]
    w = 0.5 * (a / a.sum() + b / b.sum()); w /= w.sum()
    rows.append((int(t), float((w * d).sum())))
nat = pd.DataFrame(rows, columns=["year", "dln"]).sort_values("year")
print(f"National DEA TFP (GVP-weighted): {100*nat.dln.mean():+.2f}%/yr mean, "
      f"cumulative {100*nat.dln.sum():+.1f}% log over {nat.year.min()}-{nat.year.max()}")

# name lookup for the panel counties
nm = (panel.dropna(subset=["county_name"]).groupby("countyid")["county_name"]
      .last().reset_index())
cty = cty.merge(nm, on="countyid", how="left")
cty["prov"] = cty.countyid // 10000

# ---------------------------------------------------------------- shapefile
g = gpd.read_file(SHP).rename(columns={"县级码": "gcode", "地名": "gname",
                                       "曾用名": "gold", "区划码": "gdiv"})
# 不设区的市 (东莞 441900, 中山 442000, 三亚, 嘉峪关) carry 县级码=0 but a valid
# 区划码 -> fall back to it; then drop the remaining code-0 rows (HK/Macao/Taiwan).
g["code"] = pd.to_numeric(g["gcode"], errors="coerce")
gdiv = pd.to_numeric(g["gdiv"], errors="coerce")
g["code"] = g["code"].where(g["code"].fillna(0) > 0, gdiv)
g = g[g["code"].fillna(0) > 0].copy()
g["code"] = g["code"].astype(int); g["prov"] = g["code"] // 10000
g = g.dissolve(by="code", as_index=False, aggfunc="first")     # 1 polygon per code

by_name, by_old, by_stem = {}, {}, {}
for r in g.itertuples():
    by_name.setdefault((r.prov, str(r.gname).strip()), r.code)
    if pd.notna(r.gold):
        by_old.setdefault((r.prov, str(r.gold).strip()), r.code)
    by_stem.setdefault((r.prov, SUF.sub("", str(r.gname).strip())), r.code)

shp_codes = set(g.code)
def resolve(cid, name, prov):
    if cid in shp_codes:                       return cid, "code"
    n = MANUAL.get(str(name).strip(), str(name).strip())
    if (prov, n) in by_name:                   return by_name[(prov, n)], "name"
    if (prov, n) in by_old:                    return by_old[(prov, n)], "former"
    st = SUF.sub("", n)
    if st and (prov, st) in by_stem:           return by_stem[(prov, st)], "stem"
    return np.nan, "unmatched"

res = [resolve(r.countyid, r.county_name, r.prov) for r in cty.itertuples()]
cty["map_code"] = [x[0] for x in res]
cty["how"] = [x[1] for x in res]
print("\njoin to 2010 shapefile:")
print(cty["how"].value_counts().to_string())
miss = cty[cty.how == "unmatched"]
print(f"  unmatched: {len(miss)} of {len(cty)} ({100*len(miss)/len(cty):.1f}%) -> drawn grey")
if len(miss):
    print("   ", list(zip(miss.countyid.head(12), miss.county_name.head(12))))

# duplicate map_code (two production counties -> one 2010 polygon): GVP-weighted mean
mapped = cty.dropna(subset=["map_code"]).copy()
mapped["map_code"] = mapped["map_code"].astype(int)
agg = (mapped.groupby("map_code")
       .agg(growth_pct=("growth_pct", "mean"), vol_pp=("vol_pp", "mean"),
            n_src=("countyid", "nunique")).reset_index())
gm = g.merge(agg, left_on="code", right_on="map_code", how="left").to_crs(ALBERS)
agg.to_csv(os.path.join(C.CLEAN_DIR, "map_county_tfp.csv"), index=False)
print(f"  polygons with data: {gm.growth_pct.notna().sum()} of {len(gm)}")

# ---------------------------------------------------------------- maps
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
prov = gm.dissolve(by="prov")                       # province outlines

specs = [
    ("growth_pct", "(A) Mean annual TFP growth, %/yr",
     "RdYlGn", "TFP growth (%/yr)", True),
    ("vol_pp", "(B) TFP volatility: sd of annual growth (pp)",
     "magma_r", "sd of annual growth (pp)", False),
]
for ax, (col, title, cmap, clab, diverge) in zip(axes, specs):
    v = gm[col].dropna()
    lo, hi = np.percentile(v, [2, 98])
    if diverge:
        m = max(abs(lo), abs(hi))
        norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
        gm.plot(column=col, ax=ax, cmap=cmap, norm=norm, linewidth=0,
                missing_kwds=dict(color="0.88"))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    else:
        gm.plot(column=col, ax=ax, cmap=cmap, vmin=lo, vmax=hi, linewidth=0,
                missing_kwds=dict(color="0.88"))
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=lo, vmax=hi))
    prov.boundary.plot(ax=ax, color="white", linewidth=.35)
    cb = fig.colorbar(sm, ax=ax, shrink=.55, pad=.01)
    cb.set_label(clab, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    ax.annotate(f"median {np.median(v):.2f}   |   grey = no data",
                (0.01, 0.02), xycoords="axes fraction", fontsize=7.5, color="0.35")

fig.suptitle("China county agricultural TFP (DEA sequential-NIRS Malmquist), 1981–2016 — "
             f"national {100*nat.dln.mean():+.2f}%/yr (GVP-weighted)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(C.FIG_DIR, "fig_map_tfp.png")
fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
print("saved:", out)
