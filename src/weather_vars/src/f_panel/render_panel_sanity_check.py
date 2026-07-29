"""
F 步收官可视化 - 方案 §11 Sanity Check
图 1: 县级 GDD 空间格局 (1982/2000/2016 对比)
图 2: 县级 SM 空间格局 (RZSM + SM_shallow, 2010 年)
图 3: 5 个代表县 35 年时间序列 (GDD + RZSM 双 panel)
图 4: 跨区划县 + NaN 县地理分布
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, OUTPUT

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================ 加载
panel = pd.read_parquet(OUTPUT / "climate_panel_1982_2016.parquet")
xian = gpd.read_file(DATA_ROOT / "00 county" / "县.shp").to_crs(4326)
xian = xian[~xian.PAC.isin([156408, 710024])]
prov = gpd.read_file(DATA_ROOT / "00 county" / "省.shp").to_crs(4326)

print(f"面板: {len(panel):,d} 行")
print(f"县几何: {len(xian)} 个")

# ============================================================ 图 1: 县级 GDD 空间格局 (3 年)
print("\n[1/4] 渲染县级 GDD 空间格局 (1982/2000/2016)...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)

for ax, y in zip(axes, [1982, 2000, 2016]):
    df_y = panel[panel.year == y][["PAC", "gdd_growing_season"]]
    gdf = xian.merge(df_y, on="PAC", how="left")
    gdf.plot(column="gdd_growing_season", cmap="YlOrRd",
             vmin=0, vmax=3000,
             ax=ax, edgecolor="none", missing_kwds={"color": "lightgrey"})
    prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
    ax.set_xlim(72, 137); ax.set_ylim(17, 55)
    ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    valid = df_y.gdd_growing_season.dropna()
    ax.set_title(f"{y} 年县级生长季 GDD\n"
                 f"mean={valid.mean():.0f}, med={valid.median():.0f}, max={valid.max():.0f}",
                 fontsize=13)

# 共享 colorbar
sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=3000), cmap="YlOrRd")
cbar = fig.colorbar(sm, ax=axes, ticks=range(0, 3001, 500), shrink=0.75, fraction=0.025,
                    pad=0.02, location="right")
cbar.set_label("生长季 GDD (day·°C)", fontsize=11)

fig.suptitle("县级 GDD 空间格局 — 方案 §11.2 Checkpoint\n"
             "南方深红 (≥3000), 黄淮华北橙 (2000-2500), 东北西北黄 (1500-2000), 青藏浅 (<500), 灰=无耕地县",
             fontsize=14, fontweight="bold")
out1 = OUTPUT / "panel_gdd_county_3years.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out1.name}")

# ============================================================ 图 2: 县级 SM 空间格局 (2010)
print("[2/4] 渲染县级 SM 空间格局 (2010)...")
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

df_2010 = panel[panel.year == 2010]
for ax, var, label in zip(axes2,
                          ["rzsm_gs", "sm_shallow_gs"],
                          ["RZSM (0-100 cm 根区)", "SM_shallow (0-28 cm 浅层)"]):
    sub = df_2010[["PAC", var]]
    gdf = xian.merge(sub, on="PAC", how="left")
    gdf.plot(column=var, cmap="YlGnBu",
             vmin=0.0, vmax=0.50,
             ax=ax, edgecolor="none", missing_kwds={"color": "lightgrey"})
    prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
    ax.set_xlim(72, 137); ax.set_ylim(17, 55)
    ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    valid = sub[var].dropna()
    ax.set_title(f"2010 县级 {label}\n"
                 f"mean={valid.mean():.3f}, med={valid.median():.3f}, max={valid.max():.3f}",
                 fontsize=13)

sm2 = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=0.50), cmap="YlGnBu")
cbar2 = fig2.colorbar(sm2, ax=axes2, ticks=np.arange(0, 0.55, 0.1), shrink=0.75,
                     fraction=0.025, pad=0.02, location="right")
cbar2.set_label("土壤湿度 (m³/m³)", fontsize=11)

fig2.suptitle("县级土壤湿度空间格局 — 方案 §11.2 Checkpoint\n"
              "东南/东北深蓝 (>0.30), 华北浅蓝 (0.15-0.25), 西北塔克拉玛干/柴达木接近 0, 灰=无耕地县",
              fontsize=14, fontweight="bold")
out2 = OUTPUT / "panel_sm_county_2010.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
plt.close(fig2)
print(f"  saved: {out2.name}")

# ============================================================ 图 3: 5 个代表县 35 年时间序列
print("[3/4] 渲染 5 个代表县 35 年时间序列...")
fig3, (ax_g, ax_s) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

# 代表县 (跨南北 + 跨气候带)
representative_counties = [
    (410105, "郑州金水 (黄淮平原)", "#d62728"),
    (230110, "哈尔滨香坊 (东北)",   "#ff7f0e"),
    (440106, "广州天河 (华南)",     "#2ca02c"),
    (510107, "成都锦江 (西南)",     "#9467bd"),
    (650102, "乌鲁木齐天山 (西北)", "#1f77b4"),
]

for pac, label, color in representative_counties:
    df_c = panel[panel.PAC == pac].sort_values("year")
    if len(df_c) == 0:
        print(f"  ⚠ PAC {pac} ({label}) 不在面板中")
        continue
    ax_g.plot(df_c.year, df_c.gdd_growing_season, "o-",
              color=color, label=label, linewidth=1.8, markersize=4)
    ax_s.plot(df_c.year, df_c.rzsm_gs, "o-",
              color=color, label=label, linewidth=1.8, markersize=4)

ax_g.set_ylabel("生长季 GDD (day·°C)", fontsize=12)
ax_g.set_title("5 个代表县 35 年生长季 GDD (方案 §11.3 Checkpoint - 应见缓慢上升)",
               fontsize=13)
ax_g.legend(loc="best", fontsize=10)
ax_g.grid(alpha=0.3)

ax_s.set_xlabel("年份", fontsize=12)
ax_s.set_ylabel("生长季 RZSM (m³/m³)", fontsize=12)
ax_s.set_title("5 个代表县 35 年生长季 RZSM (方案 §11.3 - 已知干旱年应有下降)",
               fontsize=13)
ax_s.legend(loc="best", fontsize=10)
ax_s.grid(alpha=0.3)

out3 = OUTPUT / "panel_5counties_timeseries.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
plt.close(fig3)
print(f"  saved: {out3.name}")

# ============================================================ 图 4: 跨区划县 + NaN 县地理分布
print("[4/4] 渲染 跨区划县 + NaN 县地理分布...")
fig4, axes4 = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)

# 用 zone_info 静态信息
zone_info = panel.drop_duplicates("PAC")[["PAC", "is_cross_zone", "zone37_main"]]
zone_info["status"] = "valid"
zone_info.loc[zone_info.is_cross_zone == True, "status"] = "cross_zone"
zone_info.loc[zone_info.zone37_main.isna(), "status"] = "no_zone"

gdf4 = xian.merge(zone_info, on="PAC", how="left")
# 没在 zone_info 里的 = 没有耕地+区划像元
gdf4["status"] = gdf4["status"].fillna("no_zone")

# 左: 跨区划 + 无 zone 县
ax = axes4[0]
gdf4[gdf4.status == "valid"].plot(ax=ax, color="#d4e7d4", edgecolor="white", linewidth=0.2,
                                    label=f"主导>90% (n={int((gdf4.status=='valid').sum())})")
gdf4[gdf4.status == "cross_zone"].plot(ax=ax, color="#d62728", edgecolor="white", linewidth=0.3,
                                        label=f"跨区划 (n={int((gdf4.status=='cross_zone').sum())})")
gdf4[gdf4.status == "no_zone"].plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.2,
                                     label=f"无耕地+区划重叠 (n={int((gdf4.status=='no_zone').sum())})")
prov.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.7)
ax.set_xlim(72, 137); ax.set_ylim(17, 55)
ax.set_xlabel("经度"); ax.set_ylabel("纬度")
ax.set_title(f"县-区划状态 (方案 §11.4 - 跨区划应 <15%)\n"
             f"红 = 跨区划 ({int((gdf4.status=='cross_zone').sum())}/{len(gdf4)} = "
             f"{100*(gdf4.status=='cross_zone').mean():.1f}%)",
             fontsize=12)
ax.legend(loc="lower left", fontsize=10)

# 右: 县-级 valid_climate_sample 覆盖 (Codex P1.1 反馈后的口径)
ax = axes4[1]
# 用 v2 的 valid_climate_sample 列 (gdd+rzsm+sm 都非 NaN)
df_2010_v = panel[panel.year == 2010][["PAC", "valid_climate_sample"]]
gdf5 = xian.merge(df_2010_v, on="PAC", how="left")
gdf5["valid_climate_sample"] = gdf5["valid_climate_sample"].fillna(False).astype(bool)

n_valid_2010 = int(gdf5["valid_climate_sample"].sum())
n_invalid_2010 = int((~gdf5["valid_climate_sample"]).sum())

gdf5[gdf5["valid_climate_sample"]].plot(
    ax=ax, color="#5b9bd5", edgecolor="white", linewidth=0.2,
    label=f"有效气候样本 (n={n_valid_2010})"
)
gdf5[~gdf5["valid_climate_sample"]].plot(
    ax=ax, color="#c41e3a", edgecolor="white", linewidth=0.2,
    label=f"无效 (n={n_invalid_2010})"
)
prov.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.7)
ax.set_xlim(72, 137); ax.set_ylim(17, 55)
ax.set_xlabel("经度"); ax.set_ylabel("纬度")
ax.set_title(f"2010 valid_climate_sample 覆盖 (方案 §11.1 + Codex P1.1 修正)\n"
             f"红 = 不在 37 农业熟制区划覆盖范围内 ({n_invalid_2010}/{len(gdf5)} = "
             f"{100*n_invalid_2010/len(gdf5):.1f}%) - 含内蒙古东北/海南/台湾边缘+部分城区",
             fontsize=11)
ax.legend(loc="lower left", fontsize=10)

fig4.suptitle("县-面板质量空间分布 - 方案 §11 Sanity Check",
              fontsize=14, fontweight="bold")
out4 = OUTPUT / "panel_quality_geography.png"
fig4.savefig(out4, dpi=130, bbox_inches="tight")
plt.close(fig4)
print(f"  saved: {out4.name}")

print("\n[OK] 4 张可视化全部生成")
