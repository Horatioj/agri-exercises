"""
D 步可视化:
图 1: 4 个抽样年份耕地空间分布 (温度网格) → 看南北分布合理性
图 2: 同一切片在温度网格 vs SWVL 网格的对比 → 验证半像元错位不影响内容
图 3: 9 个 LUCC 切片耕地总量演变 + 35 年映射
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CW_DIR = INTERMEDIATE / "cropland_weight"
prov = gpd.read_file(DATA_ROOT / "00 county/省.shp").to_crs(4326)


def load_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        bounds = src.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return arr, extent


# 配色: 黄→绿, 0 透明
cmap = plt.get_cmap("YlGn").copy()
cmap.set_under("white", alpha=0)
vmin, vmax = 0.5, 110  # vmin=0.5 让 0 像元不显示

# ============================================================ 图 1: 4 年抽样空间分布
print("[1/3] 渲染 4 年空间分布...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
samples = [1980, 1995, 2005, 2015]
for ax, year in zip(axes.flat, samples):
    arr, extent = load_tif(CW_DIR / f"cropland_weight_{year}_tempgrid.tif")
    im = ax.imshow(arr, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest")
    prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.55)
    ax.set_xlim(72, 137); ax.set_ylim(17, 55)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    total_km2 = int(arr[arr > 0].sum())
    n_pixels = int((arr > 0).sum())
    max_val = float(arr.max())
    ax.set_title(f"{year} 年 LUCC 耕地分布 (温度 0.1° 网格)\n"
                 f"全国 {total_km2:,d} km² | {n_pixels:,d} 像元含耕地 | 最大 {max_val:.0f} km²/像元",
                 fontsize=12)

cbar = fig.colorbar(im, ax=axes, ticks=range(0, 121, 20), shrink=0.7,
                    pad=0.02, fraction=0.025, location="right")
cbar.set_label("耕地 km² / 0.1° 像元", fontsize=11)

fig.suptitle("D 步成果: 中国耕地空间分布 (Resampling.sum 重采样)\n"
             "黄淮华北 / 东北 / 长江中下游 / 四川盆地 应是高值区",
             fontsize=15, fontweight="bold")
out1 = CW_DIR / "preview_cropland_4years.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out1.name}")

# ============================================================ 图 2: 温度网格 vs SWVL 网格 (2000 年)
print("[2/3] 渲染 网格对比 (2000)...")
fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)

arr_t, ext_t = load_tif(CW_DIR / "cropland_weight_2000_tempgrid.tif")
arr_s, ext_s = load_tif(CW_DIR / "cropland_weight_2000_smgrid.tif")

# 左: 温度网格
ax = axes2[0]
ax.imshow(arr_t, extent=ext_t, cmap=cmap, vmin=vmin, vmax=vmax,
          origin="upper", interpolation="nearest")
prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
ax.set_xlim(72, 137); ax.set_ylim(17, 55)
ax.set_title(f"温度 0.1° 网格 (B 步)\n400 × 700 像元, 像元中心 70.05/54.95 起点\n"
             f"总和 {int(arr_t.sum()):,d} km²", fontsize=12)
ax.set_xlabel("经度"); ax.set_ylabel("纬度")

# 中: SWVL 网格
ax = axes2[1]
ax.imshow(arr_s, extent=ext_s, cmap=cmap, vmin=vmin, vmax=vmax,
          origin="upper", interpolation="nearest")
prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
ax.set_xlim(72, 137); ax.set_ylim(17, 55)
ax.set_title(f"SWVL 0.1° 网格 (C 步)\n391 × 651 像元, 像元中心 72.00/55.00 起点\n"
             f"总和 {int(arr_s.sum()):,d} km²", fontsize=12)
ax.set_xlabel("经度"); ax.set_ylabel("纬度")

# 右: 局部放大 (黄淮 110-120E, 32-38N) 直观看半像元错位
ax = axes2[2]
ax.imshow(arr_t, extent=ext_t, cmap=cmap, vmin=vmin, vmax=vmax,
          origin="upper", interpolation="nearest", alpha=0.7)
ax.imshow(arr_s, extent=ext_s, cmap="Reds", vmin=vmin, vmax=vmax,
          origin="upper", interpolation="nearest", alpha=0.4)
prov.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.7)
ax.set_xlim(112, 118); ax.set_ylim(33, 38)
ax.set_title("局部放大: 黄淮平原 (112-118°E, 33-38°N)\n"
             "绿 = 温度网格, 红 = SWVL 网格 (半像元错位可视化)", fontsize=11)
ax.set_xlabel("经度"); ax.set_ylabel("纬度")

fig2.suptitle("2000 年 LUCC 切片: 温度网格 vs SWVL 网格对比\n"
              "两套网格各 0.1°, 内容几乎一致, 仅像元中心错位 0.05° (半个像元)",
              fontsize=14, fontweight="bold")
out2 = CW_DIR / "preview_cropland_grid_comparison.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
plt.close(fig2)
print(f"  saved: {out2.name}")

# ============================================================ 图 3: 9 切片耕地总量演变 + 35 年映射
print("[3/3] 渲染 35 年映射 + 耕地总量演变...")
fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9),
                                       gridspec_kw={"height_ratios": [1.3, 1]},
                                       constrained_layout=True)

# 上: 9 切片耕地总量柱状图
slices = [1980, 1990, 1995, 2000, 2005, 2008, 2010, 2013, 2015]
totals = []
for y in slices:
    with rasterio.open(CW_DIR / f"cropland_weight_{y}_tempgrid.tif") as src:
        totals.append(float(src.read(1).sum()))

bars = ax_top.bar([str(y) for y in slices], totals, color="#5b9b40", edgecolor="black")
# 标注峰值
peak_idx = int(np.argmax(totals))
bars[peak_idx].set_color("#c41e3a")
for i, (yr, v) in enumerate(zip(slices, totals)):
    pct_change = 100 * (v - totals[0]) / totals[0]
    ax_top.text(i, v + 5000, f"{v/1e4:.1f} 万 km²\n({pct_change:+.1f}% vs 1980)",
                ha="center", fontsize=9, fontweight="bold" if i == peak_idx else "normal")
ax_top.set_ylim(0, max(totals) * 1.12)
ax_top.set_ylabel("全国耕地总量 (km²)", fontsize=12)
ax_top.set_title("9 个 LUCC 切片耕地总量 (红色 = 2000 年峰值, 与数据说明 §4.5 一致)",
                 fontsize=13)
ax_top.grid(alpha=0.3, axis="y")

# 下: 35 年→切片映射可视化
df_map = pd.read_csv(CW_DIR / "year_to_lucc_mapping.csv")
unique_slices = sorted(df_map.lucc_slice.unique())
color_map = {s: plt.cm.tab10(i % 10) for i, s in enumerate(unique_slices)}
for _, row in df_map.iterrows():
    y, slc = int(row.year), int(row.lucc_slice)
    ax_bot.barh(0, 1, left=y, color=color_map[slc], edgecolor="white", linewidth=0.5)
    if y == df_map[df_map.lucc_slice == slc].year.iloc[0]:  # 切片第一年标注
        ax_bot.text(y - 0.3, 0, str(slc), va="center", ha="right",
                    fontsize=10, fontweight="bold", color=color_map[slc])

ax_bot.set_xlim(1981, 2017)
ax_bot.set_ylim(-0.5, 0.5)
ax_bot.set_yticks([])
ax_bot.set_xlabel("研究年份 (1982-2016)", fontsize=12)
ax_bot.set_title("35 年 → 切片映射 (左侧标签为切片年, 颜色代表使用同一切片的年份段)",
                 fontsize=12)
ax_bot.grid(alpha=0.3, axis="x")
ax_bot.set_xticks(range(1982, 2017, 2))
ax_bot.tick_params(axis="x", rotation=45)

out3 = CW_DIR / "preview_cropland_trend_mapping.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
plt.close(fig3)
print(f"  saved: {out3.name}")

print("\n[OK] 3 张可视化全部生成")
