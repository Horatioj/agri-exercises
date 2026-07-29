"""
渲染 peak_month.tif 为可读的 PNG.
2026-05-23 v2: 修中文字体 + 清晰图例 + 关键区域标注.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE

# 全局中文字体设置 (macOS)
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

GS = INTERMEDIATE / "growing_season"

with rasterio.open(GS / "peak_month.tif") as src:
    peak = src.read(1)
    bounds = src.bounds
extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
peak_show = np.where(peak == 0, np.nan, peak.astype(float))

# 季节配色: 春蓝, 夏绿, 秋黄橙, 冬紫红 (12 个月)
month_colors = [
    "#3b2a52",  # 1 月 深紫 (冬)
    "#5e3d7a",  # 2
    "#7a6cb3",  # 3 春 (蓝紫)
    "#5b9bd5",  # 4 蓝
    "#7fc97f",  # 5 浅绿
    "#33a02c",  # 6 绿
    "#b8d23a",  # 7 黄绿 ← 主要
    "#ffd92f",  # 8 黄    ← 主要
    "#ff9933",  # 9 橙
    "#e6550d",  # 10 深橙
    "#c41e3a",  # 11 红
    "#7a0d1a",  # 12 暗红
]
cmap = mcolors.ListedColormap(month_colors)
norm = mcolors.BoundaryNorm(np.arange(0.5, 13.5, 1), cmap.N)

# 加载省界
prov = gpd.read_file(DATA_ROOT / "00 county/省.shp").to_crs(4326)

# ============================ 大图: peak_month + 省界 + 关键区域标注
fig, ax = plt.subplots(figsize=(15, 11))
im = ax.imshow(peak_show, extent=extent, cmap=cmap, norm=norm,
               origin="upper", interpolation="nearest")
prov.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.6)

ax.set_xlim(72, 137)
ax.set_ylim(17, 55)
ax.set_xlabel("经度 (°E)", fontsize=12)
ax.set_ylabel("纬度 (°N)", fontsize=12)
ax.set_title("A 步成果：中国 NDVI 物候期 peak month (1982-2016 35 年气候态)\n"
             "颜色 = 一年中最绿的月份（即每个像元自己学出来的生长季中心）",
             fontsize=14, pad=15)

# 关键区域箭头标注 (lon, lat, 文字, 颜色解读)
annotations = [
    (110, 35, "黄淮平原", "→ 7 月 (黄绿色)\n冬麦+夏玉米"),
    (118, 35, "山东丘陵", "→ 8 月 (黄色)"),
    (125, 46, "松嫩平原", "→ 8 月 (黄色)\n东北单季"),
    (110, 22, "华南低平原", "→ 9 月 (橙色)\n晚三熟"),
    (110, 19, "海南", "→ 9-11 月\n热带"),
    (90, 30, "青藏高原", "→ 8 月\n喜凉一熟"),
    (84, 42, "新疆绿洲", "→ 7-8 月"),
]
for lon, lat, name, desc in annotations:
    ax.annotate(f"{name}\n{desc}",
                xy=(lon, lat), xytext=(lon - 12, lat - 4),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))

# ============================ 离散 colorbar (12 个月份逐一标号)
cbar = fig.colorbar(im, ax=ax, ticks=range(1, 13), shrink=0.75,
                    pad=0.02, fraction=0.04)
cbar.set_label("peak month (一年最绿的月份)", fontsize=11)
month_labels = [f"{m} 月" for m in range(1, 13)]
cbar.ax.set_yticklabels(month_labels, fontsize=9)

# ============================ 右下角统计框
stats_text = (
    "全国有效像元: 380,984\n"
    "  7 月 peak:  102,079 (26.8%) ← 黄淮、华北、东北南\n"
    "  8 月 peak:  199,895 (52.5%) ← 内蒙、西北、东北北、青藏\n"
    "  9 月 peak:   27,784 ( 7.3%) ← 云南、华南\n"
    "  10 月 peak:  12,336 ( 3.2%) ← 华南\n"
    "  11 月 peak:  13,442 ( 3.5%) ← 滇南山地\n"
    "  其他月份:    25,448 ( 6.7%) ← 多在热带\n"
    "\n方法: NDVI 35 年半月气候态 + 7 期滑动平均 + argmax\n"
    "对齐 Ortiz-Bobea 2021 NCC 方法 (严格)"
)
ax.text(73, 19, stats_text, fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", alpha=0.9),
        verticalalignment="bottom")

fig.tight_layout()
out_f = GS / "preview_peak_month_v2.png"
fig.savefig(out_f, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_f}")

# ============================ 第二张: 37 区划众数 vs 像元 peak 对比
# 把每个区划画成一个面, 填色用 peak_month_main
import pandas as pd
df = pd.read_csv(GS / "zone37_growing_season.csv")
zones = gpd.read_file(
    DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp"
).to_crs(4326)
zones = zones.merge(df[["OBJECTID_1", "peak_month_main", "NAME"]],
                    on="OBJECTID_1", how="left",
                    suffixes=("_orig", ""))

fig2, axes = plt.subplots(1, 2, figsize=(20, 9))

# 左: 像元层 peak (细颗粒)
ax = axes[0]
im = ax.imshow(peak_show, extent=extent, cmap=cmap, norm=norm,
               origin="upper", interpolation="nearest")
zones.boundary.plot(ax=ax, color="black", linewidth=0.5)
ax.set_xlim(72, 137); ax.set_ylim(17, 55)
ax.set_title("【像元层】每个 0.05° 像元的 peak month\n"
             "区划边界叠加 (黑线)", fontsize=12)
ax.set_xlabel("经度"); ax.set_ylabel("纬度")

# 右: 区划层 peak (粗粒度, A7 输出的众数)
ax = axes[1]
zones.plot(column="peak_month_main", cmap=cmap, norm=norm,
           ax=ax, edgecolor="black", linewidth=0.5, legend=False)
ax.set_xlim(72, 137); ax.set_ylim(17, 55)
ax.set_title("【区划层】37 个区划的 peak month 众数\n"
             "(A7 的 50% 耕地阈值筛选 + 众数)", fontsize=12)
ax.set_xlabel("经度"); ax.set_ylabel("纬度")

# 共享 colorbar
cbar = fig2.colorbar(im, ax=axes, ticks=range(1, 13), shrink=0.7,
                     pad=0.02, fraction=0.025, location="right")
cbar.ax.set_yticklabels([f"{m} 月" for m in range(1, 13)], fontsize=10)
cbar.set_label("peak month", fontsize=11)

fig2.suptitle("像元层 vs 区划层 peak month — 应当大体一致 (区划内颜色应均匀)",
              fontsize=14, fontweight="bold")
out_f2 = GS / "preview_peak_pixel_vs_zone.png"
fig2.savefig(out_f2, dpi=140, bbox_inches="tight")
plt.close(fig2)
print(f"saved: {out_f2}")
