"""
渲染 peak_month.tif - 仅显示耕地像元 (crop_share >= 5%).
非耕地区域 (沙漠、海洋、高山) 不进入 A7 众数, 不该在图上分散注意力.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

GS = INTERMEDIATE / "growing_season"

with rasterio.open(GS / "peak_month.tif") as src:
    peak = src.read(1).astype(float)
    bounds = src.bounds
with rasterio.open(GS / "cropland_share_ndvi_grid.tif") as src:
    crop = src.read(1)
with rasterio.open(GS / "zone_raster_ndvi_grid.tif") as src:
    zr = src.read(1)

extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

# 多个 mask 层
peak_all = np.where(peak == 0, np.nan, peak)
peak_in_zone = np.where(zr > 0, peak_all, np.nan)
peak_cropland = np.where(crop >= 0.05, peak_in_zone, np.nan)
peak_high_crop = np.where(crop >= 0.3, peak_in_zone, np.nan)

# 配色
colors = [
    "#3b2a52","#5e3d7a","#7a6cb3","#5b9bd5","#7fc97f","#33a02c",
    "#b8d23a","#ffd92f","#ff9933","#e6550d","#c41e3a","#7a0d1a",
]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(0.5, 13.5, 1), cmap.N)

prov = gpd.read_file(DATA_ROOT / "00 county/省.shp").to_crs(4326)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

panels = [
    (axes[0,0], peak_all,        "(A) 全部像元 (含沙漠/海洋, 信号噪声混杂)"),
    (axes[0,1], peak_in_zone,    "(B) 仅 37 农业熟制区划内 (28.5% 像元)"),
    (axes[1,0], peak_cropland,   "(C) 耕地占比 ≥ 5% 的像元 (A7 主要看的)"),
    (axes[1,1], peak_high_crop,  "(D) 耕地占比 ≥ 30% 的像元 (核心农田)"),
]
for ax, data, title in panels:
    im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm,
                   origin="upper", interpolation="nearest")
    prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
    ax.set_xlim(72, 137); ax.set_ylim(17, 55)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    n_valid = int(np.sum(~np.isnan(data)))
    ax.text(0.02, 0.98, f"有效像元: {n_valid:,d}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

cbar = fig.colorbar(im, ax=axes, ticks=range(1, 13), shrink=0.6,
                    pad=0.02, fraction=0.025, location="right")
cbar.ax.set_yticklabels([f"{m} 月" for m in range(1, 13)], fontsize=11)
cbar.set_label("peak month (一年最绿月份)", fontsize=12)

fig.suptitle(
    "A 步 peak_month 在不同耕地阈值下的图像\n"
    "→ 从左到右、从上到下: 噪声越来越少, 真实农业信号越来越清晰",
    fontsize=15, fontweight="bold"
)
out_f = GS / "preview_peak_cropland_progression.png"
fig.savefig(out_f, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_f}")

# === 单独一张「核心农田」(crop>=30%) 大图, 不带统计文字干扰
fig2, ax = plt.subplots(figsize=(15, 11))
im = ax.imshow(peak_high_crop, extent=extent, cmap=cmap, norm=norm,
               origin="upper", interpolation="nearest")
prov.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.7)
ax.set_xlim(72, 137); ax.set_ylim(17, 55)
ax.set_xlabel("经度 (°E)", fontsize=12)
ax.set_ylabel("纬度 (°N)", fontsize=12)
ax.set_title(
    "A 步成果 (清洁版): 中国耕地区 peak month\n"
    "只显示耕地占比 ≥ 30% 的像元 — A7 真正基于这些像元做的众数判断",
    fontsize=14, pad=15
)

# 关键区域标注
annotations = [
    (110, 35, "黄淮平原 → 7 月\n(冬麦+夏玉米)", (95, 30)),
    (118, 36, "山东平原 → 8 月", (130, 40)),
    (125, 46, "松嫩平原 → 8 月\n(东北一熟)", (130, 50)),
    (113, 23, "华南低平原 → 9 月\n(晚三熟)", (95, 21)),
    (110, 19, "海南 → 9-11 月\n(热带)", (115, 17.5)),
    (87, 44, "北疆灌溉 → 7 月", (75, 50)),
    (80, 40, "南疆绿洲 → 8 月", (73, 36)),
]
for lon, lat, label, textpos in annotations:
    ax.annotate(label, xy=(lon, lat), xytext=textpos,
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fffacc", ec="black", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

cbar = fig2.colorbar(im, ax=ax, ticks=range(1, 13), shrink=0.75,
                     pad=0.02, fraction=0.04)
cbar.set_label("peak month", fontsize=11)
cbar.ax.set_yticklabels([f"{m} 月" for m in range(1, 13)], fontsize=10)

fig2.tight_layout()
out_f2 = GS / "preview_peak_cropland_only.png"
fig2.savefig(out_f2, dpi=140, bbox_inches="tight")
plt.close(fig2)
print(f"saved: {out_f2}")
