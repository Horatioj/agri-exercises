"""
C 步可视化: SWVL 空间格局 (1982/2000/2016) + 35 年时间趋势 + 5 个代表分区时间序列

与 B 步的 render_gdd_review.py 结构对齐, 但聚焦 RZSM (主变量); SM_shallow 在趋势图
里作对照线一并展示.

吸收 B 步 Codex 复审教训: 全部均值用 np.nanmean, 不要用 > 0 过滤
(RZSM = 0 在沙漠是合法物理值, 排除会高估干旱区均值).
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = INTERMEDIATE / "swi_pixel"

# 加载主输出
ds = xr.open_dataset(OUT_DIR / "swvl_growing_season_1982_2016.nc")
rzsm = ds["rzsm_gs"].values        # (35, 391, 651)
sm_sh = ds["sm_shallow_gs"].values
years = ds["year"].values
# 与 B 步统一: lat / lon (Codex P2)
lats = ds["lat"].values
lons = ds["lon"].values
extent = (lons.min(), lons.max(), lats.min(), lats.max())

prov = gpd.read_file(DATA_ROOT / "00 county/省.shp").to_crs(4326)

# 配色: 用 BrBG / Blues 这种干湿色阶, 比 YlOrRd 更直观
# 蓝色 = 湿, 棕黄色 = 干
cmap = plt.get_cmap("YlGnBu")  # 黄→绿→蓝, 浅=干、深=湿
vmin, vmax = 0.0, 0.50

# ============================================================ 图 1: 空间格局 (1982/2000/2016)
print("[1/3] 渲染空间格局图...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)
samples = [1982, 2000, 2016]
for ax, y in zip(axes, samples):
    idx = list(years).index(y)
    a = rzsm[idx]
    show = np.where(np.isfinite(a), a, np.nan)
    im = ax.imshow(show, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest")
    prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
    ax.set_xlim(72, 137); ax.set_ylim(17, 55)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    valid = show[~np.isnan(show)]
    ax.set_title(f"{y} 年生长季 RZSM\n"
                 f"mean={np.nanmean(valid):.3f}  med={np.median(valid):.3f}  max={np.max(valid):.3f}",
                 fontsize=13)

cbar = fig.colorbar(im, ax=axes, ticks=np.arange(0, 0.55, 0.1), shrink=0.7,
                    pad=0.02, fraction=0.025, location="right")
cbar.set_label("RZSM (m³/m³)", fontsize=11)

fig.suptitle("C 步成果: 生长季根区土壤湿度 RZSM 空间格局抽样\n"
             "颜色越深 = 越湿润 (东南湿, 西北干)",
             fontsize=14, fontweight="bold")
out1 = OUT_DIR / "preview_swvl_spatial_3years.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out1.name}")

# ============================================================ 图 2: 35 年全国均值时间序列
print("[2/3] 渲染 35 年时间趋势...")
fig2, ax = plt.subplots(figsize=(13, 6.5))

# 用 isfinite, 不用 > 0 (Codex 复审教训)
rzsm_yearly = np.array([np.nanmean(rzsm[i]) for i in range(len(years))])
sm_sh_yearly = np.array([np.nanmean(sm_sh[i]) for i in range(len(years))])

ax.plot(years, rzsm_yearly, "o-", color="#1f4e79", label="RZSM (0-100cm 根区)", linewidth=2.2)
ax.plot(years, sm_sh_yearly, "s--", color="#5b9bd5", label="SM_shallow (0-28cm 浅层)",
        linewidth=1.8, alpha=0.85)

# 线性趋势
coef_r = np.polyfit(years, rzsm_yearly, 1)
trend_r = np.polyval(coef_r, years)
ax.plot(years, trend_r, ":", color="#1f4e79", linewidth=2,
        label=f"RZSM 线性趋势: {coef_r[0]*10:+.4f} m³/m³ / decade")

coef_s = np.polyfit(years, sm_sh_yearly, 1)
ax.plot(years, np.polyval(coef_s, years), ":", color="#5b9bd5", linewidth=1.5,
        label=f"SM_shallow 趋势: {coef_s[0]*10:+.4f} m³/m³ / decade", alpha=0.7)

ax.set_xlabel("年份"); ax.set_ylabel("生长季土壤湿度 (m³/m³)")
ax.set_title(f"35 年全国土壤湿度时间趋势 (Checkpoint 3 扩展)\n"
             f"RZSM: 1982={rzsm_yearly[0]:.3f} → 2016={rzsm_yearly[-1]:.3f} "
             f"({rzsm_yearly[-1]-rzsm_yearly[0]:+.3f})", fontsize=13)
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)
out2 = OUT_DIR / "preview_swvl_trend_35yr.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
plt.close(fig2)
print(f"  saved: {out2.name}")

# ============================================================ 图 3: 5 个代表分区时间序列
print("[3/3] 渲染 5 个代表分区时间序列...")
fig3, ax = plt.subplots(figsize=(13, 6.5))
with rasterio.open(OUT_DIR / "zone_raster_sm_grid.tif") as src:
    zr = src.read(1)

# 与 B 步相同的 5 个代表分区, 便于跨图对照
representatives = [
    (13, "黄淮平原 (peak=7)",       "#1f77b4"),
    (25, "松嫩平原 (peak=8)",       "#ff7f0e"),
    (37, "华南低平原 (peak=9)",     "#2ca02c"),
    (1,  "藏东南 (peak=8, 高原)",   "#d62728"),
    (20, "南疆绿洲 (peak=8, 干旱)", "#9467bd"),
]
for oid, label, color in representatives:
    m = zr == oid
    if not m.any():
        print(f"  ⚠ OID {oid} 在 SM 网格无像元, 跳过")
        continue
    series = np.array([float(np.nanmean(rzsm[i][m])) for i in range(len(years))])
    ax.plot(years, series, "o-", linewidth=1.8, markersize=4,
            color=color, label=label)

ax.set_xlabel("年份"); ax.set_ylabel("生长季 RZSM (m³/m³)")
ax.set_title("5 个代表性分区 35 年生长季 RZSM 演变\n"
             "(与 B 步同 5 个分区, 便于跨变量对照)", fontsize=13)
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)
out3 = OUT_DIR / "preview_swvl_zone_series.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
plt.close(fig3)
print(f"  saved: {out3.name}")

ds.close()
print("\n[OK] 3 张可视化全部生成")
