"""
G 步可视化: 灌溉空间格局 (1982/2000/2016) + 35 年全国 sum 时间趋势 + 5 个代表分区时间序列

与 C 步 render_swvl_review.py 结构对齐, 但因为灌溉是体积 flux:
  - 空间图: 颜色 ~ 单格 sum (10^8 m³ / 格 / 生长季)
  - 时间序列: 全国 (或分区) 内 sum, 不是 mean
  - 5 个代表分区与 B/C 步同 OID (13/25/37/1/20), 便于跨变量对照

吸收 B/C 步 Codex 复审教训: NaN 像元保留 (海洋/沙漠/水体), 不强行填 0.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = INTERMEDIATE / "irr_pixel"

# 加载主输出
ds = xr.open_dataset(OUT_DIR / "irr_growing_season_1982_2016.nc")
irr = ds["irr_gs"].values            # (35, 355, 615), 单位 10^8 m³ / 格 / 生长季
years = ds["year"].values
lats = ds["lat"].values              # N→S 递减 (与 B/C 一致)
lons = ds["lon"].values
extent = (lons.min(), lons.max(), lats.min(), lats.max())

prov = gpd.read_file(DATA_ROOT / "00 county/省.shp").to_crs(4326)

# 配色: 蓝色阶, 浅=低用水, 深=高用水. 灌溉量级跨度大 (P50≈0.0007, P99≈0.30),
# 用 PowerNorm(gamma=0.4) 压缩动态范围, 让低值可见.
cmap = plt.get_cmap("YlGnBu")
vmin, vmax = 0.0, 0.30                       # vmax 约 P99
norm = mcolors.PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)

# ============================================================ 图 1: 空间格局 (1982/2000/2016)
print("[1/3] 渲染空间格局图...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)
samples = [1982, 2000, 2016]
for ax, y in zip(axes, samples):
    idx = list(years).index(y)
    a = irr[idx]
    show = np.where(np.isfinite(a), a, np.nan)
    im = ax.imshow(show, extent=extent, cmap=cmap, norm=norm,
                   origin="upper", interpolation="nearest")
    prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
    ax.set_xlim(72, 137); ax.set_ylim(17, 55)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    valid = show[~np.isnan(show)]
    total = float(np.nansum(valid))
    ax.set_title(
        f"{y} 年生长季灌溉用水\n"
        f"全国 sum = {total:.0f} × 10^8 m³  "
        f"(P50={np.percentile(valid,50):.3f}  P95={np.percentile(valid,95):.3f})",
        fontsize=13,
    )

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, fraction=0.025, location="right")
cbar.set_label("irr_gs (10^8 m³ / 格 / 生长季, PowerNorm γ=0.4)", fontsize=11)

fig.suptitle(
    "G 步成果: 生长季灌溉用水空间格局抽样\n"
    "颜色越深 = 用水越多 (华北/新疆绿洲/长江中下游 高, 高原/林区/牧区 低)",
    fontsize=14, fontweight="bold",
)
out1 = OUT_DIR / "preview_irr_spatial_3years.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  saved: {out1.name}")

# ============================================================ 图 2: 35 年全国 sum 时间趋势
print("[2/3] 渲染 35 年全国 sum 时间趋势...")
fig2, ax = plt.subplots(figsize=(13, 6.5))

# 加载灌溉自己的 zone_raster, 只对 zone 内 sum (排除海洋/沙漠对总量的污染)
with rasterio.open(OUT_DIR / "zone_raster_irr_grid.tif") as src:
    zr = src.read(1)
zmask_any = (zr > 0)

# zone 内全国 sum (每年一个标量), 单位 10^8 m³
irr_yearly_sum = np.array([float(np.nansum(irr[i][zmask_any])) for i in range(len(years))])

ax.plot(years, irr_yearly_sum, "o-", color="#1f4e79",
        label="zone 内 + 生长季 sum", linewidth=2.2, markersize=5)

# 线性趋势
coef = np.polyfit(years, irr_yearly_sum, 1)
trend = np.polyval(coef, years)
ax.plot(years, trend, ":", color="#1f4e79", linewidth=2,
        label=f"线性趋势: {coef[0]*10:+.1f} × 10^8 m³ / decade")

ax.set_xlabel("年份")
ax.set_ylabel("生长季灌溉用水 (10^8 m³, zone 内 sum)")
ax.set_title(
    f"35 年全国生长季灌溉用水时间趋势 (Checkpoint 扩展)\n"
    f"1982={irr_yearly_sum[0]:.0f} → 2016={irr_yearly_sum[-1]:.0f}  "
    f"(Δ={irr_yearly_sum[-1]-irr_yearly_sum[0]:+.0f}, "
    f"{100*(irr_yearly_sum[-1]/irr_yearly_sum[0]-1):+.1f}%)",
    fontsize=13,
)
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)
out2 = OUT_DIR / "preview_irr_trend_35yr.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
plt.close(fig2)
print(f"  saved: {out2.name}")

# ============================================================ 图 3: 5 个代表分区时间序列
print("[3/3] 渲染 5 个代表分区 sum 时间序列...")
fig3, ax = plt.subplots(figsize=(13, 6.5))

# 与 B/C 步相同的 5 个代表分区, 便于跨图对照
representatives = [
    (13, "黄淮平原 (peak=7)",       "#1f77b4"),
    (25, "松嫩平原 (peak=8)",       "#ff7f0e"),
    (37, "华南低平原 (peak=9)",     "#2ca02c"),
    (1,  "藏东南 (peak=8, 高原)",   "#d62728"),
    (20, "南疆绿洲 (peak=8, 干旱)", "#9467bd"),
]
for oid, label, color in representatives:
    m = (zr == oid)
    if not m.any():
        print(f"  ⚠ OID {oid} 在灌溉网格无像元, 跳过")
        continue
    series = np.array([float(np.nansum(irr[i][m])) for i in range(len(years))])
    ax.plot(years, series, "o-", linewidth=1.8, markersize=4,
            color=color, label=label)

ax.set_xlabel("年份")
ax.set_ylabel("生长季灌溉用水 (10^8 m³, 分区内 sum)")
ax.set_title(
    "5 个代表性分区 35 年生长季灌溉用水演变\n"
    "(与 B/C 步同 5 个分区, 便于跨变量对照)",
    fontsize=13,
)
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)
out3 = OUT_DIR / "preview_irr_zone_series.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
plt.close(fig3)
print(f"  saved: {out3.name}")

ds.close()
print("\n[OK] 3 张可视化全部生成")
