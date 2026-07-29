"""
B 步可视化: GDD 空间格局 (1982/2000/2016 抽样) + 35 年时间趋势
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

GDD_DIR = INTERMEDIATE / "gdd_pixel"

# 加载 NetCDF (35 年)
ds = xr.open_dataset(GDD_DIR / "gdd_growing_season_1982_2016.nc")
arr = ds["gdd_growing_season"].values  # (35, H, W)
years = ds["year"].values
lons = ds["lon"].values
lats = ds["lat"].values
extent = (lons.min(), lons.max(), lats.min(), lats.max())

prov = gpd.read_file(DATA_ROOT / "00 county/省.shp").to_crs(4326)
zones = gpd.read_file(DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp").to_crs(4326)

# === 图 1: 1982/2000/2016 三年抽样空间格局
fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)
cmap = plt.get_cmap("YlOrRd")
vmin, vmax = 0, 3300

samples = [1982, 2000, 2016]
for ax, y in zip(axes, samples):
    idx = list(years).index(y)
    a = arr[idx]
    show = np.where(a >= 0, a, np.nan)
    show = np.where(show == -9999, np.nan, show)
    im = ax.imshow(show, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest")
    prov.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)
    ax.set_xlim(72, 137); ax.set_ylim(17, 55)
    ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    valid = show[~np.isnan(show)]
    ax.set_title(f"{y} 年生长季 GDD\n"
                 f"mean={valid.mean():.0f}  med={np.median(valid):.0f}  max={valid.max():.0f}",
                 fontsize=13)

cbar = fig.colorbar(im, ax=axes, ticks=range(0, 3501, 500), shrink=0.7,
                    pad=0.02, fraction=0.025, location="right")
cbar.set_label("生长季 GDD (日·°C)", fontsize=11)

fig.suptitle("B 步成果: 生长季 GDD 空间格局抽样\n"
             "颜色越深 = 积温越高 (南方高, 北方/高原低)", fontsize=14, fontweight="bold")
out1 = GDD_DIR / "preview_gdd_spatial_3years.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out1}")

# === 图 2: 35 年全国均值时间序列 (Checkpoint 3)
# 每年 zone 内有效像元均值
fig2, ax = plt.subplots(figsize=(13, 6))
# 用 isfinite 不用 > 0: 高原 GDD=0 是合法物理值, 排除会高估冷区均值 (Codex 复审)
yearly_mean = np.array([np.nanmean(arr[i]) for i in range(len(years))])
yearly_med = np.array([np.nanmedian(arr[i]) for i in range(len(years))])
ax.plot(years, yearly_mean, "o-", color="#c41e3a", label="年均 (zone 内像元)", linewidth=2)
ax.plot(years, yearly_med, "s--", color="#ff9933", label="年中位数", linewidth=1.5, alpha=0.7)

# 线性趋势
from numpy.polynomial import polynomial as P
coef = np.polyfit(years, yearly_mean, 1)
trend = np.polyval(coef, years)
ax.plot(years, trend, ":", color="black", linewidth=2,
        label=f"线性趋势: +{coef[0]*10:.0f} day·°C / decade")

ax.set_xlabel("年份"); ax.set_ylabel("生长季 GDD (日·°C)")
ax.set_title(f"35 年全国 GDD 时间趋势 (Checkpoint 3)\n"
             f"1982 mean={yearly_mean[0]:.0f} → 2016 mean={yearly_mean[-1]:.0f} "
             f"净增 {yearly_mean[-1]-yearly_mean[0]:+.0f}", fontsize=13)
ax.legend(loc="upper left", fontsize=11)
ax.grid(alpha=0.3)
out2 = GDD_DIR / "preview_gdd_trend_35yr.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
plt.close(fig2)
print(f"saved: {out2}")

# === 图 3: 几个代表区划的 35 年时间序列
fig3, ax = plt.subplots(figsize=(13, 6.5))
with rasterio.open(GDD_DIR / "zone_raster_temp_grid.tif") as src:
    zr = src.read(1)

df_zone = pd.read_csv(INTERMEDIATE / "growing_season" / "zone37_growing_season.csv")
representatives = [
    (13, "黄淮平原 (peak=7)"),
    (25, "松嫩平原 (peak=8)"),
    (37, "华南低平原 (peak=9)"),
    (1, "藏东南 (peak=8, 高原)"),
    (20, "南疆绿洲 (peak=8, 干旱)"),
]
for oid, label in representatives:
    m = zr == oid
    if not m.any(): continue
    series = [float(np.nanmean(arr[i][m])) for i in range(len(years))]
    ax.plot(years, series, "o-", linewidth=1.8, markersize=4, label=label)

ax.set_xlabel("年份"); ax.set_ylabel("生长季 GDD (日·°C)")
ax.set_title("5 个代表性分区的 35 年生长季 GDD 演变", fontsize=13)
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)
out3 = GDD_DIR / "preview_gdd_zone_series.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
plt.close(fig3)
print(f"saved: {out3}")
