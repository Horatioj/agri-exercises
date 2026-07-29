"""
渲染 A 步最终成果: 37 个农业分区, 每个分区一个 peak month (整块上色).
这是 B/C 步真正会用的 zone37_growing_season.csv 的可视化.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "STHeiti", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

GS = INTERMEDIATE / "growing_season"

# 加载 37 区划 + CSV
zones = gpd.read_file(
    DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp"
).to_crs(4326)
df = pd.read_csv(GS / "zone37_growing_season.csv")
zones = zones.merge(df[["OBJECTID_1", "peak_month_main", "NAME",
                         "green_season_months"]],
                    on="OBJECTID_1", how="left", suffixes=("_orig", ""))
prov = gpd.read_file(DATA_ROOT / "00 county/省.shp").to_crs(4326)

# 配色 (与 peak_month.tif 一致)
month_colors = [
    "#3b2a52","#5e3d7a","#7a6cb3","#5b9bd5","#7fc97f","#33a02c",
    "#b8d23a","#ffd92f","#ff9933","#e6550d","#c41e3a","#7a0d1a",
]
cmap = mcolors.ListedColormap(month_colors)
norm = mcolors.BoundaryNorm(np.arange(0.5, 13.5, 1), cmap.N)

fig, ax = plt.subplots(figsize=(15, 11))

# 主图: 37 个分区, 用 peak_month_main 上色
zones.plot(column="peak_month_main", cmap=cmap, norm=norm,
           ax=ax, edgecolor="black", linewidth=0.6)
# 省界轻叠 (浅灰)
prov.boundary.plot(ax=ax, color="#666666", linewidth=0.25, alpha=0.5)

# 给每个分区标上 OID 和 peak 月份
for _, r in zones.iterrows():
    if pd.isna(r.peak_month_main):
        continue
    centroid = r.geometry.representative_point()
    ax.annotate(f"{int(r.OBJECTID_1)}\n→{int(r.peak_month_main)}月",
                xy=(centroid.x, centroid.y),
                ha="center", va="center", fontsize=7.5,
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))

ax.set_xlim(72, 137)
ax.set_ylim(17, 55)
ax.set_xlabel("经度 (°E)", fontsize=12)
ax.set_ylabel("纬度 (°N)", fontsize=12)
ax.set_title("⭐ A 步最终成果：37 个农业熟制区划的 peak month\n"
             "每分区一种颜色 = 该分区的 peak 月份；这是 B/C 步直接读的成果",
             fontsize=14, pad=15)

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    ticks=range(1, 13), shrink=0.75, pad=0.02, fraction=0.04)
cbar.set_label("分区 peak month", fontsize=11)
cbar.ax.set_yticklabels([f"{m} 月" for m in range(1, 13)], fontsize=10)

# 统计框
n7 = int((df.peak_month_main == 7).sum())
n8 = int((df.peak_month_main == 8).sum())
n9 = int((df.peak_month_main == 9).sum())
n11 = int((df.peak_month_main == 11).sum())
stats = (f"37 区划 peak 月份分布:\n"
         f"  7 月: {n7} 区\n"
         f"  8 月: {n8} 区 (最常见)\n"
         f"  9 月: {n9} 区\n"
         f"  11 月: {n11} 区 (滇南山地)")
ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fffacc", ec="black", alpha=0.95))

fig.tight_layout()
out_f = GS / "preview_zone_level_peak_FINAL.png"
fig.savefig(out_f, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out_f}")
