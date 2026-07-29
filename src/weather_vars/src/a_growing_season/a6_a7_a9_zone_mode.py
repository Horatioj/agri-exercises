"""
A6+A7+A9: 37 区划聚合 + 50% 耕地阈值众数 + 生长季月份

输入:
  intermediate/growing_season/peak_month.tif        (像元月份, NDVI 网格)
  intermediate/growing_season/bottom_month.tif
  data/03 plantation/2000/2000.tif                  (LUCC 2000, 1km Albers, 算耕地占比基准年)
  data/05 agricultural division/01 .../quhua.shp    (37 区划)

方法对齐 Ortiz-Bobea 2021 (1_7_peak_bottom_ndvi.R) 406-481 行:
  1. 把 LUCC code 11(水田) + 12(旱地) → 1km 二值耕地图
  2. 重采样到 NDVI 0.05° 网格, 用 Resampling.average → 每个 NDVI 像元的耕地占比 0-1
  3. 把 37 区划栅格化到 NDVI 网格 (像元值 = OBJECTID_1)
  4. 每个区划独立:
       - 收集该区所有有效像元 (peak_month, bottom_month, crop_share)
       - 阈值 = crop_share 的 50% 分位数; 保留 crop_share >= 阈值的像元
       - peak/bottom 取简单众数

输出:
  intermediate/growing_season/zone37_growing_season.csv  (37 行, 主分析 + 各种 metadata)
  intermediate/growing_season/zone_raster_ndvi_grid.tif  (栅格化 37 区划, 可视化用)
  intermediate/growing_season/cropland_share_ndvi_grid.tif (耕地占比栅格, 可视化用)
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE, lucc_path

GS_DIR = INTERMEDIATE / "growing_season"
SHP_37 = DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp"

BASELINE_LUCC_YEAR = 2000  # 用户确认


def main():
    # ============================ A6: 加载 37 区划
    print("[A6] 加载 37 区划")
    gdf = gpd.read_file(SHP_37)
    print(f"     原 CRS: {gdf.crs.name}  要素数: {len(gdf)}")
    gdf = gdf.to_crs(4326)
    print(f"     to_crs(4326) bounds: {tuple(round(x, 2) for x in gdf.total_bounds)}")
    print(f"     OBJECTID_1 范围: {gdf.OBJECTID_1.min()}-{gdf.OBJECTID_1.max()}")

    # ============================ 读 NDVI 网格元信息 (从 A5 输出的 peak_month.tif)
    peak_f = GS_DIR / "peak_month.tif"
    bot_f = GS_DIR / "bottom_month.tif"
    assert peak_f.exists() and bot_f.exists(), "缺少 A5 输出"

    with rasterio.open(peak_f) as src:
        peak_m = src.read(1)
        ndvi_transform = src.transform
        ndvi_crs = src.crs
        ndvi_shape = src.shape  # (H, W)
        ndvi_profile = src.profile
    with rasterio.open(bot_f) as src:
        bot_m = src.read(1)

    print(f"\n[NDVI 网格] shape={ndvi_shape}, crs={ndvi_crs}")

    # ============================ A7-1: 耕地占比栅格 (2000 LUCC → NDVI 网格)
    print(f"\n[A7-1] 计算耕地占比 (LUCC {BASELINE_LUCC_YEAR})")
    lucc_f = lucc_path(BASELINE_LUCC_YEAR)
    with rasterio.open(lucc_f) as src:
        lucc = src.read(1)
        lucc_nd = src.nodata
        lucc_transform = src.transform
        lucc_crs = src.crs
        lucc_shape = src.shape

    # 耕地二值 (11=水田, 12=旱地)
    safe = np.where(lucc == lucc_nd, 0, lucc)
    cropland_bin = ((safe == 11) | (safe == 12)).astype("float32")
    print(f"        LUCC {lucc_shape} @ Albers 1km → 耕地像元 {int(cropland_bin.sum()):,d}")

    # 重采样到 NDVI 网格 (用 average → 落在每个 NDVI 像元的 1km 块的耕地比例)
    crop_share = np.zeros(ndvi_shape, dtype="float32")
    reproject(
        source=cropland_bin,
        destination=crop_share,
        src_transform=lucc_transform,
        src_crs=lucc_crs,
        dst_transform=ndvi_transform,
        dst_crs=ndvi_crs,
        resampling=Resampling.average,
    )
    print(f"        重采样到 NDVI 网格. crop_share 分布:")
    for p in [50, 75, 90, 99]:
        print(f"          P{p:2d} = {np.percentile(crop_share, p):.3f}")
    print(f"          max  = {crop_share.max():.3f}")

    # 存盘可视化检查
    out_crop = GS_DIR / "cropland_share_ndvi_grid.tif"
    with rasterio.open(
        out_crop, "w",
        driver="GTiff", dtype="float32", nodata=-1,
        width=ndvi_shape[1], height=ndvi_shape[0], count=1,
        crs=ndvi_crs, transform=ndvi_transform, compress="lzw",
    ) as dst:
        dst.write(crop_share, 1)
    print(f"        存 {out_crop.name}")

    # ============================ A7-2: 区划栅格化
    print(f"\n[A7-2] 栅格化 37 区划到 NDVI 网格")
    shapes = [(geom, oid) for geom, oid in zip(gdf.geometry, gdf.OBJECTID_1)]
    zone_raster = rasterize(
        shapes,
        out_shape=ndvi_shape,
        transform=ndvi_transform,
        fill=0,                # 0 = 区划外
        dtype="int32",
    )
    n_in_zone = int((zone_raster > 0).sum())
    print(f"        落在任一区划内的像元: {n_in_zone:,d}/{zone_raster.size:,d} ({100*n_in_zone/zone_raster.size:.1f}%)")

    out_zone = GS_DIR / "zone_raster_ndvi_grid.tif"
    with rasterio.open(
        out_zone, "w",
        driver="GTiff", dtype="int32", nodata=0,
        width=ndvi_shape[1], height=ndvi_shape[0], count=1,
        crs=ndvi_crs, transform=ndvi_transform, compress="lzw",
    ) as dst:
        dst.write(zone_raster, 1)
    print(f"        存 {out_zone.name}")

    # ============================ A7-3: 每区划 50% 阈值众数
    print(f"\n[A7-3] 每区划 50% 耕地阈值 + 众数")
    print(f"        + 同时算稳健性 (无阈值简单众数)")

    rows = []
    # 把所有有效像元 (peak_m > 0 且 zone > 0) 拉平成 1D
    valid_mask = (peak_m > 0) & (zone_raster > 0)
    flat_zone = zone_raster[valid_mask]
    flat_peak = peak_m[valid_mask]
    flat_bot = bot_m[valid_mask]
    flat_crop = crop_share[valid_mask]

    print(f"        有效像元总数: {flat_zone.size:,d}")

    for _, row in gdf.iterrows():
        oid = int(row["OBJECTID_1"])
        name = str(row.get("NAME", ""))

        sel = (flat_zone == oid)
        n_cells_all = int(sel.sum())
        if n_cells_all == 0:
            rows.append({
                "OBJECTID_1": oid, "NAME": name,
                "n_cells_all": 0,
                "peak_month_main": None,
                "bottom_month_main": None,
                "peak_month_unfiltered": None,
                "bottom_month_unfiltered": None,
                "crop_share_p50": None,
                "n_cells_filtered": 0,
            })
            continue

        z_peak = flat_peak[sel]
        z_bot = flat_bot[sel]
        z_crop = flat_crop[sel]

        # 主分析: 50% 阈值 + 简单众数
        thresh = float(np.percentile(z_crop, 50))
        keep = z_crop >= thresh
        if keep.sum() == 0:
            peak_main = bot_main = None
        else:
            peak_main = int(pd.Series(z_peak[keep]).mode().iloc[0])
            bot_main = int(pd.Series(z_bot[keep]).mode().iloc[0])

        # 稳健性: 不筛选直接众数
        peak_un = int(pd.Series(z_peak).mode().iloc[0])
        bot_un = int(pd.Series(z_bot).mode().iloc[0])

        rows.append({
            "OBJECTID_1": oid, "NAME": name,
            "n_cells_all": n_cells_all,
            "peak_month_main": peak_main,
            "bottom_month_main": bot_main,
            "peak_month_unfiltered": peak_un,
            "bottom_month_unfiltered": bot_un,
            "crop_share_p50": round(thresh, 4),
            "n_cells_filtered": int(keep.sum()),
        })

    df = pd.DataFrame(rows)

    # ============================ A9: 生长季 5 月 (peak 前后各 2 个月, 跨年正确处理)
    # 顺序: peak-2, peak-1, peak, peak+1, peak+2 (自然时间顺序, 跨年 wrap)
    # 例: peak=11 → 9,10,11,12,1; peak=7 → 5,6,7,8,9
    # 不要 sorted, 否则跨年的会被字典序打乱
    # B/C/D 步解析必须用 set/isin, 不能用 [0] 当起始月、[-1] 当结束月
    def season_months(peak_m):
        if peak_m is None:
            return ""
        months = [((peak_m - 2 + i - 1) % 12) + 1 for i in range(5)]
        return ",".join(str(m) for m in months)

    df["green_season_months"] = df["peak_month_main"].apply(season_months)
    df["brown_season_months"] = df["bottom_month_main"].apply(season_months)

    # ============================ 输出
    out_csv = GS_DIR / "zone37_growing_season.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[A9] 写入 {out_csv.name}")

    print(f"\n========== 37 区划生长季 (主分析) ==========")
    show = df[["OBJECTID_1", "NAME", "peak_month_main", "bottom_month_main",
               "green_season_months", "n_cells_all", "n_cells_filtered"]].copy()
    show["NAME"] = show["NAME"].str.slice(0, 22)
    print(show.to_string(index=False))

    print(f"\n========== peak_month_main 分布 ==========")
    print(df["peak_month_main"].value_counts().sort_index().to_string())

    print(f"\n========== 主分析 vs 不筛选 (peak month 不同的区划) ==========")
    diff = df[df["peak_month_main"] != df["peak_month_unfiltered"]]
    if len(diff) == 0:
        print("  无差异: 主分析与不筛选给出相同 peak month, 阈值筛选效果在 37 区划层面不显著")
    else:
        print(diff[["OBJECTID_1", "NAME", "peak_month_main", "peak_month_unfiltered"]].to_string(index=False))


if __name__ == "__main__":
    main()
