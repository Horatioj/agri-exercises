"""
D 步: 构造年度耕地权重栅格 (per 方案 §8)

输入:
  data/03 plantation/{year}/{name}.tif      9 个 LUCC 切片 (Krasovsky Albers, 1km)
  intermediate/gdd_pixel/zone_raster_temp_grid.tif    B 步温度 0.1° 网格 (700×400)
  intermediate/swi_pixel/zone_raster_sm_grid.tif      C 步 SWVL 0.1° 网格 (651×391)

输出 (intermediate/cropland_weight/):
  cropland_weight_{slice_year}_tempgrid.tif      9 个 (温度网格上)
  cropland_weight_{slice_year}_smgrid.tif        9 个 (SWVL 网格上)
  year_to_lucc_mapping.csv                       35 年 → 切片映射

方法:
1. 加载 LUCC 切片, 提取耕地二值栅格 (code 11 水田 + 12 旱地)
2. 用 rasterio.warp.reproject(..., Resampling.sum) 把 1km Albers → 0.1° EPSG:4326
   结果 = 每个目标像元覆盖范围内的耕地 1km 像元数 (≈ km²)
3. 输出 18 个 GeoTIFF + 1 个映射 CSV

最近年匹配规则 (方案 §8.D2):
  1982-1984 → 1980; 1985-1992 → 1990; 1993-1997 → 1995; 1998-2002 → 2000
  2003-2006 → 2005; 2007-2009 → 2008; 2010-2011 → 2010; 2012-2014 → 2013
  2015-2016 → 2015

⚠ 重要 (Codex D 步复审 P2 文档级):
  严格说这不是"纯机械最近年规则" (nearest-year), 而是"按预设年份段映射"
  (predefined-segment mapping with manual tie handling). 3 个等距 tie 年的
  tie-breaking 方向不统一:
    1985 (距 1980/1990 各 5 年) → 取 1990 (向前)
    2009 (距 2008/2010 各 1 年) → 取 2008 (向后)
    2014 (距 2013/2015 各 1 年) → 取 2013 (向后)
  代码严格按方案表实现, 与方案 §8.D2 一致. 但论文方法学描述时不要写成
  "nearest-year rule", 而应说"按方案预设的年份段映射 (manual segmentation)".

注: B/C 两个目标网格虽然都是 0.1°, 但像元中心错位 0.05° (Codex C 步复审 P1),
所以同一个 LUCC 切片必须各重采样一次, 不能复用.

E 步使用建议 (Codex D 步复审):
  - GDD/温度变量用 cropland_weight_*_tempgrid.tif (B 步网格)
  - RZSM/SM_shallow 用 cropland_weight_*_smgrid.tif (C 步网格)
  - 不能混用 (两套网格半像元错位且范围不同)
  - exactextract.weighted_mean 自动归一化, 无需预处理
  - 若要县域耕地面积, 对 weight 做 coverage-weighted sum
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import INTERMEDIATE, lucc_path

OUT_DIR = INTERMEDIATE / "cropland_weight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LUCC_SLICES = [1980, 1990, 1995, 2000, 2005, 2008, 2010, 2013, 2015]
TEMP_REF = INTERMEDIATE / "gdd_pixel" / "zone_raster_temp_grid.tif"
SM_REF   = INTERMEDIATE / "swi_pixel" / "zone_raster_sm_grid.tif"


def get_grid_info(reference_path: Path) -> dict:
    """从参考栅格读 transform/crs/shape."""
    with rasterio.open(reference_path) as src:
        return dict(
            transform=src.transform,
            crs=src.crs,
            shape=src.shape,
            width=src.width,
            height=src.height,
        )


def load_cropland_1km(year: int):
    """加载 LUCC 切片, 提取耕地二值栅格 (uint8/float32: 1=耕地, 0=非耕地)."""
    f = lucc_path(year)
    with rasterio.open(f) as src:
        arr = src.read(1)
        nd = src.nodata
        transform = src.transform
        crs = src.crs
    safe = np.where(arr == nd, 0, arr)
    cropland = ((safe == 11) | (safe == 12)).astype("float32")
    return cropland, transform, crs


def reproject_sum_to_target(src_arr, src_transform, src_crs, target: dict) -> np.ndarray:
    """用 Resampling.sum 把二值 1km Albers 耕地图重采样到目标 0.1° EPSG:4326 网格.

    结果是每个目标像元覆盖的 1km 耕地像元数 (= 耕地 km²).
    """
    dst = np.zeros(target["shape"], dtype="float32")
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target["transform"],
        dst_crs=target["crs"],
        resampling=Resampling.sum,
    )
    return dst


def year_to_slice_mapping() -> dict:
    """方案 §8.D2 的最近年匹配规则."""
    rules = [
        ((1982, 1984), 1980),
        ((1985, 1992), 1990),
        ((1993, 1997), 1995),
        ((1998, 2002), 2000),
        ((2003, 2006), 2005),
        ((2007, 2009), 2008),
        ((2010, 2011), 2010),
        ((2012, 2014), 2013),
        ((2015, 2016), 2015),
    ]
    mp = {}
    for (lo, hi), slc in rules:
        for y in range(lo, hi + 1):
            mp[y] = slc
    assert len(mp) == 35, f"映射表应覆盖 35 年, 实际 {len(mp)}"
    return mp


def write_tif(path: Path, arr: np.ndarray, grid: dict):
    profile = dict(
        driver="GTiff", dtype="float32", nodata=-1.0,
        count=1, compress="lzw",
        width=grid["width"], height=grid["height"],
        crs=grid["crs"], transform=grid["transform"],
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def main():
    t_global = time.time()

    # Setup
    print("[setup] 加载目标网格信息")
    temp_grid = get_grid_info(TEMP_REF)
    sm_grid   = get_grid_info(SM_REF)
    print(f"  温度网格 (B 步): shape={temp_grid['shape']}, transform={temp_grid['transform']}")
    print(f"  SWVL  网格 (C 步): shape={sm_grid['shape']}, transform={sm_grid['transform']}")
    print(f"  → 半像元错位 (Codex C 步复审 P1), 必须分别重采样")

    # 主循环
    print(f"\n[run] 处理 {len(LUCC_SLICES)} 个 LUCC 切片 × 2 网格 = {2*len(LUCC_SLICES)} 个输出")
    summary_rows = []
    for i, slc_year in enumerate(LUCC_SLICES, 1):
        t0 = time.time()
        print(f"\n  [{i}/{len(LUCC_SLICES)}] LUCC {slc_year}")
        cropland_1km, lucc_tr, lucc_crs = load_cropland_1km(slc_year)
        n_cropland = int(cropland_1km.sum())
        print(f"    1km 耕地像元: {n_cropland:,d}")

        # 温度网格
        cropland_temp = reproject_sum_to_target(cropland_1km, lucc_tr, lucc_crs, temp_grid)
        f_temp = OUT_DIR / f"cropland_weight_{slc_year}_tempgrid.tif"
        write_tif(f_temp, cropland_temp, temp_grid)
        n_temp = float(cropland_temp.sum())
        max_temp = float(cropland_temp.max())
        size_kb_t = f_temp.stat().st_size / 1024
        print(f"    → 温度网格: 总像元值和 = {n_temp:,.0f} (≈ 耕地 km²), max/像元 = {max_temp:.1f}, 文件 {size_kb_t:.0f} KB")

        # SWVL 网格
        cropland_sm = reproject_sum_to_target(cropland_1km, lucc_tr, lucc_crs, sm_grid)
        f_sm = OUT_DIR / f"cropland_weight_{slc_year}_smgrid.tif"
        write_tif(f_sm, cropland_sm, sm_grid)
        n_sm = float(cropland_sm.sum())
        max_sm = float(cropland_sm.max())
        size_kb_s = f_sm.stat().st_size / 1024
        print(f"    → SWVL  网格: 总像元值和 = {n_sm:,.0f} (≈ 耕地 km²), max/像元 = {max_sm:.1f}, 文件 {size_kb_s:.0f} KB")

        dt = time.time() - t0
        summary_rows.append({
            "lucc_slice": slc_year,
            "n_cropland_1km": n_cropland,
            "sum_on_temp_grid": int(n_temp),
            "sum_on_sm_grid": int(n_sm),
            "max_per_pixel_temp": max_temp,
            "max_per_pixel_sm": max_sm,
            "duration_sec": round(dt, 1),
        })
        print(f"    耗时 {dt:.1f}s")

    # 汇总
    print(f"\n[summary] 9 切片重采样统计")
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # 一致性检查: 重采样后总耕地 km² 应接近原始 1km 像元数 (略有损耗在边界)
    diff_temp = (df_summary["sum_on_temp_grid"] - df_summary["n_cropland_1km"]).abs()
    diff_sm   = (df_summary["sum_on_sm_grid"] - df_summary["n_cropland_1km"]).abs()
    pct_temp  = 100 * diff_temp / df_summary["n_cropland_1km"]
    pct_sm    = 100 * diff_sm / df_summary["n_cropland_1km"]
    print(f"\n  重采样保留率:")
    print(f"    温度网格: {(100 - pct_temp).mean():.2f}% (mean), {(100 - pct_temp).min():.2f}% (min)")
    print(f"    SWVL  网格: {(100 - pct_sm).mean():.2f}% (mean), {(100 - pct_sm).min():.2f}% (min)")

    # 年份→切片映射
    year_map = year_to_slice_mapping()
    df_map = pd.DataFrame([{"year": y, "lucc_slice": s} for y, s in sorted(year_map.items())])
    f_map = OUT_DIR / "year_to_lucc_mapping.csv"
    df_map.to_csv(f_map, index=False)
    print(f"\n[OK] 写入 {f_map.name} (35 年映射)")
    print(df_map.head(6).to_string(index=False))
    print("    ...")
    print(df_map.tail(4).to_string(index=False))

    # 反向: 每个切片被几年使用
    rev = df_map.groupby("lucc_slice")["year"].agg(list).to_dict()
    print(f"\n  切片使用年份:")
    for slc in sorted(rev):
        yrs = rev[slc]
        print(f"    {slc}: {yrs[0]}-{yrs[-1]} ({len(yrs)} 年)")

    print(f"\n[OK] D 步完成, 总耗时 {(time.time()-t_global)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
