"""
E 步: 像元层 → 县-年长表 (PAC × year)

输入:
  data/00 county/县.shp                                  2900 个县 (去 2 个异常 → 2898)
  intermediate/gdd_pixel/gdd_growing_season_1982_2016.nc B 步 GDD (35×400×700)
  intermediate/swi_pixel/swvl_growing_season_1982_2016.nc C 步 SWVL (35×391×651, rzsm_gs+sm_shallow_gs)
  intermediate/irr_pixel/irr_growing_season_1982_2016.nc G 步 灌溉 (35×355×615, irr_gs)
  intermediate/cropland_weight/cropland_weight_*_{tempgrid,smgrid}.tif  D 步 18 个权重栅格
  intermediate/cropland_weight/year_to_lucc_mapping.csv  35 年 → 切片映射
  intermediate/gdd_pixel/zone_raster_temp_grid.tif       温度网格上的 37 区划 (跨区划标记用)

输出 (output/):
  county_year_gdd.parquet          PAC × year × gdd_growing_season
  county_year_rzsm.parquet         PAC × year × rzsm_gs
  county_year_sm_shallow.parquet   PAC × year × sm_shallow_gs
  county_year_irrigation.parquet   PAC × year × {irr_gs, irr_annual}  (G 步新增)
  county_year_cropland.parquet     PAC × year × cropland_area_km2
  county_zone_info.parquet         PAC × zone37_main × dominant_share × is_cross_zone (静态)

方法 (per 方案 §9 + 2026-06 方案 B 升级):
- E1  加载县 + 清理 (去 PAC=156408, 710024) + to_crs(4326)
- E2  跨区划标记: exact_extract(zone_raster, xian, ops=['unique','frac']),
      按 coverage_fraction 取主导 zone (面积加权, 不需 cropland_weight).
      *** 2026-06 升级: 旧版用 rasterize+cropland_weight 做 numpy 聚合, 会漏小县
                       (东城/西城等市辖区 130 个), 已下沉到此处统一修复 ***
- E3  每年县耕地面积 (per slice 算一次, 按 mapping 展到 35 年)
- E4  每年每县 3 变量 GDD/RZSM/SM_shallow (35×3 次 weighted_mean, 用对应网格的 cropland_weight)
- E4b 每年每县 灌溉 sum (35 次 sum, 无权重 — 灌溉值本身就是 0.1° 格的总用水量)
- E5  写 6 个 parquet

关键约束 (Codex D 步复审):
- GDD       必须用 cropland_weight_*_tempgrid.tif (weighted_mean)
- RZSM      必须用 cropland_weight_*_smgrid.tif    (weighted_mean)
- SM_shallow 必须用 cropland_weight_*_smgrid.tif    (weighted_mean)
- 灌溉      不用权重栅格, 直接 ops=['sum']        (体积量, 已经是 0.1° 格的总量)
- 不能混用 (B/C/G 三套网格各自半像元错位且范围不同)

exactextract API (0.2.0):
  exact_extract(value_raster, gdf, ops=['weighted_mean'], weights=<path>,
                include_cols=['PAC'], output='pandas')   # GDD/RZSM/SM_shallow
  exact_extract(value_raster, gdf, ops=['sum'],
                include_cols=['PAC'], output='pandas')   # 灌溉, 无 weights
  exact_extract(zone_raster, gdf, ops=['unique','frac'],
                include_cols=['PAC'], output='pandas')   # E2 跨区划标记
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from exactextract import exact_extract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE, OUTPUT

warnings.filterwarnings("ignore", category=RuntimeWarning)

XIAN_SHP = DATA_ROOT / "00 county" / "县.shp"
GDD_NC   = INTERMEDIATE / "gdd_pixel" / "gdd_growing_season_1982_2016.nc"
SWVL_NC  = INTERMEDIATE / "swi_pixel" / "swvl_growing_season_1982_2016.nc"
IRR_NC   = INTERMEDIATE / "irr_pixel" / "irr_growing_season_1982_2016.nc"   # G 步产出
CW_DIR   = INTERMEDIATE / "cropland_weight"
MAPPING  = CW_DIR / "year_to_lucc_mapping.csv"
ZONE_TEMP_TIF = INTERMEDIATE / "gdd_pixel" / "zone_raster_temp_grid.tif"  # E2 跨区划标记用


def prep_value_da(da: xr.DataArray) -> xr.DataArray:
    """给 NetCDF 切出来的 DataArray 补 CRS + spatial_dims (B/C 步存 nc 时 lat/lon 没显式 CRS)."""
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    return da


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-range", default="1982-2016", help="e.g. 1982-2016 or 2010-2010 (MVP)")
    args = ap.parse_args()
    y0, y1 = map(int, args.year_range.split("-"))
    target_years = set(range(y0, y1 + 1))
    is_full = (y0 == 1982 and y1 == 2016)
    print(f"[E 步] 年份范围 {y0}-{y1} ({len(target_years)} 年)  {'FULL' if is_full else 'MVP'}")

    t_global = time.time()

    # ============================ E1: 加载县
    print("\n[E1] 加载县界")
    xian = gpd.read_file(XIAN_SHP)
    n_raw = len(xian)
    xian = xian[~xian.PAC.isin([156408, 710024])].to_crs(4326).reset_index(drop=True)
    print(f"     原 {n_raw} 行 → 去 2 异常 → {len(xian)} 行, PAC 唯一 {xian.PAC.nunique()}")

    # 年份映射
    mapping = pd.read_csv(MAPPING)
    slice_to_years = mapping.groupby("lucc_slice")["year"].apply(list).to_dict()
    # 只保留 target_years 内的
    slice_to_years_filt = {
        slc: [y for y in yrs if y in target_years]
        for slc, yrs in slice_to_years.items()
    }
    slice_to_years_filt = {k: v for k, v in slice_to_years_filt.items() if v}
    n_year_total = sum(len(v) for v in slice_to_years_filt.values())
    print(f"     mapping: {len(slice_to_years_filt)} 切片, 覆盖 {n_year_total} 年")

    # ============================ E2: 跨区划标记 (静态)
    # !! 方案 B 升级 (2026-06 Codex 复审采纳):
    #    旧版用 rasterize(县→温度网格) + cropland_weight 做主导, 但 rasterize 用几何中心染色,
    #    小县 (东城/西城等市辖区) 中心不落在任何像元中心 → 完全不染色, 130 个县 zone37_main=NaN.
    #    新版用 exact_extract(unique, frac) 按 coverage_fraction 取主导, 同时:
    #      - 不需要 cropland_weight (按面积加权, 不按耕地)
    #      - 不会漏小县
    #      - 与之前 fix_panel_post_codex_review.py 的 recompute_zone_info 算法一致
    print("\n[E2] 跨区划标记 (exact_extract unique+frac, 按面积加权; 替代旧 rasterize)")
    r = exact_extract(
        str(ZONE_TEMP_TIF), xian,
        ops=["unique", "frac"],
        include_cols=["PAC"],
        output="pandas",
    )
    rows = []
    for _, row in r.iterrows():
        pac = int(row["PAC"])
        zones = list(row["unique"]) if row["unique"] is not None else []
        fracs = list(row["frac"]) if row["frac"] is not None else []
        # 剔除 zone=0 (37 区划之外的部分, OOB)
        valid = [(z, f) for z, f in zip(zones, fracs) if z > 0]
        if not valid:
            rows.append({
                "PAC": pac,
                "zone37_main": pd.NA,
                "dominant_share": pd.NA,
                "is_cross_zone": pd.NA,
            })
            continue
        # 只在 zone>0 内重新归一化 (排除 OOB 像素)
        total_in_zones = sum(f for _, f in valid)
        z_dom, f_dom = max(valid, key=lambda x: x[1])
        share = f_dom / total_in_zones if total_in_zones > 0 else 0.0
        rows.append({
            "PAC": pac,
            "zone37_main": int(z_dom),
            "dominant_share": float(share),
            "is_cross_zone": bool(share < 0.9),
        })
    df_zone_info = pd.DataFrame(rows).sort_values("PAC").reset_index(drop=True)
    n_with_zone = int(df_zone_info["zone37_main"].notna().sum())
    n_cz = int(df_zone_info["is_cross_zone"].fillna(False).astype(bool).sum())
    print(f"     有 zone 的县: {n_with_zone}/{len(xian)}")
    print(f"     跨区划县 (主导<90%): {n_cz}/{n_with_zone} ({100*n_cz/max(1,n_with_zone):.1f}%)")

    # ============================ E3: 每年县耕地面积
    print("\n[E3] 每年每县耕地面积 (per slice 一次 + 按 mapping 展到 35 年)")
    cropland_per_slice = {}
    for slc, years in slice_to_years_filt.items():
        tif = CW_DIR / f"cropland_weight_{slc}_tempgrid.tif"
        r = exact_extract(str(tif), xian, ops=["sum"],
                          include_cols=["PAC"], output="pandas")
        r = r.rename(columns={"sum": "cropland_area_km2"})
        cropland_per_slice[slc] = r
        print(f"     切片 {slc}: 总和 {float(r['cropland_area_km2'].sum()):,.0f} km² / {len(years)} 年")
    cropland_rows = []
    for slc, years in slice_to_years_filt.items():
        for y in years:
            df_y = cropland_per_slice[slc].copy()
            df_y["year"] = y
            cropland_rows.append(df_y)
    df_cropland = pd.concat(cropland_rows, ignore_index=True)[["PAC", "year", "cropland_area_km2"]]

    # ============================ E4: 三变量 per 年 zonal stats
    print(f"\n[E4] 每年每县 GDD / RZSM / SM_shallow (3 × {n_year_total} = {3*n_year_total} 次 exact_extract)")
    ds_gdd  = xr.open_dataset(GDD_NC)
    ds_swvl = xr.open_dataset(SWVL_NC)

    all_gdd, all_rzsm, all_sm = [], [], []
    t0 = time.time()
    counter = 0
    for slc, years in slice_to_years_filt.items():
        cw_temp_path = str(CW_DIR / f"cropland_weight_{slc}_tempgrid.tif")
        cw_sm_path   = str(CW_DIR / f"cropland_weight_{slc}_smgrid.tif")
        for y in years:
            counter += 1
            t_y = time.time()
            # GDD
            gdd_y = prep_value_da(ds_gdd["gdd_growing_season"].sel(year=y))
            r = exact_extract(gdd_y, xian, ops=["weighted_mean"],
                              weights=cw_temp_path, include_cols=["PAC"], output="pandas")
            r["year"] = y
            all_gdd.append(r.rename(columns={"weighted_mean": "gdd_growing_season"}))
            # RZSM
            rzsm_y = prep_value_da(ds_swvl["rzsm_gs"].sel(year=y))
            r = exact_extract(rzsm_y, xian, ops=["weighted_mean"],
                              weights=cw_sm_path, include_cols=["PAC"], output="pandas")
            r["year"] = y
            all_rzsm.append(r.rename(columns={"weighted_mean": "rzsm_gs"}))
            # SM_shallow
            sm_y = prep_value_da(ds_swvl["sm_shallow_gs"].sel(year=y))
            r = exact_extract(sm_y, xian, ops=["weighted_mean"],
                              weights=cw_sm_path, include_cols=["PAC"], output="pandas")
            r["year"] = y
            all_sm.append(r.rename(columns={"weighted_mean": "sm_shallow_gs"}))
            print(f"     [{counter:3d}/{n_year_total}] {y} (切片 {slc}): {time.time()-t_y:.1f}s")
    print(f"     E4 完成 {(time.time()-t0)/60:.1f} 分钟")

    df_gdd  = pd.concat(all_gdd,  ignore_index=True)[["PAC", "year", "gdd_growing_season"]]
    df_rzsm = pd.concat(all_rzsm, ignore_index=True)[["PAC", "year", "rzsm_gs"]]
    df_sm   = pd.concat(all_sm,   ignore_index=True)[["PAC", "year", "sm_shallow_gs"]]

    # ============================ E4b: 灌溉 per 年 zonal sum (无耕地权重)
    # 灌溉值本身是 0.1° 格的总用水量, 县级聚合用 ops=['sum'] = ∑ (像元值 × 县在像元内面积占比).
    # NaN 像元 (海洋/沙漠/水体) 自动被 exactextract 忽略.
    # 两个变量并列: irr_gs (生长季 sum) + irr_annual (全年 sum), 同口径可比.
    print(f"\n[E4b] 每年每县 irrigation sum ({2*n_year_total} 次 exact_extract: irr_gs + irr_annual, 无 weights)")
    ds_irr = xr.open_dataset(IRR_NC)
    all_irr_gs, all_irr_an = [], []
    t0 = time.time()
    counter = 0
    for y in sorted(target_years):
        counter += 1
        t_y = time.time()
        # irr_gs (生长季)
        da_gs = prep_value_da(ds_irr["irr_gs"].sel(year=y))
        r_gs = exact_extract(da_gs, xian, ops=["sum"],
                             include_cols=["PAC"], output="pandas")
        r_gs["year"] = y
        all_irr_gs.append(r_gs.rename(columns={"sum": "irr_gs"}))
        # irr_annual (全年 12 月)
        da_an = prep_value_da(ds_irr["irr_annual"].sel(year=y))
        r_an = exact_extract(da_an, xian, ops=["sum"],
                             include_cols=["PAC"], output="pandas")
        r_an["year"] = y
        all_irr_an.append(r_an.rename(columns={"sum": "irr_annual"}))
        print(f"     [{counter:3d}/{n_year_total}] {y}: {time.time()-t_y:.1f}s (gs+annual)")
    print(f"     E4b 完成 {(time.time()-t0)/60:.1f} 分钟")
    df_irr_gs = pd.concat(all_irr_gs, ignore_index=True)[["PAC", "year", "irr_gs"]]
    df_irr_an = pd.concat(all_irr_an, ignore_index=True)[["PAC", "year", "irr_annual"]]
    df_irr = df_irr_gs.merge(df_irr_an, on=["PAC", "year"], how="outer")

    # ============================ E5: 写 parquet
    print("\n[E5] 写入 parquet 到 output/")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_files = {
        "county_year_gdd.parquet":         df_gdd,
        "county_year_rzsm.parquet":        df_rzsm,
        "county_year_sm_shallow.parquet":  df_sm,
        "county_year_irrigation.parquet":  df_irr,
        "county_year_cropland.parquet":    df_cropland,
        "county_zone_info.parquet":        df_zone_info,
    }
    for fname, df in out_files.items():
        f = OUTPUT / fname
        df.to_parquet(f, index=False)
        print(f"     {fname}: {len(df):>7,d} 行  ({f.stat().st_size/1024:.0f} KB)")

    # ============================ 健全性检查
    print("\n[validate]")
    n_pac = len(xian)
    expected = n_pac * n_year_total
    print(f"  期望长表行数: {n_pac} × {n_year_total} = {expected:,d}")
    print(f"  GDD       : {len(df_gdd):,d}    NaN {int(df_gdd.gdd_growing_season.isna().sum()):,d} ({100*df_gdd.gdd_growing_season.isna().mean():.1f}%)")
    print(f"  RZSM      : {len(df_rzsm):,d}    NaN {int(df_rzsm.rzsm_gs.isna().sum()):,d} ({100*df_rzsm.rzsm_gs.isna().mean():.1f}%)")
    print(f"  SM_shallow: {len(df_sm):,d}    NaN {int(df_sm.sm_shallow_gs.isna().sum()):,d} ({100*df_sm.sm_shallow_gs.isna().mean():.1f}%)")
    print(f"  Irrigation: {len(df_irr):,d}    irr_gs NaN {int(df_irr.irr_gs.isna().sum()):,d} ({100*df_irr.irr_gs.isna().mean():.1f}%)  "
          f"irr_annual NaN {int(df_irr.irr_annual.isna().sum()):,d} ({100*df_irr.irr_annual.isna().mean():.1f}%)")
    print(f"  Cropland  : {len(df_cropland):,d}    PAC unique {df_cropland.PAC.nunique()}")
    print(f"  Zone info : {len(df_zone_info):,d}    跨区划 {df_zone_info.is_cross_zone.sum()}")

    # 灌溉量纲健全性: 县级总和 ≈ 像元总和 (exactextract sum 保形, 误差应 <1%)
    irr_gs_total = float(df_irr["irr_gs"].sum(skipna=True))
    irr_an_total = float(df_irr["irr_annual"].sum(skipna=True))
    print(f"\n  Irrigation 量纲检查 (县级 sum 总和, 10^8 m³, 全 35 年):")
    print(f"    irr_gs     总和 = {irr_gs_total:,.0f}   (期望 1982-2016 累计 ~60000-70000)")
    print(f"    irr_annual 总和 = {irr_an_total:,.0f}   (期望 > irr_gs, 比值 ≈ 生长季月数/12)")
    if irr_an_total > 0:
        print(f"    生长季占比      = {100*irr_gs_total/irr_an_total:.1f}%")

    # 2010 抽样: 几个标志点位 (含灌溉两个变量)
    if 2010 in target_years:
        print("\n  2010 标志县抽样:")
        for pac, name in [(110105, "北京朝阳"), (410105, "郑州金水"), (320105, "南京建邺")]:
            g = df_gdd[(df_gdd.PAC == pac) & (df_gdd.year == 2010)]
            r = df_rzsm[(df_rzsm.PAC == pac) & (df_rzsm.year == 2010)]
            i = df_irr[(df_irr.PAC == pac) & (df_irr.year == 2010)]
            c = df_cropland[(df_cropland.PAC == pac) & (df_cropland.year == 2010)]
            if len(g) and len(r):
                gv = float(g.gdd_growing_season.iloc[0]) if not g.gdd_growing_season.isna().iloc[0] else None
                rv = float(r.rzsm_gs.iloc[0]) if not r.rzsm_gs.isna().iloc[0] else None
                igs = float(i.irr_gs.iloc[0]) if len(i) and not i.irr_gs.isna().iloc[0] else None
                ian = float(i.irr_annual.iloc[0]) if len(i) and not i.irr_annual.isna().iloc[0] else None
                cv = float(c.cropland_area_km2.iloc[0]) if len(c) else None
                print(f"    PAC {pac} ({name}): GDD={gv}, RZSM={rv}, irr_gs={igs}, irr_annual={ian}, 耕地 km²={cv}")

    print(f"\n[OK] E 步完成, 总耗时 {(time.time()-t_global)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
