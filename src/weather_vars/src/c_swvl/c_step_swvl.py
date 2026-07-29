"""
C 步: ERA5 SWVL → 生长季根区/浅层土壤湿度

按修订方案 §7 (2026-05-28) 实现:
- C1 加载: 单文件 open_dataset, 切 1982-2016, 断言 420 月
- C2 层厚加权: RZSM(0-100cm) = (7*sm1 + 21*sm2 + 72*sm3)/100
              SM_shallow(0-28cm) = (7*sm1 + 21*sm2)/28
- C3 区划栅格化: 按 SM 的 0.1° 网格重做; 区划必须 to_crs(4326);
                **主键用 OBJECTID_1 (A/B 步同源, 连续 1-37), 不要用 areanumber**
- C4 逐区划年均: 状态变量取均值, 月份用 set/isin (utils.parse_season_months)

输入:
  /Users/zhangxinzhen/Desktop/TFP/Climate data/02 SM_data.nc
  intermediate/growing_season/zone37_growing_season.csv (peak_month_main, green_season_months)
  data/05 agricultural division/01 中国农业熟制区划quhua/quhua.shp

输出:
  intermediate/swi_pixel/swvl_growing_season_1982_2016.nc  (year=35, lat=391, lon=651, vars=rzsm_gs+sm_shallow_gs)
  intermediate/swi_pixel/zone_raster_sm_grid.tif           (37 区划 in 0.1° SM 网格)
  intermediate/swi_pixel/{rzsm_gs,sm_shallow_gs}_{1982,2000,2016}.tif  抽样供 QGIS

跨年生长季约定 (与 B 步一致):
  OID 5 peak=11, green={9,10,11,12,1}, "1982 年值" = 1982 calendar year 内的 Jan + Sep-Dec 平均.
  这与 NCC 原方法一致, 但意味着跨年季的"年值"实际混了上一季尾巴和下一季开头.
"""
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE, parse_season_months

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

SM_NC   = DATA_ROOT / "02 SM_data.nc"
SHP_37  = DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp"
CSV_A   = INTERMEDIATE / "growing_season" / "zone37_growing_season.csv"
OUT_DIR = INTERMEDIATE / "swi_pixel"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_global = time.time()

    # ============================ C1: 加载 + 切片 + 断言
    print("[C1] 加载 ERA5 SWVL")
    print(f"     源文件: {SM_NC.name}")
    ds = xr.open_dataset(SM_NC)
    print(f"     原始时间: {str(ds.valid_time.values[0])[:10]} → {str(ds.valid_time.values[-1])[:10]} ({len(ds.valid_time)} 月)")
    ds = ds.sel(valid_time=slice("1982-01-01", "2016-12-31"))
    n_months = len(ds.valid_time)
    assert n_months == 420, f"期望 420 月 (35×12), 实得 {n_months}"
    print(f"     切片后 : {str(ds.valid_time.values[0])[:10]} → {str(ds.valid_time.values[-1])[:10]} ({n_months} 月 ✓)")
    print(f"     网格   : shape=({len(ds.latitude)}, {len(ds.longitude)})")
    print(f"     lat    : {float(ds.latitude[0]):.2f} → {float(ds.latitude[-1]):.2f} (N→S, step={float(ds.latitude[1]-ds.latitude[0]):.3f})")
    print(f"     lon    : {float(ds.longitude[0]):.2f} → {float(ds.longitude[-1]):.2f} (W→E, step={float(ds.longitude[1]-ds.longitude[0]):.3f})")

    # ============================ C2: 层厚加权合成
    print("\n[C2] 深度加权合成 RZSM (0-100cm) + SM_shallow (0-28cm)")
    sm1 = ds["swvl1"].astype("float32")
    sm2 = ds["swvl2"].astype("float32")
    sm3 = ds["swvl3"].astype("float32")

    rzsm_monthly = ((7 * sm1 + 21 * sm2 + 72 * sm3) / 100).astype("float32")
    sm_shallow_monthly = ((7 * sm1 + 21 * sm2) / 28).astype("float32")

    rzsm_monthly.attrs = {
        "units": "m^3/m^3",
        "long_name": "Root-zone soil moisture (0-100 cm, depth-weighted)",
        "formula": "(7*swvl1 + 21*swvl2 + 72*swvl3) / 100",
        "source": "ERA5 SWVL layers 1-3",
    }
    sm_shallow_monthly.attrs = {
        "units": "m^3/m^3",
        "long_name": "Shallow soil moisture (0-28 cm, depth-weighted)",
        "formula": "(7*swvl1 + 21*swvl2) / 28",
        "source": "ERA5 SWVL layers 1-2",
    }

    # 抽样比对
    print(f"     RZSM       1982-01 全国均值 = {float(rzsm_monthly.isel(valid_time=0).mean(skipna=True)):.4f} m^3/m^3")
    print(f"     SM_shallow 1982-01 全国均值 = {float(sm_shallow_monthly.isel(valid_time=0).mean(skipna=True)):.4f} m^3/m^3")
    print(f"     RZSM       2010-07 全国均值 = {float(rzsm_monthly.sel(valid_time='2010-07-01').mean(skipna=True)):.4f}")

    # 拉到 numpy 以便快速逐区划循环
    rzsm_arr  = rzsm_monthly.values             # (420, 391, 651)
    sm_sh_arr = sm_shallow_monthly.values
    H, W = rzsm_arr.shape[1], rzsm_arr.shape[2]
    times = pd.to_datetime(ds.valid_time.values)
    months_idx = times.month.to_numpy()
    years_idx  = times.year.to_numpy()
    ds.close()

    # ============================ C3: 区划栅格化到 SM 0.1° 网格
    print("\n[C3] 栅格化 37 区划到 SM 网格 (按 OBJECTID_1)")
    gdf = gpd.read_file(SHP_37)
    print(f"     原 CRS: {gdf.crs.name} (Krasovsky Albers)")
    gdf = gdf.to_crs(4326)
    print(f"     to_crs(4326) 完成, {len(gdf)} 个面")

    lat_arr = np.linspace(55.0, 16.0, 391).astype("float32")    # 与数据一致
    lon_arr = np.linspace(72.0, 137.0, 651).astype("float32")
    # 仿射变换: 西北角起算 (lat 自北向南)
    transform = from_origin(west=72.0 - 0.05, north=55.0 + 0.05, xsize=0.1, ysize=0.1)

    # !! 关键: 用 OBJECTID_1 而不是 areanumber
    shapes = [(geom, int(oid)) for geom, oid in zip(gdf.geometry, gdf.OBJECTID_1)]
    zone_raster = rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    n_in_zone = int((zone_raster > 0).sum())
    unique_oids = sorted(set(np.unique(zone_raster).tolist()) - {0})
    print(f"     落入区划: {n_in_zone:,d}/{H*W:,d} ({100*n_in_zone/(H*W):.1f}%)")
    print(f"     出现的 OBJECTID_1: {len(unique_oids)} 个 (期望 37)")
    if len(unique_oids) < 37:
        missing = set(range(1, 38)) - set(unique_oids)
        print(f"     ⚠ 缺失的 OID: {sorted(missing)}")

    # 存盘
    zone_tif = OUT_DIR / "zone_raster_sm_grid.tif"
    with rasterio.open(
        zone_tif, "w",
        driver="GTiff", dtype="int32", nodata=0,
        width=W, height=H, count=1,
        crs="EPSG:4326", transform=transform, compress="lzw",
    ) as dst:
        dst.write(zone_raster, 1)
    print(f"     存 {zone_tif.name}")

    # 生长季查找表 (与 A/B 步一致, 用 OBJECTID_1)
    df_zone = pd.read_csv(CSV_A)
    season_lookup = {
        int(r.OBJECTID_1): parse_season_months(r.green_season_months)
        for _, r in df_zone.iterrows() if pd.notna(r.peak_month_main)
    }
    print(f"     season_lookup: {len(season_lookup)} 区划")
    for oid in [5, 13, 25]:
        if oid in season_lookup:
            print(f"       例 OID {oid:2d}: 生长季月份 = {sorted(season_lookup[oid])}")

    # ============================ C4: 逐区划生长季年均 (双变量)
    print("\n[C4] 逐区划生长季年均 (RZSM + SM_shallow)")
    years_list = list(range(1982, 2017))
    n_years = len(years_list)
    rzsm_gs = np.full((n_years, H, W), np.nan, dtype="float32")
    sm_sh_gs = np.full((n_years, H, W), np.nan, dtype="float32")

    t0 = time.time()
    skipped = 0
    for oid in range(1, 38):
        if oid not in season_lookup:
            skipped += 1
            continue
        months = season_lookup[oid]
        zmask = (zone_raster == oid)
        if not zmask.any():
            print(f"     ⚠ OID {oid} 在 SM 网格无像元落入, 跳过")
            continue

        in_season_global = np.isin(months_idx, list(months))

        for yi, y in enumerate(years_list):
            ym = in_season_global & (years_idx == y)
            n_m = int(ym.sum())
            if n_m != len(months):
                # 一般不会发生; 1982 一月之前没数据, 但跨年季 OID 5 包含 Jan 1982 → 应正常
                pass
            rzsm_z = np.nanmean(rzsm_arr[ym], axis=0)       # (H, W)
            sm_z   = np.nanmean(sm_sh_arr[ym], axis=0)
            rzsm_gs[yi] = np.where(zmask, rzsm_z, rzsm_gs[yi])
            sm_sh_gs[yi] = np.where(zmask, sm_z,   sm_sh_gs[yi])

    print(f"     完成 {time.time()-t0:.1f}s (skipped {skipped} 个区划)")

    # n_valid 健全性 (35 年应稳定)
    print("\n[validate] 每年 n_valid (应稳定)")
    for y in [1982, 1990, 2000, 2010, 2016]:
        yi = years_list.index(y)
        n_v = int(np.sum(~np.isnan(rzsm_gs[yi])))
        print(f"     {y}: n_valid = {n_v:,d}")

    # ============================ 存 NetCDF
    # 维度名用 lat / lon, 与 B 步保持一致 (Codex 复审 P2: latitude/longitude vs lat/lon 不一致)
    print("\n[save] 写 NetCDF + 抽样 GeoTIFF")
    out_ds = xr.Dataset(
        {
            "rzsm_gs":      (("year", "lat", "lon"), rzsm_gs),
            "sm_shallow_gs": (("year", "lat", "lon"), sm_sh_gs),
        },
        coords={"year": years_list, "lat": lat_arr, "lon": lon_arr},
    )
    out_ds["rzsm_gs"].attrs = {
        "units": "m^3/m^3",
        "long_name": "Growing-season root-zone soil moisture",
        "depth": "0-100 cm",
        "formula": "(7*swvl1 + 21*swvl2 + 72*swvl3)/100, then mean over zone-specific growing-season months",
        "source": "ERA5 SWVL layers 1-3 + A-step zone37_growing_season.csv",
    }
    out_ds["sm_shallow_gs"].attrs = {
        "units": "m^3/m^3",
        "long_name": "Growing-season shallow soil moisture",
        "depth": "0-28 cm",
        "formula": "(7*swvl1 + 21*swvl2)/28, then mean over zone-specific growing-season months",
        "source": "ERA5 SWVL layers 1-2 + A-step zone37_growing_season.csv",
    }
    out_nc = OUT_DIR / "swvl_growing_season_1982_2016.nc"
    if out_nc.exists():
        out_nc.unlink()
    out_ds.to_netcdf(out_nc, encoding={
        "rzsm_gs":      {"zlib": True, "complevel": 4},
        "sm_shallow_gs": {"zlib": True, "complevel": 4},
    })
    print(f"     存 {out_nc.name} ({out_nc.stat().st_size/1024**2:.1f} MB)")

    # 抽样 GeoTIFF
    profile = dict(
        driver="GTiff", dtype="float32", nodata=-9999.0,
        width=W, height=H, count=1,
        crs="EPSG:4326", transform=transform, compress="lzw",
    )
    for sample_year in [1982, 2000, 2010, 2016]:
        if sample_year not in years_list:
            continue
        yi = years_list.index(sample_year)
        for var_name, arr in [("rzsm_gs", rzsm_gs), ("sm_shallow_gs", sm_sh_gs)]:
            a = arr[yi].copy()
            a[np.isnan(a)] = -9999.0
            f = OUT_DIR / f"{var_name}_{sample_year}.tif"
            with rasterio.open(f, "w", **profile) as dst:
                dst.write(a, 1)
            print(f"     存 {f.name}")

    # ============================ Checkpoint 3 验证
    print("\n" + "=" * 70)
    print("Checkpoint 3: 2010 RZSM / SM_shallow 标志点位")
    print("=" * 70)
    yi_2010 = years_list.index(2010)
    points = [
        ("广州 (华南)",          113.27, 23.13, "湿润, 期望 ~0.41"),
        ("成都 (西南)",          104.07, 30.67, "湿润, 期望 ~0.32"),
        ("拉萨 (青藏东缘)",      91.13,  29.65, "偏湿, 期望 ~0.31"),
        ("哈尔滨 (东北)",        126.65, 45.75, "偏湿, 期望 ~0.31"),
        ("石家庄 (华北平原)",    114.50, 38.05, "半湿润偏干, 期望 ~0.20"),
        ("柴达木",               95.0,   37.0,  "干旱, 期望 <0.10"),
        ("塔克拉玛干 (沙漠核心)", 83.0,   39.0,  "极旱, 期望接近 0"),
    ]
    print(f"\n{'点位':22s}  {'lon':>7s} {'lat':>6s}     {'RZSM':>6s} {'SM_sh':>6s}   {'zone':>6s}  {'参考':s}")
    for name, plon, plat, expect in points:
        yi_lat = int(np.argmin(np.abs(lat_arr - plat)))
        xi_lon = int(np.argmin(np.abs(lon_arr - plon)))
        r = float(rzsm_gs[yi_2010, yi_lat, xi_lon])
        s = float(sm_sh_gs[yi_2010, yi_lat, xi_lon])
        z = int(zone_raster[yi_lat, xi_lon])
        r_s = "NaN  " if np.isnan(r) else f"{r:.3f}"
        s_s = "NaN  " if np.isnan(s) else f"{s:.3f}"
        z_s = f"OID {z}" if z > 0 else "zone外"
        print(f"{name:22s}  {plon:7.2f} {plat:6.2f}     {r_s} {s_s}   {z_s:>6s}  {expect}")

    # 全国分位数 (zone 内, 2010 年)
    valid_2010 = rzsm_gs[yi_2010][~np.isnan(rzsm_gs[yi_2010])]
    print(f"\n2010 RZSM zone 内分位数 (期望 P5≈0.06, P50≈0.32, P95≈0.46):")
    for q in [5, 25, 50, 75, 95]:
        print(f"  P{q:2d} = {np.percentile(valid_2010, q):.3f}")

    print(f"\n[OK] C 步完成, 总耗时 {(time.time()-t_global)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
