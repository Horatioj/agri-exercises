"""
G 步: 06 灌溉 NetCDF → 像元层年灌溉用水量 (生长季 + 全年, 两个变量并列)

与 B/C 步并列, 作为面板的第三个气候/水利变量。
逻辑选择 (用户拍板):
  - 时间维度: 与 B/C 同款 (1982-2016, parse_season_months, 跨年季拼接, 严格 NaN 传播)
  - 季节聚合: 生长季内 sum (irr_gs) + 全 12 月 sum (irr_annual) 并列
              两者空间范围同 (zone>0 内), 仅月份集合不同, 便于同口径对比
  - 县级聚合: E 步直接 exact_extract sum, 不用耕地权重 (灌溉数据本身就是 0.1° 格的总用水量)

输入:
  /Users/zhangxinzhen/Desktop/TFP/Climate data/06 HSWUD_irr_.nc   (1965-01 → 2022-12, 696 月)
  intermediate/growing_season/zone37_growing_season.csv           (peak_month_main, green_season_months)
  data/05 agricultural division/01 中国农业熟制区划quhua/quhua.shp

输出 (intermediate/irr_pixel/):
  irr_growing_season_1982_2016.nc            (year=35, lat=355, lon=615; vars=irr_gs+irr_annual)
  zone_raster_irr_grid.tif                   (37 区划在 0.1° 灌溉网格)
  irr_growing_season_{1982,2000,2016}.tif    生长季抽样可视化
  irr_annual_{1982,2000,2016}.tif            全年抽样可视化

灌溉网格 (preflight + 核查):
  - lat: 53.5006 → 17.9994 (N→S, 355 点, 步长 ≈ 0.10029°, 不严格 0.1°)
  - lon: 73.4997 → 135.0003 (W→E, 615 点, 步长 ≈ 0.10016°)
  - 与 B 步温度网格 / C 步 SM 网格都不一样, 必须独立栅格化
  - transform 用文件实际 step 构造, 不强行规整成 0.1° (否则像元中心与数据值错位)

⚠ 关键 (utils.write_crs_if_missing 的副作用):
  open_irrigation(with_crs=True) 会把 dim 从 lat/lon rename 成 latitude/longitude
  → G 步必须用 with_crs=False 加载, 自己处理 CRS; 否则 ds.lat 取不到.

⚠ 关键 (E 步兼容性):
  G 步输出的 nc 维度必须命名为 ("year","lat","lon"),
  与 C 步 swvl_growing_season_*.nc 一致, 这样 E 步 prep_value_da
  (写死 x_dim="lon", y_dim="lat") 能直接复用.

跨年生长季约定 (与 B/C 步一致):
  OID 5 peak=11, green={9,10,11,12,1}, "1982 年值" = 1982 calendar year 内的
  Jan + Sep-Dec 5 个月的灌溉量累加.
  实际混了 1981/82 季尾巴 + 1982/83 季开头, 但与 B/C 同口径, 不可改.

NaN 传播策略 (严格, 与 B 步同款):
  生长季内任一月 NaN → 该像元该年 NaN.
  实现: np.sum 默认 NaN 传播, 不用 np.nansum.
  灌溉数据的 NaN 主要是空间掩膜外 (海洋/沙漠/水体), 这些格子全年所有月都是 NaN,
  严格传播 = 全年 NaN, 与"无数据"语义一致.
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
from utils import DATA_ROOT, INTERMEDIATE, open_irrigation, parse_season_months

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

SHP_37  = DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp"
CSV_A   = INTERMEDIATE / "growing_season" / "zone37_growing_season.csv"
OUT_DIR = INTERMEDIATE / "irr_pixel"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_global = time.time()

    # ============================ G1: 加载 + 切片 + 断言
    print("[G1] 加载 06 灌溉 NetCDF")
    # 关键: with_crs=False, 避免 write_crs_if_missing 把 lat/lon rename 成 latitude/longitude
    ds = open_irrigation(with_crs=False)
    print(f"     原始时间: {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]} ({len(ds.time)} 月)")
    ds = ds.sel(time=slice("1982-01-01", "2016-12-31"))
    n_months = len(ds.time)
    assert n_months == 420, f"期望 420 月 (35×12), 实得 {n_months}"
    print(f"     切片后 : {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]} ({n_months} 月 ✓)")
    print(f"     网格   : shape=({len(ds.lat)}, {len(ds.lon)})")
    print(f"     lat    : {float(ds.lat[0]):.4f} → {float(ds.lat[-1]):.4f} (N→S, step={float(ds.lat[0]-ds.lat[1]):.6f})")
    print(f"     lon    : {float(ds.lon[0]):.4f} → {float(ds.lon[-1]):.4f} (W→E, step={float(ds.lon[1]-ds.lon[0]):.6f})")

    # ============================ G2: 取 numpy + 时间/坐标索引
    print("\n[G2] 准备 numpy 数组 + 时间/坐标索引")
    irr_arr = ds["irr"].values.astype("float32")    # (420, H, W); open_irrigation 已 mask 哨兵
    H, W = irr_arr.shape[1], irr_arr.shape[2]
    times = pd.to_datetime(ds.time.values)
    months_idx = times.month.to_numpy()
    years_idx  = times.year.to_numpy()
    # 用文件实际坐标 (不规整成严格 0.1°), 保证 transform 中心 = 数据像元中心
    lat_arr = ds.lat.values.astype("float64")        # [53.5006, ..., 17.9994]
    lon_arr = ds.lon.values.astype("float64")
    print(f"     irr_arr shape: {irr_arr.shape}, dtype: {irr_arr.dtype}")
    print(f"     NaN 比例   : {np.isnan(irr_arr).mean()*100:.1f}% (空间掩膜外: 海洋/沙漠/水体)")
    print(f"     1990-07 全国 sum = {np.nansum(irr_arr[(years_idx==1990)&(months_idx==7)]):.1f} × 10^8 m³")
    ds.close()

    # ============================ G3: 区划栅格化到灌溉自有网格
    print("\n[G3] 栅格化 37 区划到灌溉 0.1° 网格 (按 OBJECTID_1)")
    dx = float(lon_arr[1] - lon_arr[0])             # ≈ +0.10016
    dy = float(lat_arr[0] - lat_arr[1])             # ≈ +0.10029 (lat N→S, 故 lat[0]-lat[1]>0)
    transform = from_origin(lon_arr[0]-dx/2, lat_arr[0]+dy/2, dx, dy)
    print(f"     transform: dx={dx:.6f}, dy={dy:.6f}")
    print(f"     西北角  : ({lon_arr[0]-dx/2:.4f}, {lat_arr[0]+dy/2:.4f})")
    print(f"     东南角  : ({lon_arr[-1]+dx/2:.4f}, {lat_arr[-1]-dy/2:.4f})")

    gdf = gpd.read_file(SHP_37)
    print(f"     原 CRS: {gdf.crs.name}")
    gdf = gdf.to_crs(4326)

    # !! 关键: 用 OBJECTID_1 (A/B/C 步同源, 连续 1-37), 不是 areanumber
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
    print(f"     出现 OID: {len(unique_oids)} 个 (期望 37)")
    if len(unique_oids) < 37:
        missing = set(range(1, 38)) - set(unique_oids)
        print(f"     ⚠ 缺失 OID: {sorted(missing)}")

    # 存盘 (供 QGIS / E 步可能使用)
    zone_tif = OUT_DIR / "zone_raster_irr_grid.tif"
    with rasterio.open(
        zone_tif, "w",
        driver="GTiff", dtype="int32", nodata=0,
        width=W, height=H, count=1,
        crs="EPSG:4326", transform=transform, compress="lzw",
    ) as dst:
        dst.write(zone_raster, 1)
    print(f"     存 {zone_tif.name}")

    # 生长季查找表 (与 B/C 步一致)
    df_zone = pd.read_csv(CSV_A)
    season_lookup = {
        int(r.OBJECTID_1): parse_season_months(r.green_season_months)
        for _, r in df_zone.iterrows() if pd.notna(r.peak_month_main)
    }
    print(f"     season_lookup: {len(season_lookup)} 区划")
    for oid in [5, 13, 25]:
        if oid in season_lookup:
            print(f"       例 OID {oid:2d}: 生长季月份 = {sorted(season_lookup[oid])}")

    # ============================ G4: 逐区划生长季 sum + 全年 sum (严格 NaN 传播)
    # 两个变量并列输出, 处理逻辑完全一致, 仅月份集合不同:
    #   irr_gs     : 该区划 green_season_months (通常 5 个月)
    #   irr_annual : 全 12 个月 (与 zone 内空间范围一致, 便于与 irr_gs 同口径对比)
    print("\n[G4] 逐区划生长季 sum + 全年 sum")
    print("     严格 NaN: 任一相关月 NaN → 该像元该年 NaN")
    print("     (np.sum 默认 NaN 传播, 不用 np.nansum)")
    years_list = list(range(1982, 2017))
    n_years = len(years_list)
    irr_gs     = np.full((n_years, H, W), np.nan, dtype="float32")
    irr_annual = np.full((n_years, H, W), np.nan, dtype="float32")

    t0 = time.time()
    skipped = 0
    for oid in range(1, 38):
        if oid not in season_lookup:
            skipped += 1
            continue
        months = season_lookup[oid]
        zmask = (zone_raster == oid)
        if not zmask.any():
            print(f"     ⚠ OID {oid} 在灌溉网格无像元落入, 跳过")
            continue

        in_season_global = np.isin(months_idx, list(months))

        for yi, y in enumerate(years_list):
            # --- 生长季 sum ---
            ym = in_season_global & (years_idx == y)
            n_m = int(ym.sum())
            # 跨年季 OID 5 (green={9,10,11,12,1}) 在 1982 也会有 5 个月: Jan + Sep-Dec
            # 与 B/C 同语义 (注释: 实际混了上一季尾巴+下一季开头), 不可改
            # !! 必须 hard fail: 若 n_m=0, slab.sum 会给出全 0, 把异常伪装成"无灌溉"
            if n_m != len(months):
                raise ValueError(
                    f"OID {oid} 在 {y} 实际命中 {n_m} 月, 期望 {len(months)} 月; "
                    "请检查 time/years_idx/months_idx 索引和 green_season_months 解析"
                )
            slab_gs = irr_arr[ym]                    # (n_m, H, W)
            # np.sum 默认 NaN 传播: 任一月 NaN → 该像元该年 NaN
            sum_gs = slab_gs.sum(axis=0)             # (H, W)
            irr_gs[yi] = np.where(zmask, sum_gs, irr_gs[yi])

            # --- 全年 sum (12 个月) ---
            ym_year = (years_idx == y)
            n_m_year = int(ym_year.sum())
            if n_m_year != 12:
                raise ValueError(
                    f"{y} 实际命中 {n_m_year} 月, 期望 12 月; 请检查 time/years_idx 索引"
                )
            slab_yr = irr_arr[ym_year]               # (12, H, W)
            sum_yr = slab_yr.sum(axis=0)             # (H, W); 任一月 NaN → NaN
            irr_annual[yi] = np.where(zmask, sum_yr, irr_annual[yi])

    print(f"     完成 {time.time()-t0:.1f}s (skipped {skipped} 个区划)")

    # n_valid 健全性: 遍历全部 35 年 (避免抽样漏掉中间年份的零星缺测)
    print("\n[validate] 每年 n_valid (应稳定不变 — 同一空间掩膜)")
    n_valid_per_year = {}
    for yi, y in enumerate(years_list):
        n_valid_per_year[y] = int(np.sum(~np.isnan(irr_gs[yi])))
    n_valid_set = set(n_valid_per_year.values())
    # 抽样打印 (避免 35 行刷屏)
    for y in [1982, 1990, 2000, 2010, 2016]:
        print(f"     {y}: n_valid = {n_valid_per_year[y]:,d}")
    if len(n_valid_set) > 1:
        # 报告异常年份
        common = max(n_valid_set, key=lambda v: sum(1 for vv in n_valid_per_year.values() if vv==v))
        odd_years = [(y, v) for y, v in n_valid_per_year.items() if v != common]
        print(f"     ⚠ n_valid 不稳定 (共 {len(n_valid_set)} 种取值): {sorted(n_valid_set)}")
        print(f"        主流取值 {common:,d}, 异常年份:")
        for y, v in odd_years:
            print(f"          {y}: n_valid = {v:,d} (Δ {v-common:+d})")
    else:
        print(f"     ✓ 全 35 年 n_valid = {n_valid_set.pop():,d} (完全稳定)")

    # ============================ 存 NetCDF
    print("\n[save] 写 NetCDF + 抽样 GeoTIFF")
    out_ds = xr.Dataset(
        {
            "irr_gs":     (("year", "lat", "lon"), irr_gs),
            "irr_annual": (("year", "lat", "lon"), irr_annual),
        },
        coords={
            "year": years_list,
            "lat": lat_arr.astype("float32"),
            "lon": lon_arr.astype("float32"),
        },
    )
    out_ds["irr_gs"].attrs = {
        "units": "10^8 m^3 / growing_season",
        "long_name": "Growing-season irrigation water use (sum)",
        "source": "06 HSWUD_irr_.nc + A-step zone37_growing_season.csv",
        "method": "Per-zone growing-season SUM over green_season_months; strict NaN propagation",
        "note": "Pixel value = total irrigation water use in this 0.1° pixel for the growing-season months",
    }
    out_ds["irr_annual"].attrs = {
        "units": "10^8 m^3 / year",
        "long_name": "Annual irrigation water use (calendar year sum)",
        "source": "06 HSWUD_irr_.nc",
        "method": "Per-pixel SUM over all 12 calendar months; strict NaN propagation",
        "note": "Same spatial range as irr_gs (zone>0 only) for same-cohort comparison",
    }
    out_ds["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    out_ds["lon"].attrs = {"units": "degrees_east", "long_name": "longitude"}

    out_nc = OUT_DIR / "irr_growing_season_1982_2016.nc"
    if out_nc.exists():
        out_nc.unlink()
    out_ds.to_netcdf(out_nc, encoding={
        "irr_gs":     {"zlib": True, "complevel": 4},
        "irr_annual": {"zlib": True, "complevel": 4},
    })
    print(f"     存 {out_nc.name} ({out_nc.stat().st_size/1024**2:.1f} MB)")

    # 抽样 GeoTIFF (QGIS 可视化) — 两个变量都存
    profile = dict(
        driver="GTiff", dtype="float32", nodata=-9999.0,
        width=W, height=H, count=1,
        crs="EPSG:4326", transform=transform, compress="lzw",
    )
    for sample_year in [1982, 2000, 2016]:
        if sample_year not in years_list:
            continue
        yi = years_list.index(sample_year)
        for var_name, arr in [("irr_growing_season", irr_gs), ("irr_annual", irr_annual)]:
            a = arr[yi].copy()
            a[np.isnan(a)] = -9999.0
            f = OUT_DIR / f"{var_name}_{sample_year}.tif"
            with rasterio.open(f, "w", **profile) as dst:
                dst.write(a, 1)
            print(f"     存 {f.name}")

    # ============================ Checkpoint: 标志点位 + 全国分位数
    print("\n" + "=" * 70)
    print("Checkpoint: 1990 年灌溉生长季 sum 在标志点位 (单位 10^8 m³ / 格 / 生长季)")
    print("=" * 70)
    yi_1990 = years_list.index(1990)
    points = [
        ("石家庄 (华北平原)",    114.50, 38.05, "灌溉密集区, 期望偏高 (0.3-1.5)"),
        ("郑州 (黄淮海)",        113.65, 34.75, "灌溉密集区, 期望偏高"),
        ("广州 (华南)",          113.27, 23.13, "雨养水稻为主, 期望低-中"),
        ("成都 (西南)",          104.07, 30.67, "灌溉中等"),
        ("拉萨 (青藏)",          91.13,  29.65, "高原雨养, 期望低"),
        ("阿克苏 (新疆绿洲)",    80.27,  41.17, "绿洲灌溉, 期望高"),
        ("哈尔滨 (东北)",        126.65, 45.75, "旱作为主, 灌溉中等"),
        ("塔克拉玛干 (沙漠核心)", 83.0,   39.0,  "无人区, 期望 NaN 或 0"),
    ]
    print(f"\n{'点位':22s}  {'lon':>7s} {'lat':>6s}    {'irr_gs':>8s} {'irr_annual':>10s}  {'gs/yr':>6s}   {'zone':>6s}  {'参考':s}")
    for name, plon, plat, expect in points:
        yi_lat = int(np.argmin(np.abs(lat_arr - plat)))
        xi_lon = int(np.argmin(np.abs(lon_arr - plon)))
        v_gs = float(irr_gs[yi_1990, yi_lat, xi_lon])
        v_an = float(irr_annual[yi_1990, yi_lat, xi_lon])
        z = int(zone_raster[yi_lat, xi_lon])
        v_gs_s = "NaN     " if np.isnan(v_gs) else f"{v_gs:8.4f}"
        v_an_s = "NaN       " if np.isnan(v_an) else f"{v_an:10.4f}"
        ratio = (v_gs/v_an) if (np.isfinite(v_gs) and np.isfinite(v_an) and v_an>0) else float('nan')
        ratio_s = " NaN  " if np.isnan(ratio) else f"{ratio:5.2%}"
        z_s = f"OID {z}" if z > 0 else "zone外"
        print(f"{name:22s}  {plon:7.2f} {plat:6.2f}    {v_gs_s} {v_an_s}  {ratio_s}   {z_s:>6s}  {expect}")

    # 全国分位数 (zone 内, 1990)
    valid_gs_1990 = irr_gs[yi_1990][~np.isnan(irr_gs[yi_1990])]
    valid_an_1990 = irr_annual[yi_1990][~np.isnan(irr_annual[yi_1990])]
    print(f"\n1990 zone 内 分位数 (10^8 m³ / 格):")
    print(f"  {'q':>5s}  {'irr_gs':>10s}  {'irr_annual':>12s}")
    for q in [5, 25, 50, 75, 95, 99]:
        print(f"  P{q:2d}    {np.percentile(valid_gs_1990, q):10.4f}  {np.percentile(valid_an_1990, q):12.4f}")
    sum_zone_gs = float(valid_gs_1990.sum())
    sum_zone_an = float(valid_an_1990.sum())
    print(f"\n  口径对照 (1990, 单位 10^8 m³):")
    print(f"    [本步 irr_gs]      zone 内 + 生长季 5 月 sum = {sum_zone_gs:7.1f}  ({sum_zone_gs/10:.0f} km³)")
    print(f"    [本步 irr_annual]  zone 内 + 全 12 月 sum    = {sum_zone_an:7.1f}  ({sum_zone_an/10:.0f} km³)")
    sum_all_pix_all_mon = float(np.nansum(irr_arr[years_idx == 1990]))
    print(f"    [原始全量]         全像元 + 全 12 月 sum     = {sum_all_pix_all_mon:7.1f}  ({sum_all_pix_all_mon/10:.0f} km³)")
    print(f"    [全国统计]         中国 1990 灌溉用水 (全年) ~ 3500       (350 km³, 仅供量级参考)")
    if sum_zone_an > 0:
        print(f"    → 生长季占全年比 (zone 内): {100*sum_zone_gs/sum_zone_an:.1f}%")

    # 35 年时序 (zone 内总和) — 双变量并列
    print(f"\n35 年时序 (zone 内总和, 10^8 m³ / 年):")
    annual_gs = []
    annual_yr = []
    for yi in range(n_years):
        annual_gs.append(float(np.nansum(irr_gs[yi][zone_raster > 0])))
        annual_yr.append(float(np.nansum(irr_annual[yi][zone_raster > 0])))
    peak_val = max(max(annual_gs), max(annual_yr))
    print(f"  {'year':>5s}  {'irr_gs':>8s}  {'irr_annual':>10s}  {'gs/yr':>6s}")
    for yi, y in enumerate(years_list):
        if y % 5 == 0 or y in (1982, 2016):
            ratio = annual_gs[yi]/annual_yr[yi] if annual_yr[yi] > 0 else float('nan')
            print(f"  {y:>5d}  {annual_gs[yi]:8.1f}  {annual_yr[yi]:10.1f}  {ratio:6.2%}")

    print(f"\n[OK] G 步完成, 总耗时 {(time.time()-t_global)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
