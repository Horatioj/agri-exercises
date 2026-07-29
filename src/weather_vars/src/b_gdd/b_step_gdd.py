"""
B 步: 每像元每年生长季 GDD (Growing Degree Days)

方法 (方案 §6 + AJAE 2020):
- B1: 日均温 tmean = (tmax + tmin) / 2
- B2: AJAE 分段公式
    tmean < 8°C       → 0
    8 ≤ tmean ≤ 32°C  → tmean - 8
    tmean > 32°C      → 24
- B3: 37 区划栅格化到 0.1° 温度网格 (一次性, 不能复用 A 步 0.05° 网格)
- B4: 每像元按其区划的 green_season_months 集合累加月 GDD → 年值

输入:
  data/01 temp_min_max/YYYY_max/YYYYMMDD_max.tif (NoData=-3.4e38)
  data/01 temp_min_max/YYYY_min/YYYYMMDD_min.tif
  intermediate/growing_season/zone37_growing_season.csv (peak_month_main, green_season_months)
  data/05 agricultural division/01 .../quhua.shp

输出:
  intermediate/gdd_pixel/gdd_growing_season_1982_2016.nc  (year=35, lat, lon)
  intermediate/gdd_pixel/zone_raster_temp_grid.tif        (37 区划 in 0.1° 网格)
  intermediate/gdd_pixel/gdd_growing_season_{1982,2000,2016}.tif  QGIS 抽样可视化

参数:
  --year-range  默认 1982-2016
"""
import argparse
import sys
import time
from calendar import isleap
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE, parse_season_months

OUT_DIR = INTERMEDIATE / "gdd_pixel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = DATA_ROOT / "01 temp_min_max"
SHP_37 = DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp"
CSV_A = INTERMEDIATE / "growing_season" / "zone37_growing_season.csv"

BASE_T = 8.0   # AJAE 基温
CAP_T = 32.0  # AJAE 封顶
SENTINEL = -1e37   # tmax/tmin NoData 哨兵 (实际是 -3.4e38)


def ajae_gdd(tmean: np.ndarray) -> np.ndarray:
    """AJAE 2020 分段 GDD 公式。tmean 单位 °C, 返回 'day·°C'."""
    out = np.where(tmean < BASE_T, 0.0,
          np.where(tmean > CAP_T, CAP_T - BASE_T,
                   tmean - BASE_T))
    # NaN 传播
    out = np.where(np.isnan(tmean), np.nan, out)
    return out.astype("float32")


def get_temp_grid_info() -> dict:
    """从抽样 tif 拿温度网格元信息."""
    sample = TEMP_DIR / "1980_max" / "19800101_max.tif"
    with rasterio.open(sample) as src:
        return dict(
            transform=src.transform,
            crs=src.crs,
            shape=src.shape,
            width=src.width,
            height=src.height,
            bounds=src.bounds,
            nodata=src.nodata,
        )


def build_zone_raster(grid_info: dict) -> tuple:
    """37 区划栅格化到温度网格 (0.1° EPSG:4326)."""
    gdf = gpd.read_file(SHP_37).to_crs(4326)
    shapes = [(geom, int(oid)) for geom, oid in zip(gdf.geometry, gdf.OBJECTID_1)]
    zone_raster = rasterize(
        shapes,
        out_shape=grid_info["shape"],
        transform=grid_info["transform"],
        fill=0,
        dtype="int32",
    )
    return zone_raster, gdf


def load_season_lookup() -> dict:
    """zone_id (int) → set(months)."""
    df = pd.read_csv(CSV_A)
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r.peak_month_main):
            continue
        out[int(r.OBJECTID_1)] = parse_season_months(r.green_season_months)
    return out


def gdd_for_year(year: int, grid_info: dict, zone_raster: np.ndarray,
                 season_lookup: dict) -> np.ndarray:
    """单年 (H, W) GDD; 仅在 zone 内、生长季月份内累加."""
    H, W = grid_info["shape"]
    max_dir = TEMP_DIR / f"{year}_max"
    min_dir = TEMP_DIR / f"{year}_min"

    # (12, H, W) 月度 GDD 累加器, NaN 起始 → 用 0 初始化但跟踪每像元有效性
    monthly = np.zeros((12, H, W), dtype="float32")
    monthly_any_nan = np.zeros((12, H, W), dtype=bool)  # 该月该像元是否曾经出现 NaN

    n_days_year = 366 if isleap(year) else 365
    d = date(year, 1, 1)
    for _ in range(n_days_year):
        ymd = d.strftime("%Y%m%d")
        max_f = max_dir / f"{ymd}_max.tif"
        min_f = min_dir / f"{ymd}_min.tif"
        with rasterio.open(max_f) as src:
            tmax = src.read(1).astype("float32")
        with rasterio.open(min_f) as src:
            tmin = src.read(1).astype("float32")
        # 屏蔽 float32 哨兵 NoData
        tmax = np.where(tmax > SENTINEL, tmax, np.nan)
        tmin = np.where(tmin > SENTINEL, tmin, np.nan)
        tmean = (tmax + tmin) / 2.0
        gdd_day = ajae_gdd(tmean)

        m_idx = d.month - 1
        # 把 NaN 当 0 加进去, 但记录 NaN 位置
        nan_mask = np.isnan(gdd_day)
        gdd_day_safe = np.where(nan_mask, 0.0, gdd_day)
        monthly[m_idx] += gdd_day_safe
        monthly_any_nan[m_idx] |= nan_mask

        d += timedelta(days=1)

    # 严格 NaN 传播策略 (Codex 复审记录):
    #   某月任意一天 NaN  → 该像元该月为 NaN
    #   生长季任一月 NaN → 该像元该年为 NaN
    # 适用前提: 当前数据 35 年 × 每年 365/366 天 全部完整, n_valid=62,034 稳定不变.
    # 如未来换数据源出现零星缺测, 可改为 "至少有效 N 天就计算月均", 但需在论文方法里写权重逻辑.
    for m in range(12):
        monthly[m] = np.where(monthly_any_nan[m], np.nan, monthly[m])

    # 按 zone 把生长季月份累加成年值
    annual = np.full((H, W), np.nan, dtype="float32")
    for oid, months in season_lookup.items():
        zmask = zone_raster == oid
        if not zmask.any():
            continue
        # 求生长季月累加 (该 zone 的月份集)
        sums = np.zeros((H, W), dtype="float32")
        nan_in_season = np.zeros((H, W), dtype=bool)
        for m in months:
            sums += np.where(np.isnan(monthly[m - 1]), 0.0, monthly[m - 1])
            nan_in_season |= np.isnan(monthly[m - 1])
        # 该 zone 像元的 annual = sums, 但季节内有 NaN 月 → NaN
        annual = np.where(zmask & ~nan_in_season, sums, annual)
        annual = np.where(zmask & nan_in_season, np.nan, annual)

    return annual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-range", default="1982-2016", help="e.g. 1982-2016 or 1982-1982")
    args = ap.parse_args()
    y0, y1 = map(int, args.year_range.split("-"))
    years = list(range(y0, y1 + 1))

    print(f"[B 步] 年份 {y0}-{y1} ({len(years)} 年)")
    print(f"       温度源: {TEMP_DIR}")
    print(f"       输出  : {OUT_DIR}")

    # === Setup
    grid = get_temp_grid_info()
    print(f"\n[setup] 温度网格: shape={grid['shape']}, res=0.1°, bounds={tuple(round(x,2) for x in grid['bounds'])}, crs={grid['crs']}")

    zone_raster, gdf = build_zone_raster(grid)
    n_in_zone = int((zone_raster > 0).sum())
    print(f"[setup] 栅格化 37 区划: {n_in_zone:,d} / {zone_raster.size:,d} 像元 ({100*n_in_zone/zone_raster.size:.1f}%) 落入区划")

    # 存 zone_raster 供 QGIS 检查
    zone_out_f = OUT_DIR / "zone_raster_temp_grid.tif"
    with rasterio.open(
        zone_out_f, "w",
        driver="GTiff", dtype="int32", nodata=0,
        width=grid["width"], height=grid["height"], count=1,
        crs=grid["crs"], transform=grid["transform"], compress="lzw",
    ) as dst:
        dst.write(zone_raster, 1)
    print(f"        存 {zone_out_f.name}")

    season_lookup = load_season_lookup()
    print(f"[setup] season_lookup: {len(season_lookup)} 区划, 每个 5 个月")

    # 抽几个看
    for oid in [5, 13, 25]:
        if oid in season_lookup:
            print(f"        例 OID {oid}: 生长季月份 = {sorted(season_lookup[oid])}")

    # === 主循环
    print(f"\n[run] 开始逐年计算 GDD ...")
    t_global = time.time()
    stack = []
    for i, y in enumerate(years, 1):
        t0 = time.time()
        ann = gdd_for_year(y, grid, zone_raster, season_lookup)
        dt = time.time() - t0
        valid = ~np.isnan(ann)
        n_v = int(valid.sum())
        if n_v > 0:
            mn = float(np.nanmin(ann))
            md = float(np.nanmedian(ann))
            mx = float(np.nanmax(ann))
            mean = float(np.nanmean(ann))
        else:
            mn = md = mx = mean = float("nan")
        print(f"  [{i:2d}/{len(years)}] {y}  {dt:5.1f}s  n_valid={n_v:>6,d}  min={mn:>6.0f}  med={md:>6.0f}  mean={mean:>6.0f}  max={mx:>6.0f}")
        stack.append(ann)

    # === 保存 (year, lat, lon) NetCDF
    print(f"\n[save] 合并 {len(stack)} 年 → NetCDF")
    arr = np.stack(stack, axis=0)  # (Y, H, W)

    # 构造坐标 (从 transform 反推)
    tr = grid["transform"]
    H, W = grid["shape"]
    # tr * (col + 0.5, row + 0.5) → (lon, lat) 像元中心
    lons = np.array([tr * (c + 0.5, 0.5) for c in range(W)])[:, 0].astype("float32")
    lats = np.array([tr * (0.5, r + 0.5) for r in range(H)])[:, 1].astype("float32")

    ds_out = xr.Dataset(
        {"gdd_growing_season": (("year", "lat", "lon"), arr)},
        coords={"year": years, "lat": lats, "lon": lons},
        attrs={
            "method": "AJAE 2020 (Chambers, Pieralli & Sheng)",
            "base_temp_C": BASE_T,
            "cap_temp_C": CAP_T,
            "unit": "day_celsius (日·℃, 生长季内累加)",
            "growing_season_source": "zone37_growing_season.csv (per-zone 5 months)",
            "n_zones": len(season_lookup),
            "temp_grid_res_deg": 0.1,
        },
    )
    out_nc = OUT_DIR / f"gdd_growing_season_{y0}_{y1}.nc"
    if out_nc.exists():
        out_nc.unlink()
    ds_out.to_netcdf(out_nc, encoding={"gdd_growing_season": {"zlib": True, "complevel": 4}})
    print(f"       存 {out_nc.name} ({out_nc.stat().st_size/1024**2:.1f} MB)")

    # === 抽样年份存 GeoTIFF (QGIS 用)
    profile = dict(
        driver="GTiff", dtype="float32", nodata=-9999.0,
        width=W, height=H, count=1,
        crs=grid["crs"], transform=grid["transform"], compress="lzw",
    )
    samples = [s for s in [1982, 2000, 2016] if s in years]
    for s in samples:
        idx = years.index(s)
        a = stack[idx].copy()
        a[np.isnan(a)] = -9999.0
        f = OUT_DIR / f"gdd_growing_season_{s}.tif"
        with rasterio.open(f, "w", **profile) as dst:
            dst.write(a, 1)
        print(f"       存 {f.name}")

    print(f"\n[OK] B 步完成, 总耗时 {(time.time()-t_global)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
