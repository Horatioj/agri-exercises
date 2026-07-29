"""
预处理 - 坑 1 验证: 各数据 CRS 实测

方案 §4.1: 矢量边界投到栅格 CRS 做 zonal stats，不要反过来。
本脚本打印 7 套数据的实际 CRS / 网格起点 / 分辨率，与《数据说明》第 8 节对照。
"""
from pathlib import Path
import rasterio
import geopandas as gpd
import xarray as xr
import numpy as np

DATA = Path("/Users/zhangxinzhen/Desktop/TFP/Climate data")


def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def show_raster(path, label):
    with rasterio.open(path) as src:
        print(f"[{label}] {path.name}")
        print(f"  CRS         : {src.crs}")
        print(f"  size        : {src.width} x {src.height}")
        print(f"  res         : {src.res}")
        print(f"  bounds      : {src.bounds}")
        print(f"  dtype       : {src.dtypes[0]}")
        print(f"  nodata      : {src.nodata}")


def show_vector(path, label):
    gdf = gpd.read_file(path, rows=1)
    print(f"[{label}] {path.name}")
    print(f"  CRS         : {gdf.crs}")
    print(f"  CRS WKT head: {str(gdf.crs.to_wkt())[:90]}...")
    full = gpd.read_file(path)
    print(f"  n_features  : {len(full)}")
    print(f"  bounds      : {tuple(round(x, 3) for x in full.total_bounds)}")


def show_nc(path, label, var_hint=None):
    ds = xr.open_dataset(path, decode_times=True)
    print(f"[{label}] {path.name}")
    print(f"  data_vars   : {list(ds.data_vars)}")
    print(f"  dims        : {dict(ds.sizes)}")
    coord_keys = [k for k in ds.coords if k in ('lon', 'longitude', 'Lon', 'lat', 'latitude', 'Lat')]
    for k in coord_keys:
        v = ds[k].values
        print(f"  {k:11s}: start={v[0]:.4f}  end={v[-1]:.4f}  step={(v[1] - v[0]):.4f}  n={len(v)}")
    ds.close()


banner("00 county - shapefile (期望 CGCS-2000 / EPSG:4490)")
for f in ["省.shp", "市.shp", "县.shp"]:
    show_vector(DATA / "00 county" / f, "county")
    print()

banner("01 temp_min_max - GeoTIFF 抽样 1980-01-01 (期望 WGS-84 / EPSG:4326, 0.1°)")
show_raster(DATA / "01 temp_min_max/1980_max/19800101_max.tif", "tmax")
show_raster(DATA / "01 temp_min_max/1980_min/19800101_min.tif", "tmin")

banner("02 soil - NetCDF 抽样 1979 (期望 WGS-84, 0.05°)")
show_nc(DATA / "02 soil/data_version-4.0_consolidated_1979.nc", "SWI")

banner("03 plantation - GeoTIFF 抽样 (期望 Krasovsky 1940 Albers, 1km)")
show_raster(DATA / "03 plantation/1980/1980.tif", "LUCC 1980")
show_raster(DATA / "03 plantation/2000/2000.tif", "LUCC 2000")

banner("04 NDVI_China - NetCDF4 抽样 1982 (期望 WGS-84 隐式, 0.05°)")
show_nc(DATA / "04 NDVI_China/Daily_Gap-filled_NDVI_1982.nc4", "NDVI")

banner("05 agricultural division - shapefile (期望 Krasovsky 1940 Albers)")
show_vector(DATA / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp", "37 zones")
print()
show_vector(DATA / "05 agricultural division/02 中国农业自然区划quhua/quhua-nongye.shp", "150 zones")

banner("06 HSWUD_irr_.nc - 灌溉 NetCDF (期望 WGS-84, 0.1°)")
show_nc(DATA / "06 HSWUD_irr_.nc", "irr")

banner("阵营汇总")
print("WGS-84 / EPSG:4326 阵营 : 01 气温, 02 SWI, 04 NDVI, 06 灌溉 (县 00 = EPSG:4490 视为同坐标)")
print("Krasovsky Albers 阵营   : 03 LUCC, 05 农业区划")
print("\n对接策略 (方案 §4.1):")
print("  - NDVI/SWI/气温 + 37 区划 → 把 37 区划 to_crs(4326)")
print("  - LUCC + 县界           → 把县界 to_crs(LUCC.crs) 在 Albers 做")
print("  - 最终聚合气候到县       → 县界 in 4326, 直接和栅格叠")
