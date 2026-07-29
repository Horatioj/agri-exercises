"""
A3+A4+A5: 半月气候态 + 跨年拼接 7 期滑动平均 + 找 peak/bottom

输入: intermediate/growing_season/ndvi_halfmonth_1982_2016.nc
      形状 (year=35, halfmonth=24, Lat=708, Lon=1233)

输出:
  intermediate/growing_season/ndvi_climatology_raw.nc     (24, Lat, Lon)
  intermediate/growing_season/ndvi_climatology_smooth.nc  (24, Lat, Lon)
  intermediate/growing_season/peak_halfmonth.tif          (Lat, Lon) int8 1..24
  intermediate/growing_season/peak_month.tif              (Lat, Lon) int8 1..12
  intermediate/growing_season/bottom_halfmonth.tif        (Lat, Lon) int8 1..24
  intermediate/growing_season/bottom_month.tif            (Lat, Lon) int8 1..12

方法对齐 Ortiz-Bobea 2021 (1_7_peak_bottom_ndvi.R):
- A3: 35 年同半月取均值 → 每像元 24 期气候态曲线
- A4: stack 22-24 + 1-24 + 1-3 = 30 期; 7 期中心滑动平均; 取中间 24 期
- A5: argmax/argmin 找 peak/bottom 半月; 月份 = ceil(halfmonth/2)
"""
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import INTERMEDIATE

OUT_DIR = INTERMEDIATE / "growing_season"
INPUT_F = OUT_DIR / "ndvi_halfmonth_1982_2016.nc"


def main():
    assert INPUT_F.exists(), f"缺少 A1+A2 产出: {INPUT_F}"
    print(f"[A3+A4+A5] 加载 {INPUT_F.name}")
    ds = xr.open_dataset(INPUT_F)
    hm = ds["NDVI"]  # (year=35, halfmonth=24, Lat=708, Lon=1233)
    print(f"           输入维度: {dict(hm.sizes)}")

    # ============================ A3: 35 年同半月均值 → 气候态
    t0 = time.time()
    clim = hm.mean(dim="year", skipna=True).astype("float32")
    print(f"  A3 气候态  {time.time() - t0:.1f}s  shape={dict(clim.sizes)}")

    # 保存原始气候态
    out_raw = OUT_DIR / "ndvi_climatology_raw.nc"
    clim.to_dataset(name="NDVI").to_netcdf(
        out_raw, encoding={"NDVI": {"zlib": True, "complevel": 4}}
    )
    print(f"     存 {out_raw.name} ({out_raw.stat().st_size / 1024**2:.1f} MB)")

    # ============================ A4: 跨年边界拼接 + 7 期中心滑动平均
    # 对应 R 原码 stack(m[[22:24]], m, m[[1:3]]) → 30 期
    t0 = time.time()
    pad_head = clim.isel(halfmonth=slice(21, 24))  # 22, 23, 24 → 索引 21,22,23
    pad_tail = clim.isel(halfmonth=slice(0, 3))    # 1, 2, 3
    clim_padded = xr.concat([pad_head, clim, pad_tail], dim="halfmonth")
    # 重新赋 halfmonth 坐标 (30 期)
    clim_padded = clim_padded.assign_coords(halfmonth=np.arange(30))

    # 7 期中心滑动平均
    clim_smooth = clim_padded.rolling(halfmonth=7, center=True, min_periods=1).mean()
    # 取回中间 24 期 (索引 3..26)
    clim_smooth = clim_smooth.isel(halfmonth=slice(3, 27))
    # 恢复原 halfmonth 编号 1..24
    clim_smooth = clim_smooth.assign_coords(halfmonth=np.arange(1, 25))
    print(f"  A4 拼接+滑动平均  {time.time() - t0:.1f}s  shape={dict(clim_smooth.sizes)}")

    out_smooth = OUT_DIR / "ndvi_climatology_smooth.nc"
    clim_smooth.to_dataset(name="NDVI").to_netcdf(
        out_smooth, encoding={"NDVI": {"zlib": True, "complevel": 4}}
    )
    print(f"     存 {out_smooth.name} ({out_smooth.stat().st_size / 1024**2:.1f} MB)")

    # ============================ A5: argmax/argmin
    t0 = time.time()
    arr = clim_smooth.values  # (24, Lat, Lon)
    # mask: 任何半月有 NaN 的像元都判为无效
    valid = ~np.isnan(arr).any(axis=0)  # (Lat, Lon)

    peak_idx = np.argmax(np.where(np.isnan(arr), -np.inf, arr), axis=0)  # 0..23
    bottom_idx = np.argmin(np.where(np.isnan(arr), np.inf, arr), axis=0)

    peak_hm = (peak_idx + 1).astype("int8")    # 1..24
    bottom_hm = (bottom_idx + 1).astype("int8")
    peak_mo = ((peak_hm + 1) // 2).astype("int8")     # halfmonth → month
    bottom_mo = ((bottom_hm + 1) // 2).astype("int8")

    # 无效像元置 0 (用作 NoData)
    peak_hm[~valid] = 0
    bottom_hm[~valid] = 0
    peak_mo[~valid] = 0
    bottom_mo[~valid] = 0

    n_valid = int(valid.sum())
    print(f"  A5 peak/bottom 计算  {time.time() - t0:.1f}s  有效像元 {n_valid:,d}/{valid.size:,d} ({100*n_valid/valid.size:.1f}%)")

    # ============================ 存 GeoTIFF (便于 QGIS 可视化检查)
    # 用 NDVI 网格 transform: lon 起点 73.4413, lat 起点 53.5620, res 0.05
    lat = clim_smooth.Lat.values
    lon = clim_smooth.Lon.values
    res_lat = lat[1] - lat[0]   # 负数 (北→南)
    res_lon = lon[1] - lon[0]   # 正数
    # 像元中心 vs 角落: GeoTIFF transform 用左上角
    # lon[0] 是第一个像元中心, 左上角 = lon[0] - 0.5*|res_lon|
    transform = from_origin(
        west=lon[0] - 0.5 * abs(res_lon),
        north=lat[0] - 0.5 * res_lat,  # res_lat 已是负值; -0.5*负=正
        xsize=abs(res_lon),
        ysize=abs(res_lat),
    )
    profile = dict(
        driver="GTiff",
        dtype="int8",
        nodata=0,
        width=arr.shape[2],
        height=arr.shape[1],
        count=1,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
    )
    for name, data in [
        ("peak_halfmonth", peak_hm),
        ("peak_month", peak_mo),
        ("bottom_halfmonth", bottom_hm),
        ("bottom_month", bottom_mo),
    ]:
        f = OUT_DIR / f"{name}.tif"
        with rasterio.open(f, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"     存 {name}.tif ({f.stat().st_size / 1024:.0f} KB)")

    # ============================ 简单分布统计 (健全性检查)
    print(f"\n  peak_month 分布 (有效像元):")
    unique, counts = np.unique(peak_mo[valid], return_counts=True)
    for u, c in zip(unique, counts):
        bar = "█" * int(60 * c / counts.max())
        print(f"    {int(u):2d} 月  {c:>8,d}  {bar}")

    print(f"\n  bottom_month 分布 (有效像元):")
    unique, counts = np.unique(bottom_mo[valid], return_counts=True)
    for u, c in zip(unique, counts):
        bar = "█" * int(60 * c / counts.max())
        print(f"    {int(u):2d} 月  {c:>8,d}  {bar}")

    ds.close()
    print(f"\n[OK] A3+A4+A5 完成")


if __name__ == "__main__":
    main()
