"""
A1+A2: NDVI 1982-2016 加载 + 日→半月聚合

按方案 §5.1 A2 节实现:
- 每月 1-15 日为上半月 (halfmonth 1, 3, 5, ..., 23)
- 每月 16 日至月末为下半月 (halfmonth 2, 4, ..., 24)
- 一年 24 期

输入: 04 NDVI_China/Daily_Gap-filled_NDVI_YYYY.nc4 (35 个文件)
输出: intermediate/growing_season/ndvi_halfmonth_1982_2016.nc
      形状 (year=35, halfmonth=24, Lat=708, Lon=1233)

参数:
  --year-range  e.g. "1982-1982" 单年烟雾测试; "1982-2016" 全量 (默认)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE

OUT_DIR = INTERMEDIATE / "growing_season"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def halfmonth_id(times: pd.DatetimeIndex) -> np.ndarray:
    """day 1-15 → 上半月; day 16+ → 下半月. 编码 1..24."""
    months = times.month.to_numpy()
    days = times.day.to_numpy()
    return (months - 1) * 2 + (days > 15).astype(int) + 1


def process_year(year: int) -> xr.DataArray:
    """单年 NDVI → (halfmonth=24, Lat, Lon) DataArray."""
    f = DATA_ROOT / "04 NDVI_China" / f"Daily_Gap-filled_NDVI_{year}.nc4"
    if not f.exists():
        raise FileNotFoundError(f)

    # 单年 NDVI 333-637 MB, 直接读进 RAM 即可 (16 GB 内存)
    ds = xr.open_dataset(f, engine="h5netcdf")
    # 物理范围清洗: NDVI 理论上 ∈ [-1, 1], 1991 原始数据存在 NDVI=3.072 等异常值
    # 实测仅 1991 受影响 (491,980 像元时点), 其他 34 年都干净
    ds["NDVI"] = ds["NDVI"].where((ds["NDVI"] >= -1) & (ds["NDVI"] <= 1))

    times = pd.to_datetime(ds.Time.values)
    n_days = len(times)
    hm = halfmonth_id(times)
    ds = ds.assign_coords(halfmonth=("Time", hm))

    # groupby halfmonth + 取均值 (skipna 保留 NaN 为 NaN)
    yh = ds["NDVI"].groupby("halfmonth").mean(dim="Time", skipna=True)
    yh = yh.astype("float32")  # 节省空间
    yh = yh.expand_dims(year=[year])

    result = yh.compute()
    ds.close()

    # sanity check
    assert result.sizes["halfmonth"] == 24, f"{year}: halfmonth={result.sizes['halfmonth']} (expected 24)"
    return result, n_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-range", default="1982-2016", help="e.g. 1982-2016 or 1982-1982")
    args = ap.parse_args()

    y0, y1 = map(int, args.year_range.split("-"))
    years = list(range(y0, y1 + 1))

    print(f"[A1+A2] 处理年份 {y0}-{y1} ({len(years)} 年)")
    print(f"        NDVI 源: {DATA_ROOT / '04 NDVI_China'}")
    print(f"        输出目录: {OUT_DIR}")
    print()

    t_global = time.time()
    parts = []
    for i, year in enumerate(years, 1):
        t0 = time.time()
        da, n_days = process_year(year)
        dt = time.time() - t0
        # 简单完整性: 1982-2016 各年应有 365/366 天
        from calendar import isleap
        expected = 366 if isleap(year) else 365
        flag = "✓" if n_days == expected else f"⚠ {n_days}d (期望 {expected})"
        nan_pct = float(np.isnan(da.values).mean()) * 100
        print(f"  [{i:2d}/{len(years)}] {year}  {dt:5.1f}s  ndays={n_days} {flag}  NaN={nan_pct:4.1f}%")
        parts.append(da)

    print(f"\n  合并 {len(parts)} 年...")
    all_hm = xr.concat(parts, dim="year")
    print(f"  最终维度: {dict(all_hm.sizes)}")
    print(f"  内存大小: {all_hm.nbytes / 1024**2:.1f} MB (float32)")

    # 保存为 Dataset 以便加压缩
    out_ds = all_hm.to_dataset(name="NDVI")
    out_f = OUT_DIR / f"ndvi_halfmonth_{y0}_{y1}.nc"
    if out_f.exists():
        out_f.unlink()
    out_ds.to_netcdf(
        out_f,
        encoding={"NDVI": {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    sz = out_f.stat().st_size / 1024**2
    print(f"\n[OK] 保存到 {out_f.name} ({sz:.1f} MB)")
    print(f"     总耗时 {(time.time() - t_global) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
