"""
预处理 - 坑 3 验证: NoData 屏蔽逻辑

方案 §4.3 致命陷阱: 气温/灌溉 NoData = -3.4e38 不会被 xarray 自动当 NaN。
本脚本对 4 个数据源各抽 1 个文件，对比"未屏蔽" vs "正确屏蔽"后的统计值。
"""
from pathlib import Path
import numpy as np
import rasterio
import xarray as xr

DATA = Path("/Users/zhangxinzhen/Desktop/TFP/Climate data")


def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


# ---------------------------------------------------------------- 01 气温
banner("01 气温: 1980-01-01 max - NoData=-3.4e38 (float32 极小值)")
f = DATA / "01 temp_min_max/1980_max/19800101_max.tif"

with rasterio.open(f) as src:
    arr_raw = src.read(1)  # 不 mask
    nd = src.nodata
print(f"  src.nodata           : {nd}")
print(f"  arr.dtype            : {arr_raw.dtype}, shape: {arr_raw.shape}")
print(f"  未屏蔽 mean          : {arr_raw.mean():.4e}   ← 天文级负数, 错误")
print(f"  未屏蔽 min/max       : {arr_raw.min():.4e} / {arr_raw.max():.4e}")
# 推荐屏蔽: rasterio masked=True
with rasterio.open(f) as src:
    arr_m = src.read(1, masked=True)
print(f"  rasterio masked=True mean : {arr_m.mean():.2f} °C  (有效像元 {arr_m.count():,d}/{arr_m.size:,d})")
# 替代: np.where
arr_w = np.where(arr_raw > -1e37, arr_raw, np.nan)
print(f"  np.where>-1e37 nanmean    : {np.nanmean(arr_w):.2f} °C")

# ---------------------------------------------------------------- 02 SWI
banner("02 SWI: 1979 - NoData=NaN (已 CF mask)")
ds = xr.open_dataset(DATA / "02 soil/data_version-4.0_consolidated_1979.nc")
v = ds["swir"].isel(valid_time=0)
arr = v.values
n_total = arr.size
n_nan = int(np.isnan(arr).sum())
print(f"  直接 mean (xarray)   : {float(v.mean(skipna=True)):.4f}")
print(f"  shape                : {arr.shape}, NaN 像元 {n_nan:,d}/{n_total:,d} ({n_nan/n_total:.1%})")
print(f"  value range (skip nan): [{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
print(f"  结论                 : 不需要额外屏蔽")
ds.close()

# ---------------------------------------------------------------- 03 LUCC
banner("03 LUCC: 1980 - NoData=128 (int16, 整数哨兵)")
f = DATA / "03 plantation/1980/1980.tif"
with rasterio.open(f) as src:
    arr = src.read(1)
    nd = src.nodata
print(f"  src.nodata           : {nd}")
print(f"  arr.dtype            : {arr.dtype}, shape: {arr.shape}")
unique, counts = np.unique(arr, return_counts=True)
total = arr.size
print(f"  unique 值分布 (前 5):")
for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
    pct = 100 * c / total
    flag = " ← NoData" if u == nd else ""
    print(f"    code={u:4d}  cnt={c:>11,d}  ({pct:5.2f}%){flag}")
arr_m = np.where(arr == nd, np.nan, arr.astype("float32"))
print(f"  屏蔽前 mean (含 128) : {arr.mean():.2f}")
print(f"  屏蔽后 nanmean       : {np.nanmean(arr_m):.2f}  (LUCC 编码均值，参考)")
print(f"  有效像元             : {(~np.isnan(arr_m)).sum():,d}/{total:,d} ({(~np.isnan(arr_m)).sum()/total:.1%})")

# 2023 单独验证 (虽然研究期不用，但记录差异)
banner("03 LUCC: 2023 - NoData=-128 (int8, 类型与 1980-2020 不一致)")
f = DATA / "03 plantation/2023/2023_int.tif"
with rasterio.open(f) as src:
    arr = src.read(1)
    nd = src.nodata
print(f"  文件名               : 2023_int.tif (非 2023.tif)")
print(f"  src.nodata           : {nd}")
print(f"  arr.dtype            : {arr.dtype}, shape: {arr.shape}")
print(f"  研究期 1982-2016 不涉及 2023, 但加载脚本需 if year==2023 分支")

# ---------------------------------------------------------------- 04 NDVI
banner("04 NDVI: 1982 - NoData=NaN (已 mask)")
ds = xr.open_dataset(DATA / "04 NDVI_China/Daily_Gap-filled_NDVI_1982.nc4")
v = ds["NDVI"].isel(Time=0)
arr = v.values
n_total = arr.size
n_nan = int(np.isnan(arr).sum())
print(f"  shape                : {arr.shape}, NaN {n_nan:,d}/{n_total:,d} ({n_nan/n_total:.1%})")
print(f"  value range (skipna) : [{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
print(f"  mean                 : {float(v.mean(skipna=True)):.4f}")
print(f"  结论                 : 不需要额外屏蔽")
ds.close()

# ---------------------------------------------------------------- 06 灌溉
banner("06 灌溉 HSWUD: NoData=-3.403e38, 未走 CF mask (致命陷阱)")
ds = xr.open_dataset(DATA / "06 HSWUD_irr_.nc")
v = ds["irr"].isel(time=0)
arr = v.values
print(f"  shape                : {arr.shape}, dtype: {arr.dtype}")
print(f"  min/max              : {arr.min():.4e} / {arr.max():.4e}")
print(f"  未屏蔽 mean          : {arr.mean():.4e}   ← 天文级负数, 错误")
arr_m = np.where(arr > -1e37, arr, np.nan)
print(f"  np.where>-1e37 nanmean: {np.nanmean(arr_m):.4f} 亿 m³/月/像元 (合理)")
print(f"  有效像元             : {(~np.isnan(arr_m)).sum():,d}/{arr.size:,d} ({(~np.isnan(arr_m)).sum()/arr.size:.1%})")
ds.close()

# ---------------------------------------------------------------- 推荐 helper
banner("推荐 helper (放进 src/utils.py 后面 B/C 步反复用)")
print("""
import numpy as np
import xarray as xr

def mask_float32_sentinel(da: xr.DataArray) -> xr.DataArray:
    \"\"\"屏蔽 -3.4e38 等 float32 极小值哨兵 (气温/灌溉)\"\"\"
    return da.where(da > -1e37)

def mask_int_nodata(arr: np.ndarray, nodata: int) -> np.ndarray:
    \"\"\"屏蔽整数 NoData (LUCC 128 / -128), 返回 float32 + NaN\"\"\"
    out = arr.astype('float32', copy=True)
    out[arr == nodata] = np.nan
    return out
""")
