"""
TFP 气候变量构建 - 公共 helper
依据《工作方案》§4.3 + preflight 验证结果固化。
"""
from pathlib import Path
import numpy as np
import xarray as xr

DATA_ROOT = Path("/Users/zhangxinzhen/Desktop/TFP/Climate data")
CODE_ROOT = Path("/Users/zhangxinzhen/Desktop/TFP/Code")
INTERMEDIATE = CODE_ROOT / "intermediate"
OUTPUT = CODE_ROOT / "output"
LOGS = CODE_ROOT / "logs"


def mask_float32_sentinel(da: xr.DataArray, threshold: float = -1e37) -> xr.DataArray:
    """屏蔽 -3.4e38 等 float32 极小值哨兵 (01 气温, 06 灌溉)。

    01 气温的 NoData 不走 CF mask，xarray 直接 mean 会溢出到 -inf。
    06 灌溉同理 (《数据说明》§7.3)。
    """
    return da.where(da > threshold)


def mask_int_nodata(arr: np.ndarray, nodata) -> np.ndarray:
    """屏蔽整数 NoData (LUCC 128 / -128), 返回 float32 + NaN."""
    out = arr.astype("float32", copy=True)
    out[arr == nodata] = np.nan
    return out


def _detect_spatial_dims(dims):
    """返回 (x_dim, y_dim) 或 (None, None)."""
    dims = set(dims)
    if "longitude" in dims and "latitude" in dims:
        return "longitude", "latitude"
    if "Lon" in dims and "Lat" in dims:
        return "Lon", "Lat"
    if "lon" in dims and "lat" in dims:
        return "lon", "lat"
    if "x" in dims and "y" in dims:
        return "x", "y"
    return None, None


def write_crs_if_missing(da_or_ds, crs: str = "EPSG:4326"):
    """给隐式 CRS 的 NetCDF 数据 (02 SWI / 04 NDVI / 06 灌溉) 手动写 CRS。

    preflight 验证: 这三套 nc 文件不带 spatial_ref 变量, rioxarray 加载后
    必须先 write_crs 才能 reproject/clip。

    Codex 复审指出: 06 灌溉的 dim 是小写 lon/lat, rioxarray 不会自动识别,
    `set_spatial_dims` 在 Dataset 上不会传播到 ds[var] 单变量。
    解决: 把不识别的命名 rename 成 rioxarray 公认的 longitude/latitude,
    rename 会修改 dim 本身, ds[var] 取出后自动继承。
    """
    import rioxarray  # noqa
    x_dim, y_dim = _detect_spatial_dims(da_or_ds.dims)
    out = da_or_ds
    # rioxarray 自动识别 (longitude,latitude) / (Lon,Lat) / (x,y); lon/lat 不识别
    rename_map = {}
    if x_dim == "lon" and y_dim == "lat":
        rename_map = {"lon": "longitude", "lat": "latitude"}
    if rename_map:
        out = out.rename(rename_map)
        x_dim, y_dim = "longitude", "latitude"
    if x_dim is not None:
        try:
            out = out.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)
        except Exception:
            pass
    return out.rio.write_crs(crs, inplace=False)


def open_swi_year(year: int, with_crs: bool = True) -> xr.Dataset:
    """打开单个 SWI 文件并修正 valid_time 偏移 (preflight 验证 38 文件均偏 1 天)."""
    f = DATA_ROOT / "02 soil" / f"data_version-4.0_consolidated_{year}.nc"
    ds = xr.open_dataset(f, decode_times=True)
    ds = ds.assign_coords(valid_time=ds.valid_time - np.timedelta64(1, "D"))
    if with_crs:
        ds = write_crs_if_missing(ds)
    return ds


def open_swi_multi(years_slice: slice = None, with_crs: bool = True) -> xr.Dataset:
    """流式打开多年 SWI 并统一修正时间戳。

    years_slice 示例 slice("1982-01-01", "2016-12-31")。
    依据 §C1, chunk valid_time=30 平衡内存与吞吐。
    需要 dask >= 2024 (修复了 np.float 已弃用问题)。
    """
    pattern = str(DATA_ROOT / "02 soil/data_version-4.0_consolidated_*.nc")
    ds = xr.open_mfdataset(pattern, chunks={"valid_time": 30}, combine="by_coords")
    ds = ds.assign_coords(valid_time=ds.valid_time - np.timedelta64(1, "D"))
    if years_slice is not None:
        ds = ds.sel(valid_time=years_slice)
    if with_crs:
        ds = write_crs_if_missing(ds)
    return ds


def open_ndvi_year(year: int, with_crs: bool = True, clip_physical: bool = True) -> xr.Dataset:
    """打开单个 NDVI nc4 文件 (04 NDVI_China).

    clip_physical=True (默认): 把 NDVI 限制在物理合法范围 [-1, 1] 内, 超出值置 NaN.
    实测发现 1991 年原始数据存在 NDVI=3.072 等异常值 (491,980 像元时点),
    若不清洗会污染气候态. Codex 反事实测试: 清洗前后区划级 peak 完全一致.
    """
    f = DATA_ROOT / "04 NDVI_China" / f"Daily_Gap-filled_NDVI_{year}.nc4"
    ds = xr.open_dataset(f, engine="h5netcdf")
    if clip_physical:
        ds["NDVI"] = ds["NDVI"].where((ds["NDVI"] >= -1) & (ds["NDVI"] <= 1))
    if with_crs:
        ds = write_crs_if_missing(ds)
    return ds


def open_irrigation(with_crs: bool = True) -> xr.Dataset:
    """打开 06 灌溉单文件 NetCDF, 屏蔽 -3.4e38 哨兵 + write_crs."""
    f = DATA_ROOT / "06 HSWUD_irr_.nc"
    ds = xr.open_dataset(f)
    ds["irr"] = mask_float32_sentinel(ds["irr"])
    if with_crs:
        ds = write_crs_if_missing(ds)
    return ds


def parse_season_months(s) -> set:
    """把 zone37_growing_season.csv 的 green_season_months / brown_season_months
    字段解析为 set[int]。

    重要 (Codex line-by-line 复审): 字符串里的月份顺序可能是自然时间顺序 (9,10,11,12,1),
    不一定是字典序。B/C/D 步只能用 set/isin 做月份筛选, 绝不能把第一个当起始月、最后一个当结束月。
    """
    if not isinstance(s, str) or not s.strip():
        return set()
    return {int(x) for x in s.split(",") if x.strip()}


def lucc_path(year: int) -> Path:
    """LUCC 文件路径, 处理 2023 命名特殊。"""
    if year == 2023:
        return DATA_ROOT / "03 plantation" / "2023" / "2023_int.tif"
    return DATA_ROOT / "03 plantation" / f"{year}" / f"{year}.tif"


def load_cropland_binary(year: int) -> tuple:
    """加载 LUCC 切片并提取耕地二值栅格 (code 11 水田 + 12 旱地)。

    Returns (cropland_uint8, transform, crs).
    1980-2020 NoData=128 (int16), 2023 NoData=-128 (int8)，统一屏蔽。
    """
    import rasterio
    f = lucc_path(year)
    with rasterio.open(f) as src:
        arr = src.read(1)
        nd = src.nodata
        transform = src.transform
        crs = src.crs
    arr_safe = np.where(arr == nd, 0, arr)
    cropland = ((arr_safe == 11) | (arr_safe == 12)).astype("uint8")
    return cropland, transform, crs
