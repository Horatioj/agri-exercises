"""
A8 稳健性检验: 仅水田 (LUCC=11) / 仅旱地 (LUCC=12) 两套单独阈值

对应原文 NCC 1_7_peak_bottom_ndvi.R 的"三套土地标准"做法。
本方案主分析 = 11+12, 已在 A7 产出。本脚本补两列:
  - peak_month_paddy    : 仅用 11 计算占比 + 50% 阈值
  - peak_month_dryland  : 仅用 12 计算占比 + 50% 阈值

追加到 intermediate/growing_season/zone37_growing_season.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE, lucc_path

GS_DIR = INTERMEDIATE / "growing_season"
SHP_37 = DATA_ROOT / "05 agricultural division/01 中国农业熟制区划quhua/quhua.shp"
BASELINE_LUCC_YEAR = 2000


def compute_share_on_ndvi_grid(target_codes, lucc_arr, lucc_nd,
                                lucc_transform, lucc_crs,
                                ndvi_shape, ndvi_transform, ndvi_crs):
    """计算指定 LUCC 编码集合 的耕地占比, 重采样到 NDVI 网格."""
    safe = np.where(lucc_arr == lucc_nd, 0, lucc_arr)
    bin_arr = np.isin(safe, target_codes).astype("float32")
    out = np.zeros(ndvi_shape, dtype="float32")
    reproject(
        source=bin_arr, destination=out,
        src_transform=lucc_transform, src_crs=lucc_crs,
        dst_transform=ndvi_transform, dst_crs=ndvi_crs,
        resampling=Resampling.average,
    )
    return out, int(bin_arr.sum())


def per_zone_mode(flat_zone, flat_peak, flat_bot, flat_share):
    """对每个 OID 区划做 50% 阈值 + 简单众数."""
    result = {}
    for oid in np.unique(flat_zone):
        if oid == 0:
            continue
        sel = (flat_zone == oid)
        z_peak = flat_peak[sel]
        z_bot = flat_bot[sel]
        z_share = flat_share[sel]
        if z_share.size == 0:
            result[int(oid)] = (None, None, None, 0)
            continue
        thresh = float(np.percentile(z_share, 50))
        keep = z_share >= thresh
        if keep.sum() == 0:
            result[int(oid)] = (None, None, thresh, 0)
        else:
            pk = int(pd.Series(z_peak[keep]).mode().iloc[0])
            bt = int(pd.Series(z_bot[keep]).mode().iloc[0])
            result[int(oid)] = (pk, bt, thresh, int(keep.sum()))
    return result


def main():
    # 读已有产物
    csv_f = GS_DIR / "zone37_growing_season.csv"
    df = pd.read_csv(csv_f)
    print(f"读入主分析 CSV: {len(df)} 行")

    # NDVI 网格 / peak / bottom
    with rasterio.open(GS_DIR / "peak_month.tif") as src:
        peak_m = src.read(1)
        ndvi_transform = src.transform
        ndvi_crs = src.crs
        ndvi_shape = src.shape
    with rasterio.open(GS_DIR / "bottom_month.tif") as src:
        bot_m = src.read(1)

    # zone raster
    with rasterio.open(GS_DIR / "zone_raster_ndvi_grid.tif") as src:
        zone_raster = src.read(1)

    # LUCC 2000
    with rasterio.open(lucc_path(BASELINE_LUCC_YEAR)) as src:
        lucc = src.read(1)
        lucc_nd = src.nodata
        lucc_transform = src.transform
        lucc_crs = src.crs

    # 拉平
    valid_mask = (peak_m > 0) & (zone_raster > 0)
    flat_zone = zone_raster[valid_mask]
    flat_peak = peak_m[valid_mask]
    flat_bot = bot_m[valid_mask]

    print("\n[A8-paddy] 仅水田 (LUCC code 11)")
    paddy_share, n_paddy = compute_share_on_ndvi_grid(
        [11], lucc, lucc_nd, lucc_transform, lucc_crs,
        ndvi_shape, ndvi_transform, ndvi_crs,
    )
    print(f"  全国 LUCC 水田像元: {n_paddy:,d}")
    flat_paddy = paddy_share[valid_mask]
    res_paddy = per_zone_mode(flat_zone, flat_peak, flat_bot, flat_paddy)

    print("\n[A8-dryland] 仅旱地 (LUCC code 12)")
    dry_share, n_dry = compute_share_on_ndvi_grid(
        [12], lucc, lucc_nd, lucc_transform, lucc_crs,
        ndvi_shape, ndvi_transform, ndvi_crs,
    )
    print(f"  全国 LUCC 旱地像元: {n_dry:,d}")
    flat_dry = dry_share[valid_mask]
    res_dry = per_zone_mode(flat_zone, flat_peak, flat_bot, flat_dry)

    # 追加到 df
    # 注: 阈值 0.0 是合法值 (区划内 >50% 像元是非耕地, 中位数自然为 0),
    # 不能用 `if x[2] else None` (Codex 复审指出, 0.0 是 falsy 会被误判为 None)
    df["peak_month_paddy"] = df["OBJECTID_1"].map(lambda x: res_paddy[x][0])
    df["bottom_month_paddy"] = df["OBJECTID_1"].map(lambda x: res_paddy[x][1])
    df["paddy_share_p50"] = df["OBJECTID_1"].map(lambda x: round(res_paddy[x][2], 4) if res_paddy[x][2] is not None else None)
    df["peak_month_dryland"] = df["OBJECTID_1"].map(lambda x: res_dry[x][0])
    df["bottom_month_dryland"] = df["OBJECTID_1"].map(lambda x: res_dry[x][1])
    df["dryland_share_p50"] = df["OBJECTID_1"].map(lambda x: round(res_dry[x][2], 4) if res_dry[x][2] is not None else None)

    df.to_csv(csv_f, index=False)
    print(f"\n[OK] 追加 6 列写入 {csv_f.name}")

    # 对比分析
    print("\n========== 4 套 peak month 对比 (主分析 / 不筛选 / 仅水田 / 仅旱地) ==========")
    cmp = df[["OBJECTID_1", "NAME",
              "peak_month_main", "peak_month_unfiltered",
              "peak_month_paddy", "peak_month_dryland"]].copy()
    cmp["NAME"] = cmp["NAME"].astype(str).str.slice(0, 22)
    print(cmp.to_string(index=False))

    print("\n========== 主分析 与 其他三套 peak 差异统计 ==========")
    for col in ["peak_month_unfiltered", "peak_month_paddy", "peak_month_dryland"]:
        m = df["peak_month_main"]
        o = df[col]
        # 仅在两边都非空的行比较
        valid = m.notna() & o.notna()
        n_diff = int((m[valid] != o[valid]).sum())
        n_total = int(valid.sum())
        print(f"  vs {col:25s}  差异: {n_diff}/{n_total}  ({100*n_diff/n_total:.1f}%)")

    # 找出 4 套都给出相同 peak 的"超稳健"区划
    df_full = df.dropna(subset=["peak_month_main", "peak_month_unfiltered",
                                  "peak_month_paddy", "peak_month_dryland"])
    same = ((df_full["peak_month_main"] == df_full["peak_month_unfiltered"]) &
            (df_full["peak_month_main"] == df_full["peak_month_paddy"]) &
            (df_full["peak_month_main"] == df_full["peak_month_dryland"]))
    print(f"\n  四套完全一致的区划: {int(same.sum())}/{len(df_full)}")


if __name__ == "__main__":
    main()
