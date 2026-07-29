"""
对比 NDVI 物理范围清洗前后, A 步结果的差异.
预期: 区划级 peak 完全一致, 像元级仅有极少改变 (Codex 反事实预测 2 个).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import INTERMEDIATE

GS = INTERMEDIATE / "growing_season"

print("=" * 70)
print("NDVI 物理范围清洗 (1991 异常值) — A 步结果对比")
print("=" * 70)

# === 区划级对比
before = pd.read_csv(GS / "zone37_growing_season_before_ndvi_clean.csv")
after = pd.read_csv(GS / "zone37_growing_season.csv")

print(f"\n[CSV] 行数 before={len(before)}  after={len(after)}")

# 比较关键列
key_cols = ["peak_month_main", "bottom_month_main",
            "peak_month_unfiltered", "bottom_month_unfiltered",
            "peak_month_paddy", "bottom_month_paddy",
            "peak_month_dryland", "bottom_month_dryland"]

print("\n=== 各列差异统计 (清洗后 != 清洗前) ===")
for col in key_cols:
    if col not in before.columns or col not in after.columns:
        continue
    diff_mask = before[col].fillna(-999) != after[col].fillna(-999)
    n_diff = int(diff_mask.sum())
    flag = "✅" if n_diff == 0 else f"⚠ {n_diff} 行"
    print(f"  {col:30s}  {flag}")
    if n_diff > 0:
        for oid in before[diff_mask].OBJECTID_1:
            b = before[before.OBJECTID_1 == oid][col].iloc[0]
            a = after[after.OBJECTID_1 == oid][col].iloc[0]
            name = str(before[before.OBJECTID_1 == oid].NAME.iloc[0])[:18]
            print(f"      OID {oid:2d} {name}: {b} → {a}")

# === 像元级 peak_month 对比
print("\n=== 像元级 peak_month.tif 对比 ===")
with rasterio.open("/tmp/peak_month_before_clean.tif") as src:
    pk_before = src.read(1)
with rasterio.open(GS / "peak_month.tif") as src:
    pk_after = src.read(1)

# 有效像元 (两边都非 0)
both_valid = (pk_before > 0) & (pk_after > 0)
diff_pixels = (pk_before != pk_after) & both_valid
n_diff_pix = int(diff_pixels.sum())
n_valid = int(both_valid.sum())

print(f"  共有效像元: {n_valid:,d}")
print(f"  peak month 改变的像元: {n_diff_pix:,d}  ({100*n_diff_pix/n_valid:.4f}%)")
if n_diff_pix > 0 and n_diff_pix <= 50:
    print(f"  差异像元 (前 20):")
    ys, xs = np.where(diff_pixels)
    for i in range(min(20, n_diff_pix)):
        print(f"    pixel({ys[i]},{xs[i]}): {pk_before[ys[i],xs[i]]} → {pk_after[ys[i],xs[i]]}")

# 也看看 有效像元数 是否变了
n_valid_before = int((pk_before > 0).sum())
n_valid_after = int((pk_after > 0).sum())
print(f"\n  有效像元数: before={n_valid_before:,d}  after={n_valid_after:,d}  diff={n_valid_after - n_valid_before:+,d}")
