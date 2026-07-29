"""
预处理 - 坑 2 验证: SWI 时间戳偏移是否全年份一致

方案 §4.2 明确要求验证多个年份。
《数据说明》§3.4 称 1979 文件 valid_time 是 1979-01-02 → 1980-01-01 (偏 1 天)。
本脚本检查 1979-2016 全 38 个文件首尾 valid_time 和长度，断言:
  - 文件名 YYYY 的 valid_time[0] = YYYY-01-02
  - 文件名 YYYY 的 valid_time[-1] = (YYYY+1)-01-01
  - 长度 = 365 (平年) 或 366 (闰年)，按 YYYY 年判断
"""
import xarray as xr
import pandas as pd
from pathlib import Path

DATA = Path("/Users/zhangxinzhen/Desktop/TFP/Climate data/02 soil")


def is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


rows = []
files = sorted(DATA.glob("data_version-4.0_consolidated_*.nc"))
print(f"Total SWI files: {len(files)}\n")

for f in files:
    year = int(f.stem.split("_")[-1])
    ds = xr.open_dataset(f, decode_times=True)
    vt = pd.to_datetime(ds.valid_time.values)
    n = len(vt)
    t0, tN = vt[0], vt[-1]
    expected_t0 = pd.Timestamp(year=year, month=1, day=2)
    expected_tN = pd.Timestamp(year=year + 1, month=1, day=1)
    expected_n = 366 if is_leap(year) else 365
    t0_ok = (t0 == expected_t0)
    tN_ok = (tN == expected_tN)
    n_ok = (n == expected_n)
    rows.append({
        "year": year,
        "n_days": n,
        "first": t0.strftime("%Y-%m-%d"),
        "last": tN.strftime("%Y-%m-%d"),
        "first_ok": t0_ok,
        "last_ok": tN_ok,
        "n_ok": n_ok,
        "all_ok": t0_ok and tN_ok and n_ok,
    })
    ds.close()

df = pd.DataFrame(rows)
print(df.to_string(index=False))

print("\n" + "=" * 60)
total = len(df)
all_ok = df["all_ok"].sum()
print(f"通过率: {all_ok}/{total}")
if all_ok == total:
    print("✅ 全部 38 个年份均符合 'valid_time = YYYY-01-02 → (YYYY+1)-01-01' 模式")
    print("   修正方法: ds.assign_coords(valid_time = ds.valid_time - 1day)")
    print("   修正后: ds.valid_time 落在 YYYY-01-01 ~ YYYY-12-31 (整年)")
else:
    print("⚠ 部分年份不符合预期，需要逐文件单独处理：")
    print(df[~df["all_ok"]].to_string(index=False))

# 详细给出"修正后"的首尾示例 (针对方案中提到的 1980/1981/2000/2016)
print("\n" + "=" * 60)
print("方案 §4.2 指定的 4 个关键校验年份：1980, 1981, 2000, 2016")
target_years = [1980, 1981, 2000, 2016]
for y in target_years:
    row = df[df.year == y].iloc[0]
    fixed_first = (pd.Timestamp(row["first"]) - pd.Timedelta("1D")).strftime("%Y-%m-%d")
    fixed_last = (pd.Timestamp(row["last"]) - pd.Timedelta("1D")).strftime("%Y-%m-%d")
    print(f"  {y}: 原 {row['first']} → {row['last']} ({row['n_days']}d)  | "
          f"修正后 {fixed_first} → {fixed_last}")
