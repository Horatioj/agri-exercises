"""
Codex 复审后的 panel 修复 (F 步后的补丁)

当前职责 (P1.2 已下沉到 E 步, 不再在此处理):
  P1.1: 加 valid_climate_sample 标记列
  P2:   加 cropland_area_smgrid_km2 列 (与 cropland_area_km2=tempgrid 并列)
  + 加入 G 步 irr_gs 列 (透传自 v1 panel)

P1.1: 加 valid_climate_sample 标记列
  现象: 137 PAC 有耕地但 GDD 全 NaN (主要是内蒙古东北/海南/台湾——这些地方不属于 37 农业熟制区划)
  根因: A 步的 37 区划本身不覆盖这些区域 → 像元层 GDD 是 NaN → 县级聚合是 NaN
  修复: 加 valid_climate_sample bool 列 (gdd/rzsm/sm_shallow 都非 NaN), 论文回归用这个筛样本

P1.2: 已下沉到 E 步 (2026-06)
  原现象: 130 个县有有效 GDD 但 zone37_main=NaN
  原根因: 旧 E2 用 rasterize() 到 0.1° 网格, 小县 (东城/西城等市辖区) 的几何中心不落在任何像元中心 → 不染色
  现处理: E 步 E2 已升级为 exact_extract(zone_raster, xian, ops=['unique','frac']),
          v1 panel 进来时 zone37_main / dominant_share / is_cross_zone 已经是正确版本.
          此处不再重算.

P2: 拆出 cropland_area_smgrid_km2 (与 cropland_area_km2=tempgrid 并列)
  现象: 51 行 cropland_area_km2=0 但 RZSM 有值, 涉及 4 个 PAC (红桥/静安/申扎/阿拉山口)
  根因: B/C 半像元错位, 小县在 tempgrid 上看不到耕地、但在 smgrid 上看到
  修复: 加一列 cropland_area_smgrid_km2, RZSM 样本筛选时用 smgrid 那列

输出:
  output/climate_panel_1982_2016_v2.parquet (覆盖旧版前先备份)
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from exactextract import exact_extract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, INTERMEDIATE, OUTPUT

XIAN_SHP    = DATA_ROOT / "00 county" / "县.shp"
ZONE_TIF    = INTERMEDIATE / "gdd_pixel" / "zone_raster_temp_grid.tif"
CW_DIR      = INTERMEDIATE / "cropland_weight"
MAPPING_CSV = CW_DIR / "year_to_lucc_mapping.csv"
PANEL_V1    = OUTPUT / "climate_panel_1982_2016.parquet"
PANEL_V2    = OUTPUT / "climate_panel_1982_2016_v2.parquet"
PANEL_BACKUP = OUTPUT / "climate_panel_1982_2016_v1_backup.parquet"


def compute_cropland_smgrid_per_year(xian: gpd.GeoDataFrame) -> pd.DataFrame:
    """用 SM 网格的 cropland_weight 重算县-年耕地面积."""
    print("  per slice exact_extract on smgrid...")
    mapping = pd.read_csv(MAPPING_CSV)
    slice_to_years = mapping.groupby("lucc_slice")["year"].apply(list).to_dict()
    rows = []
    for slc, years in slice_to_years.items():
        tif = CW_DIR / f"cropland_weight_{slc}_smgrid.tif"
        r = exact_extract(str(tif), xian, ops=["sum"], include_cols=["PAC"], output="pandas")
        r = r.rename(columns={"sum": "cropland_area_smgrid_km2"})
        for y in years:
            df_y = r.copy()
            df_y["year"] = y
            rows.append(df_y)
    return pd.concat(rows, ignore_index=True)[["PAC", "year", "cropland_area_smgrid_km2"]]


def main():
    t0 = time.time()

    # 备份 v1
    if not PANEL_BACKUP.exists():
        import shutil
        shutil.copy(PANEL_V1, PANEL_BACKUP)
        print(f"[backup] v1 → {PANEL_BACKUP.name}")
    else:
        print(f"[backup] {PANEL_BACKUP.name} 已存在, 跳过备份")

    # 加载 v1 + 县几何
    print("\n[1/3] 加载 v1 + 县几何")
    panel = pd.read_parquet(PANEL_V1)
    xian = gpd.read_file(XIAN_SHP)
    xian = xian[~xian.PAC.isin([156408, 710024])].to_crs(4326).reset_index(drop=True)
    print(f"  v1 panel: {len(panel):,d} 行, {panel.PAC.nunique()} PAC")
    print(f"  v1 zone37_main 有值的 PAC: {int(panel.drop_duplicates('PAC').zone37_main.notna().sum())}/{panel.PAC.nunique()}")
    print(f"  (P1.2 已在 E 步 E2 用 exact_extract 修复, 此处不再重算)")
    print(f"  县几何: {len(xian)} 个")

    # ============================ P2 修复: 加 smgrid 耕地面积列
    print("\n[2/3] [P2] 计算 cropland_area_smgrid_km2")
    df_crop_sm = compute_cropland_smgrid_per_year(xian)
    n_p2_old = int(((panel.cropland_area_km2 == 0) & panel.rzsm_gs.notna()).sum())
    print(f"  v1 中 cropland=0 但 RZSM 有值的行数: {n_p2_old}")

    # ============================ 合并 v2
    print("\n[3/3] 合并 + 加 valid_climate_sample + 写 v2")
    # zone 列从 v1 透传 (E 步已经写对); 只加 smgrid 耕地面积
    panel_v2 = panel.merge(df_crop_sm, on=["PAC", "year"], how="left")
    # 加 valid_climate_sample (P1.1)
    panel_v2["valid_climate_sample"] = (
        panel_v2["gdd_growing_season"].notna() &
        panel_v2["rzsm_gs"].notna() &
        panel_v2["sm_shallow_gs"].notna()
    )

    # 列顺序
    cols_v2 = [
        "PAC", "year", "NAME", "province", "county_type",
        "zone37_main", "dominant_share", "is_cross_zone",
        "gdd_growing_season", "rzsm_gs", "sm_shallow_gs",
        "irr_gs",                       # G 步生长季灌溉用水量 (10^8 m³, 县级 sum)
        "irr_annual",                   # G 步全年灌溉用水量 (10^8 m³, 同 zone 范围)
        "cropland_area_km2",            # tempgrid (B 步用)
        "cropland_area_smgrid_km2",     # smgrid (C 步用)
        "valid_climate_sample",         # 论文回归筛样本用 (注: 不含 irr_*, 灌溉单独看)
    ]
    panel_v2 = panel_v2[cols_v2].sort_values(["PAC", "year"]).reset_index(drop=True)

    panel_v2.to_parquet(PANEL_V2, index=False)
    sz = PANEL_V2.stat().st_size / 1024
    print(f"  写入 {PANEL_V2.name} ({sz:.0f} KB)")

    # 验证
    print("\n=== v1 → v2 关键变化 ===")
    n_pac = panel_v2.PAC.nunique()
    print(f"  总行数: {len(panel_v2):,d}  (期望 {n_pac}×35 = {n_pac*35:,d})")
    print(f"  PAC 唯一: {n_pac}")
    print()
    # zone 覆盖 (v1 透传, 不再 v1→v2 对比)
    n_zone_pac = int(panel_v2.drop_duplicates("PAC")["zone37_main"].notna().sum())
    n_cz_v2 = int(panel_v2.drop_duplicates("PAC")["is_cross_zone"].fillna(False).astype(bool).sum())
    print(f"  zone37_main 有值的 PAC : {n_zone_pac}/{n_pac} ({100*n_zone_pac/n_pac:.1f}%)  (E 步 exact_extract 直出)")
    print(f"  跨区划县 (is_cross_zone): {n_cz_v2} ({100*n_cz_v2/n_pac:.1f}%)")
    # smgrid 耕地
    n_p2_new = int(((panel_v2.cropland_area_smgrid_km2 == 0) & panel_v2.rzsm_gs.notna()).sum())
    n_p2_in_v2 = int(((panel_v2.cropland_area_km2 == 0) & panel_v2.rzsm_gs.notna()).sum())
    print(f"  P2 检验: tempgrid=0 但 RZSM 有值 = {n_p2_in_v2} (v1 中是 {n_p2_old}, 应该一致)")
    print(f"          smgrid=0 但 RZSM 有值   = {n_p2_new} (期望 0, 因为 RZSM 本来就是 smgrid 上算的)")
    # valid_climate_sample
    n_valid_sample = int(panel_v2.valid_climate_sample.sum())
    pac_with_valid = panel_v2[panel_v2.valid_climate_sample].PAC.nunique()
    print(f"  valid_climate_sample=True: {n_valid_sample:,d} 行 ({100*n_valid_sample/len(panel_v2):.1f}%)")
    print(f"  覆盖 PAC: {pac_with_valid}/{n_pac} ({100*pac_with_valid/n_pac:.1f}%)")

    # irrigation 覆盖 (G 步: irr_gs + irr_annual)
    for col in ["irr_gs", "irr_annual"]:
        n_nan = int(panel_v2[col].isna().sum())
        n_pac_have = panel_v2[panel_v2[col].notna()].PAC.nunique()
        total = float(panel_v2[col].sum(skipna=True))
        print(f"\n  {col} 列检查:")
        print(f"    NaN 行数: {n_nan:,d}/{len(panel_v2):,d} ({100*n_nan/len(panel_v2):.1f}%)")
        print(f"    覆盖 PAC: {n_pac_have}/{n_pac} ({100*n_pac_have/n_pac:.1f}%)")
        print(f"    总和    : {total:,.0f} × 10^8 m³ (全 35 年累计)")
    # gs / annual 比值
    gs_t = float(panel_v2['irr_gs'].sum(skipna=True))
    an_t = float(panel_v2['irr_annual'].sum(skipna=True))
    if an_t > 0:
        print(f"\n  生长季占全年比 (县级 sum 汇总): {100*gs_t/an_t:.1f}%")

    # 抽样: 之前用 rasterize 漏染色的小县, E 步升级后应该都有 zone
    print("\n=== 抽样: 历史上 rasterize 漏染色的小县, 现在应该 zone 非 NaN ===")
    test_pacs = [110101, 110102, 120101, 130108, 130202]
    for pac in test_pacs:
        sub = panel_v2[panel_v2.PAC == pac]
        if len(sub) == 0:
            print(f"  PAC {pac}: 不在 panel 中")
            continue
        row = sub.iloc[0]
        print(f"  PAC {pac} ({row.NAME}): zone={row.zone37_main}, share={row.dominant_share}, "
              f"irr_gs={row.irr_gs}, valid_climate={row.valid_climate_sample}")

    print(f"\n[OK] 修复完成, 耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
