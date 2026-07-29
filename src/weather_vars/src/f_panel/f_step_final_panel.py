"""
F 步: 合并 E 步 6 个 parquet → 最终县级气候面板 (per 方案 §10)

输入 (output/):
  county_year_gdd.parquet          PAC × year × gdd_growing_season
  county_year_rzsm.parquet         PAC × year × rzsm_gs
  county_year_sm_shallow.parquet   PAC × year × sm_shallow_gs
  county_year_irrigation.parquet   PAC × year × irr_gs                  (G 步新增)
  county_year_cropland.parquet     PAC × year × cropland_area_km2
  county_zone_info.parquet         PAC × zone37_main × dominant_share × is_cross_zone

加上县基础信息 (00 county/县.shp):
  NAME, 省, 类型

输出:
  output/climate_panel_1982_2016.parquet  最终交付的县级气候面板 (v1)

字段 (per 方案 §10.1):
  PAC                  int     县级行政区划代码 (唯一键)
  year                 int     年份 1982-2016
  NAME                 str     县名
  province             str     省
  county_type          str     县/市辖区/县级市 等
  zone37_main          int     主导农业熟制区划 OBJECTID_1
  dominant_share       float   主导区划耕地占比
  cross_zone_flag      bool    主导区划占比 < 0.9
  gdd_growing_season   float   B 步生长季 GDD (day·°C)
  rzsm_gs              float   C 步生长季根区土壤湿度 (m³/m³, 0-100cm)
  sm_shallow_gs        float   C 步生长季浅层土壤湿度 (m³/m³, 0-28cm)
  irr_gs               float   G 步生长季灌溉用水量 (10^8 m³, 县级 sum)
  irr_annual           float   G 步全年灌溉用水量 (10^8 m³, 县级 sum, 同 zone 内空间范围)
  cropland_area_km2    float   该县当年耕地面积 (km²)
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA_ROOT, OUTPUT

XIAN_SHP = DATA_ROOT / "00 county" / "县.shp"


def main():
    print("[F 步] 合并县级气候面板")

    # E 步产出
    df_gdd      = pd.read_parquet(OUTPUT / "county_year_gdd.parquet")
    df_rzsm     = pd.read_parquet(OUTPUT / "county_year_rzsm.parquet")
    df_sm       = pd.read_parquet(OUTPUT / "county_year_sm_shallow.parquet")
    df_irr      = pd.read_parquet(OUTPUT / "county_year_irrigation.parquet")
    df_cropland = pd.read_parquet(OUTPUT / "county_year_cropland.parquet")
    df_zone     = pd.read_parquet(OUTPUT / "county_zone_info.parquet")

    print(f"  GDD       : {len(df_gdd):,d}")
    print(f"  RZSM      : {len(df_rzsm):,d}")
    print(f"  SM_shallow: {len(df_sm):,d}")
    print(f"  Irrigation: {len(df_irr):,d}")
    print(f"  Cropland  : {len(df_cropland):,d}")
    print(f"  Zone info : {len(df_zone):,d}")

    # 县基础信息
    xian = gpd.read_file(XIAN_SHP)
    xian = xian[~xian.PAC.isin([156408, 710024])]
    df_xian = xian[["PAC", "NAME", "省", "类型"]].copy().rename(
        columns={"省": "province", "类型": "county_type"}
    )

    # 合并
    panel = (df_gdd
        .merge(df_rzsm,     on=["PAC", "year"], how="outer")
        .merge(df_sm,       on=["PAC", "year"], how="outer")
        .merge(df_irr,      on=["PAC", "year"], how="outer")
        .merge(df_cropland, on=["PAC", "year"], how="outer")
        .merge(df_zone,     on="PAC",           how="left")
        .merge(df_xian,     on="PAC",           how="left")
    )

    # 列顺序
    cols = [
        "PAC", "year", "NAME", "province", "county_type",
        "zone37_main", "dominant_share", "is_cross_zone",
        "gdd_growing_season", "rzsm_gs", "sm_shallow_gs",
        "irr_gs", "irr_annual",
        "cropland_area_km2",
    ]
    panel = panel[cols].sort_values(["PAC", "year"]).reset_index(drop=True)

    # 写 parquet
    out_f = OUTPUT / "climate_panel_1982_2016.parquet"
    panel.to_parquet(out_f, index=False)
    print(f"\n[OK] 最终面板写入 {out_f.name} ({out_f.stat().st_size/1024:.0f} KB)")

    # 健全性
    print(f"\n=== 最终面板规模 ===")
    print(f"  总行数     : {len(panel):,d}")
    print(f"  PAC 唯一   : {panel.PAC.nunique():,d}")
    print(f"  年份覆盖   : {sorted(panel.year.unique())[0]}-{sorted(panel.year.unique())[-1]}")
    print(f"  每 PAC 平均年数: {len(panel)/panel.PAC.nunique():.1f}")

    print(f"\n=== 关键列 NaN 比例 ===")
    for c in ["gdd_growing_season", "rzsm_gs", "sm_shallow_gs",
              "irr_gs", "irr_annual",
              "cropland_area_km2", "zone37_main", "NAME"]:
        n_nan = int(panel[c].isna().sum())
        print(f"  {c:25s}  NaN {n_nan:>6,d} ({100*n_nan/len(panel):.1f}%)")

    print(f"\n=== 抽样 (前 5 行) ===")
    print(panel.head().to_string(index=False))

    print(f"\n=== 跨区划县统计 ===")
    n_cz = int((panel.drop_duplicates("PAC")["is_cross_zone"] == True).sum())
    print(f"  跨区划县数: {n_cz}/{panel.PAC.nunique()}")


if __name__ == "__main__":
    main()
