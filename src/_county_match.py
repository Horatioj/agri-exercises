# -*- coding: utf-8 -*-
"""
Layered county matcher: production `countyid` -> 2010 census-boundary code.

Production codes/names and the 2010 boundary file do not align exactly, so the
match is attempted in four decreasing-confidence passes:
    1. exact county CODE
    2. exact NAME within the same province
    3. FORMER name 曾用名 within the same province   (撤县设市/设区 renames)
    4. NAME STEM with the 市/县/区/旗... suffix stripped, same province
Anything still unmatched is returned as NaN and must be reported, never guessed.

Shared by 27_map_tfp.py (TFP maps) and the weather join, so both use identical
county correspondence.
"""
from __future__ import annotations
import re, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import pandas as pd

SHP_2010 = r"E:/Poverty_Data/2010/县级/T2010年初县级.shp"
SUF = re.compile(r"(自治县|自治旗|特区|林区|地区|盟|市|县|区|旗)$")

# ONLY true renames / administrative successions -- never a different place.
# 共青城市 is deliberately absent: it was carved out of 德安县+永修县+星子县 in
# 2010, so mapping it onto any one of them would attribute the wrong county.
MANUAL = {"满州里市": "满洲里市",   # typo: 州 -> 洲
          "霍县": "霍州市",         # renamed 1989
          "布特哈旗": "扎兰屯市",    # renamed 1983
          "江浦县": "浦口区"}       # merged into 浦口区, 2002


def load_boundaries(shp: str = SHP_2010, crs="EPSG:4326"):
    """2010 county polygons with a usable integer `code` column.

    Reprojected to `crs` (default EPSG:4326) because the file ships in
    EPSG:3857: zonal extraction against the 4326 climate rasters silently
    returns all-NaN if the polygons are left in Web Mercator.
    """
    import geopandas as gpd
    g = gpd.read_file(shp).rename(columns={"县级码": "gcode", "地名": "gname",
                                          "曾用名": "gold", "区划码": "gdiv"})
    if crs is not None:
        g = g.to_crs(crs)
    code = pd.to_numeric(g["gcode"], errors="coerce")
    gdiv = pd.to_numeric(g["gdiv"], errors="coerce")
    # 不设区的市 (东莞 441900, 中山 442000, 三亚, 嘉峪关) carry 县级码=0 but a
    # valid 区划码; drop the remaining code-0 rows (HK / Macao / Taiwan).
    g["code"] = code.where(code.fillna(0) > 0, gdiv)
    g = g[g["code"].fillna(0) > 0].copy()
    g["code"] = g["code"].astype(int)
    g["prov"] = g["code"] // 10000
    return g.dissolve(by="code", as_index=False, aggfunc="first")


def build_lookups(g):
    by_name, by_old, by_stem = {}, {}, {}
    for r in g.itertuples():
        by_name.setdefault((r.prov, str(r.gname).strip()), r.code)
        if pd.notna(r.gold):
            by_old.setdefault((r.prov, str(r.gold).strip()), r.code)
        by_stem.setdefault((r.prov, SUF.sub("", str(r.gname).strip())), r.code)
    return set(g.code), by_name, by_old, by_stem


def match_counties(df, g, id_col="countyid", name_col="county_name"):
    """Add `map_code` and `how` columns to df. Reports the tally."""
    codes, by_name, by_old, by_stem = build_lookups(g)

    def one(cid, name, prov):
        if cid in codes:
            return cid, "code"
        n = MANUAL.get(str(name).strip(), str(name).strip())
        if (prov, n) in by_name:
            return by_name[(prov, n)], "name"
        if (prov, n) in by_old:
            return by_old[(prov, n)], "former"
        st = SUF.sub("", n)
        if st and (prov, st) in by_stem:
            return by_stem[(prov, st)], "stem"
        return np.nan, "unmatched"

    prov = df[id_col] // 10000
    res = [one(int(c), nm, int(p)) for c, nm, p in zip(df[id_col], df[name_col], prov)]
    out = df.copy()
    out["map_code"] = [x[0] for x in res]
    out["how"] = [x[1] for x in res]
    n_un = int((out["how"] == "unmatched").sum())
    print("  county match:", out["how"].value_counts().to_dict())
    print(f"  matched {len(out)-n_un}/{len(out)} ({100*(len(out)-n_un)/len(out):.1f}%)")
    if n_un:
        miss = out[out.how == "unmatched"]
        print("  unmatched:", list(zip(miss[id_col].head(15), miss[name_col].head(15))))
    return out
