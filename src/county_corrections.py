# -*- coding: utf-8 -*-
"""
County administrative harmonisation table + apply/validate functions.

This is the editable single source of truth for fixing the county roster BEFORE
any cleaning (used by 01_clean_panel.py).  Three operations:

  RENAME_COMBINE  old codes -> the NEW administrative entity (code + name). The
                  member series are concatenated under the canonical (new) code;
                  on overlapping years the canonical's own record wins.
  REMOVE          codes dropped entirely (duplicate codes / unusable aggregates /
                  explicitly-listed fully-NA units).
  NAME_FIX        same code, corrected name (typos).
  Hainan (SID 46): founded 1988 -> drop its pre-1988 rows.

Codes were validated against data/cty_prod_account_agg_forAXN.dta (see report()).
"""
import numpy as np
import pandas as pd

# (canonical_code, canonical_name, [member_codes])  -- canonical = the NEW entity
RENAME_COMBINE = [
    (469001, "五指山市",  [460001, 469001]),   # 通什市 -> 五指山市
    (469002, "琼海市",    [460002, 469002]),
    (469021, "定安县",    [460025, 469021]),   # 丁安县(typo) -> 定安县
    (650109, "米东区",    [652303, 650109]),   # 米泉市 -> 米东区
    (654003, "奎屯市",    [654001, 654003]),
    (150782, "牙克石市",  [152104, 150782]),   # 喜桂图旗 -> 牙克石市  (code 152104)
    (150785, "根河市",    [152105, 150785]),   # 额尔古纳左旗 -> 根河市
    (150784, "额尔古纳市", [152106, 150784]),   # 额尔古纳右旗 -> 额尔古纳市
    (231223, "青冈县",    [232326, 231223]),   # 青岗(typo) -> 青冈县  (code 232326)
    (330109, "萧山区",    [330181, 330109]),   # 肖山市 -> 萧山区
    (330803, "衢江区",    [330821, 330803]),   # 衢县 -> 衢江区
    (361128, "鄱阳县",    [362330, 361128]),   # 波阳县 -> 鄱阳县
    (451281, "宜州市",    [452702, 451281]),   # 宜山县 -> 宜州市
    (612503, "华州区",    [610521, 612503]),   # 华县 -> 华州区
    (141181, "孝义市",    [142301, 141181]),   # 孝县 -> 孝义市
    (540426, "朗县",      [542627, 540426]),   # 郎县(typo) -> 朗县
    (362402, "井冈山市",  [362432, 362402]),   # 宁冈县 MERGED into 井冈山市 in 2000 (summed in 00e)
]

REMOVE = [
    421281,            # 赤壁市  (keep 421103)
    421182,            # 武穴市  (keep 421009)
    412701, 412801,    # 周口市, 驻马店市 (prefecture aggregates)
    372901,            # 菏泽市
    621224,            # 康县    (keep 622625)
    622201,            # 张掖市
    141182,            # 汾阳市
    150101,            # 呼和浩特市
    230201,            # 齐齐哈尔市
    450127,            # 横县    (keep 452122)
    # explicitly-listed unusable units (also caught by the fully-NA rule in 01)
    420381, 542135, 542136, 542134, 542137, 542528,
    654004, 659004, 659006, 659003, 360982,
]

NAME_FIX = {341281: "亳县", 460028:"临高县"}          # 毫县 -> 亳县

HAINAN_SID, HAINAN_FROM = 46, 1988

# canonical codes that must SURVIVE the urban-district (区) name filter, because
# they are农业 counties that were撤县设区 (kept per the user's instruction).
CANON_KEEP = {canon for canon, _, _ in RENAME_COMBINE}

# old member code -> canonical (new) code, so the cropland ag-county list (which
# uses 2010 codes) can be mapped into the same canonical space as the panel.
MEMBER_TO_CANON = {m: canon for canon, _, members in RENAME_COMBINE
                   for m in members if m != canon}


def to_canonical(code):
    """Map a (possibly old) county code to its canonical code."""
    return MEMBER_TO_CANON.get(int(code), int(code))


GVP_COL = "GVP_allagr_impute"


def apply_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with name fixes, rename/combine, removals and Hainan-1988 applied."""
    df = df.copy()
    for code, nm in NAME_FIX.items():
        df.loc[df["countyid"] == code, "county_name"] = nm

    df["_prio"] = 0
    for canon, name, members in RENAME_COMBINE:
        m = df["countyid"].isin(members)
        df.loc[m, "_prio"] = np.where(df.loc[m, "countyid"] == canon, 1, 2)
        df.loc[m, "countyid"] = canon
        df.loc[m, "county_name"] = name

    # dedupe overlapping (countyid, year): GVP-present first, then canonical-own
    df["_gvpok"] = df[GVP_COL].notna().astype(int)
    df = (df.sort_values(["countyid", "year", "_gvpok", "_prio"],
                         ascending=[True, True, False, True])
            .drop_duplicates(["countyid", "year"], keep="first"))

    df = df[~df["countyid"].isin(REMOVE)]
    df = df[~((df["SID"] == HAINAN_SID) & (df["year"] < HAINAN_FROM))]
    return df.drop(columns=["_prio", "_gvpok"]).reset_index(drop=True)


def report(raw: pd.DataFrame) -> None:
    """Validate every code in the table against the raw panel; print a summary."""
    info = raw.groupby("countyid").agg(
        name=("county_name", lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
        y0=("year", "min"), y1=("year", "max"))
    def show(c):
        if c in info.index:
            r = info.loc[c]; return f"{c} {r['name']} {int(r.y0)}-{int(r.y1)}"
        return f"{c} *** NOT IN DATA ***"
    print("RENAME_COMBINE (-> canonical NEW entity):")
    for canon, name, members in RENAME_COMBINE:
        print(f"  {canon} {name}  <=  " + " + ".join(show(m) for m in members))
    print("REMOVE:")
    for c in REMOVE:
        print("  " + show(c))
    print("NAME_FIX:", {c: (info.loc[c, "name"] if c in info.index else "?", nm)
                        for c, nm in NAME_FIX.items()})
