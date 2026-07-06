# -*- coding: utf-8 -*-
"""
Resolve every production county to a clean (code, name) anchored on the
authoritative cropland ag-county list, per the agreed rules:

  6/7 REMOVE prefecture-level units (地级市): countyid%100==0, or name with
      市辖 / 市郊 / 市中区 / ...郊区  (济南市中区, 扬州市郊区, 株洲市430200, 南昌市辖).
  5   COMBINE rename pairs (county_corrections.RENAME_COMBINE): 通什市+五指山市, ...
  4/A code IS in the ag list -> keep the code, ADOPT the ag name (fixes typos like
      杨凌区->杨陵区 and 县/区 admin changes; the code is authoritative).
  3   name matches a unique ag county (same province) -> re-code to that ag code.
  1   urban 区 whose code/name is NOT in the ag list -> remove.
  2   ...but 区 that ARE in the ag list are kept (handled by the code/name match).
  B   wrong code with a same-province prefix-2 ag name -> suggest re-code (REVIEW).

Outputs (data/):
  roster_resolution.csv  one row per production county with `action` + final code/name
  county_roster_final.csv the kept agricultural counties (final code + name)
Nothing is applied to the panel yet.
Run:  python src/00d_resolve_roster.py
"""
import os, sys, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import pandas as pd, numpy as np
import _common as C
import county_corrections as CC

def norm(s): return str(s).strip() if pd.notna(s) else ""

ag = pd.read_csv(C.AG_LIST, encoding="utf-8-sig")
ag["code"] = pd.to_numeric(ag["code"], errors="coerce")
ag = ag.dropna(subset=["code"]); ag["code"] = ag["code"].astype(int)
ag["prov2"] = ag["code"] // 10000
ag["agname"] = ag["地名"].map(norm)
ag_codeset = set(ag["code"])
ag_name_of = ag.drop_duplicates("code").set_index("code")["agname"]
ag_name2codes = ag.groupby(["prov2", "agname"])["code"].apply(lambda s: sorted(set(s))).to_dict()
ag_by_prov = {p: list(zip(g["agname"], g["code"])) for p, g in ag.groupby("prov2")}

combine_canon = {m: canon for canon, _, members in CC.RENAME_COMBINE for m in members}
combine_name  = {canon: name for canon, name, _ in CC.RENAME_COMBINE}

# Manual overrides from review (take precedence over the rules below):
#   keep_own = distinct place, do NOT apply the wrong prefix-2 merge (kept under its
#              own code; these 6 are NOT in the ag list -> flagged in_ag_list=False).
#   remove   = drop (地级市 / duplicate code).
OVERRIDES = {
    152632: "keep_own", 152726: "keep_own", 152825: "keep_own", 152826: "keep_own",
    653221: "keep_own", 652702: "keep_own",          # 后旗/杭锦旗/乌拉特中-后旗/和田县/阿拉山口市
    130104: "keep_own",                               # 石家庄桥西区 (keep distinct)
    441900: "keep_own",                               # 东莞市 (不设区的市 -> keep)
    362401: "remove",                                 # 吉安市 (地级市)
    130503: "remove",                                 # 邢台桥西区 (drop duplicate)
    140202: "remove",                                 # 城区
    340501: "remove",                                 # 马鞍山辖
    429001: "remove",                                 # 随州市
    372901: "remove",                                 # 菏泽市
    230201: "remove",                                 # 齐齐哈尔市 (地级市)
    142601: "remove", 320882: "remove", 352201: "remove",   # 临汾市/淮安市/宁德市
    612701: "remove", 642221: "remove", 150101: "remove",   # 榆林市/固原市/呼和浩特市
    412701: "remove", 412801: "remove",                     # 周口市/驻马店市
    350527: "remove",                                       # 金门县 (Taiwan-administered)
    621224: "remove",                                       # 康县 (keep 622625)
    152301: "remove",                                       # 通辽市
    610103: "remove",                                       # 碑林区
}
# also honour the county_corrections.REMOVE list (duplicate codes / unusable units)
for _c in CC.REMOVE:
    OVERRIDES.setdefault(int(_c), "remove")

PREF_TOKENS = ("市辖", "市郊", "市中区", "辖")        # 辖 catches 马鞍山辖 / 南昌市辖
def is_prefecture(cid, names):
    if cid % 100 == 0:
        return True
    for n in names:
        if any(t in n for t in PREF_TOKENS) or n.endswith("郊区"):
            return True
    return False

base = pd.read_stata(C.BASE_DTA, convert_categoricals=False)
g = base.groupby("countyid")
# last NON-EMPTY name (some counties go blank in later years, e.g. 120107 塘沽区,
# 142201 忻县 -> name_latest must not be the trailing empty string)
def _last_nonempty(s):
    vals = [norm(x) for x in s if norm(x)]
    return vals[-1] if vals else ""
name_latest = base.sort_values("year").groupby("countyid")["county_name"].apply(_last_nonempty)
allnames = g["county_name"].apply(lambda s: sorted({norm(x) for x in s if norm(x)}))
cov = g["GVP_allagr_impute"].apply(lambda s: int(s.notna().sum()))

# universe = ALL rural county-level units (县/县级市/旗… kept even if NOT in the ag
# list) + urban 区 that ARE in the ag list.  地级市/aggregates removed.
rows = []
for cid in g.groups:
    cid = int(cid); prov2 = cid // 10000
    names = list(allnames[cid]); nm = name_latest.get(cid, "")
    # apply name-typo fixes (毫县->亳县, 林高县->临高县) before matching
    if cid in CC.NAME_FIX:
        nm = CC.NAME_FIX[cid]
        if nm not in names: names = names + [nm]
    rec = dict(prod_countyid=cid, prov2=prov2, name_latest=nm,
               all_names="|".join(names), gvp_nonNA=int(cov.get(cid, 0)),
               action="", final_code="", final_name="", similarity="")
    # manual overrides take precedence
    if cid in OVERRIDES:
        if OVERRIDES[cid] == "remove":
            rec["action"] = "remove_override"
        else:
            rec.update(action="keep_own", final_code=cid, final_name=nm)
        rows.append(rec); continue
    # remove counties with no valid name
    if not names or not nm:
        rec["action"] = "remove_noname"; rows.append(rec); continue
    # 6/7 prefecture-level -> remove
    if is_prefecture(cid, names):
        rec["action"] = "remove_prefecture"; rows.append(rec); continue
    # 5 combine (rename pairs)
    if cid in combine_canon:
        canon = combine_canon[cid]
        rec.update(action="combine", final_code=canon, final_name=combine_name[canon])
        rows.append(rec); continue
    # name match to ag (with typo-fixed names)
    cand = sorted({c for n in names for c in ag_name2codes.get((prov2, n), [])})
    is_urban = nm.endswith("区") and not nm.endswith("林区")
    if cid in ag_codeset:
        rec.update(action="keep_adopt_name", final_code=cid, final_name=ag_name_of[cid])
    elif len(cand) == 1:
        rec.update(action="recode_name_match", final_code=cand[0], final_name=ag_name_of[cand[0]])
    elif len(cand) > 1:
        rec.update(action="ambiguous_dup", final_code=cand[0], final_name=ag_name_of[cand[0]])
    elif is_urban:
        rec["action"] = "remove_district"              # urban 区 not in ag -> drop
    else:
        # RURAL not in ag: try 撤市设区 prefix-2 (->ag 区), else KEEP under own code
        best = None
        for agname, agcode in ag_by_prov.get(prov2, []):
            if len(nm) >= 2 and nm[:2] == agname[:2]:
                r = difflib.SequenceMatcher(None, nm, agname).ratio()
                if best is None or r > best[0]:
                    best = (r, agname, agcode)
        if best:
            rec.update(action="recode_prefix2_REVIEW", final_code=best[2],
                       final_name=best[1], similarity=round(best[0], 2))
        else:
            rec.update(action="keep_rural", final_code=cid, final_name=nm)
    rows.append(rec)

res = pd.DataFrame(rows)
order = ["remove_prefecture", "remove_override", "remove_noname", "remove_district",
         "ambiguous_dup", "recode_prefix2_REVIEW", "keep_own", "keep_rural",
         "recode_name_match", "combine", "keep_adopt_name"]
res = res.sort_values(["action", "gvp_nonNA"],
                      key=lambda s: s.map({a: i for i, a in enumerate(order)}) if s.name == "action" else s,
                      ascending=[True, False])
res.to_csv(os.path.join(C.DATA, "roster_resolution.csv"), index=False, encoding="utf-8-sig")

KEEP = {"keep_adopt_name", "recode_name_match", "combine", "recode_prefix2_REVIEW",
        "keep_own", "keep_rural"}
final = res[res["action"].isin(KEEP)].copy()
final["final_code"] = final["final_code"].astype(int)
final["in_ag_list"] = final["final_code"].isin(ag_codeset)
roster = (final.groupby("final_code")
          .agg(final_name=("final_name", "first"),
               n_prod_codes=("prod_countyid", "nunique"),
               prod_codes=("prod_countyid", lambda s: "|".join(map(str, sorted(s)))),
               in_ag_list=("in_ag_list", "first")).reset_index().sort_values("final_code"))
roster.to_csv(os.path.join(C.DATA, "county_roster_final.csv"), index=False, encoding="utf-8-sig")

print("Action counts:")
print(res["action"].value_counts().reindex(order).to_string())
print(f"\nFinal roster: {len(roster)} unique counties "
      f"({int(roster['in_ag_list'].sum())} in ag list, "
      f"{int((roster['n_prod_codes']>1).sum())} merged from >1 prod code)")
print(f"Saved -> data/roster_resolution.csv, data/county_roster_final.csv")
print("\nrecode_prefix2_REVIEW (verify these — watch Inner-Mongolia 旗 false positives):")
print(res[res.action=='recode_prefix2_REVIEW'].head(15)[
    ['prod_countyid','name_latest','final_code','final_name','similarity','gvp_nonNA']].to_string(index=False))
