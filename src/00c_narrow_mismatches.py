# -*- coding: utf-8 -*-
"""
Narrow the NO_MATCH list to the production counties that ALMOST match an ag county
(same province) -- i.e. the likely typos / 市↔区 renames / wrong codes worth a quick
manual diagnose -- and drop the rest (genuine urban non-ag districts).

For each NO_MATCH production county, search the ag list in the SAME province for a
near name, by three signals:
   prefix2  first two characters equal   (番禺市 ~ 番禺区, 万山特区 ~ 万山区, 文山壮族苗族自治州 ~ 文山市)
   edit1    Levenshtein distance <= 1     (毫县 ~ 亳县, 林高县 ~ 临高县)
   ratio    difflib similarity >= 0.6     (catch-all, used for ranking)

Output a short, prioritised sheet -> data/mismatch_suggestions.csv
    prod_countyid, prod_name, suggested ag_code/ag_name, reason, similarity, coverage
Run:  python src/00c_narrow_mismatches.py
"""
import os, sys, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import pandas as pd, numpy as np
import _common as C

def lev1(a, b):
    """True if Levenshtein distance <= 1."""
    if a == b: return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la == lb:                                   # one substitution
        return sum(x != y for x, y in zip(a, b)) <= 1
    # one insertion/deletion
    s, t = (a, b) if la < lb else (b, a)
    i = j = diff = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1; j += 1
        else:
            diff += 1; j += 1
            if diff > 1: return False
    return True

cw = pd.read_csv(os.path.join(C.DATA, "county_crosswalk_review.csv"), encoding="utf-8-sig")
nomatch = cw[cw["match_type"] == "NO_MATCH"].copy()

ag = pd.read_csv(C.AG_LIST, encoding="utf-8-sig")
ag["code"] = pd.to_numeric(ag["code"], errors="coerce")
ag = ag.dropna(subset=["code"]); ag["code"] = ag["code"].astype(int)
ag["prov2"] = ag["code"] // 10000
ag["agname"] = ag["地名"].astype(str).str.strip()
ag_codeset = set(ag["code"])
ag_name_of = ag.drop_duplicates("code").set_index("code")["agname"]
ag_by_prov = {p: list(zip(g["agname"], g["code"])) for p, g in ag.groupby("prov2")}

rows = []
for _, r in nomatch.iterrows():
    cid = int(r["prod_countyid"]); nm = str(r["name_latest"]).strip(); prov2 = int(r["prov2"])
    # GROUP A: the production code IS in the ag list but the name didn't match
    #          -> the code is right, the name is a typo / old name (adopt ag name).
    if cid in ag_codeset:
        rows.append(dict(group="A_name_typo", prod_countyid=cid, prod_name=nm, prov2=prov2,
                         suggest_ag_code=cid, suggest_ag_name=ag_name_of.get(cid, ""),
                         reason="code_in_ag", similarity=round(
                             difflib.SequenceMatcher(None, nm, ag_name_of.get(cid, "")).ratio(), 2),
                         y0=int(r["y0"]), y1=int(r["y1"]), gvp_nonNA=int(r["gvp_nonNA"])))
        continue
    # GROUP B: code NOT in ag -> wrong code; suggest the ag county whose name shares
    #          the first two characters (place stem): 番禺市~番禺区, 万山特区~万山区.
    best = None
    for agname, agcode in ag_by_prov.get(prov2, []):
        if len(nm) >= 2 and nm[:2] == agname[:2]:
            ratio = difflib.SequenceMatcher(None, nm, agname).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, agname, agcode)
    if best:
        ratio, agname, agcode = best
        rows.append(dict(group="B_wrong_code", prod_countyid=cid, prod_name=nm, prov2=prov2,
                         suggest_ag_code=agcode, suggest_ag_name=agname, reason="prefix2",
                         similarity=round(ratio, 2), y0=int(r["y0"]), y1=int(r["y1"]),
                         gvp_nonNA=int(r["gvp_nonNA"])))

sug = pd.DataFrame(rows).sort_values(["group", "gvp_nonNA"], ascending=[True, False])
out = os.path.join(C.DATA, "mismatch_suggestions.csv")
sug.to_csv(out, index=False, encoding="utf-8-sig")

print(f"NO_MATCH counties: {len(nomatch)}")
print(f"  -> NARROWED to likely fixes: {len(sug)}  "
      f"(A name-typo: {(sug.group=='A_name_typo').sum()}, B wrong-code: {(sug.group=='B_wrong_code').sum()})")
print(f"  -> no plausible ag match (genuine non-ag/urban): {len(nomatch) - len(sug)}")
print(f"\nSaved -> {os.path.relpath(out, C.ROOT)}")
print("\nGROUP A — code correct, name typo (adopt ag name):")
print(sug[sug.group=='A_name_typo'].head(20)[
    ["prod_countyid","prod_name","suggest_ag_name","similarity","gvp_nonNA"]].to_string(index=False))
print("\nGROUP B — wrong code, prefix-2 ag suggestion:")
print(sug[sug.group=='B_wrong_code'].head(20)[
    ["prod_countyid","prod_name","suggest_ag_code","suggest_ag_name","similarity","gvp_nonNA"]].to_string(index=False))
