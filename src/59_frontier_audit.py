# -*- coding: utf-8 -*-
"""
Who defines the frontier, and how much would it move without them?

MOTIVATION.  The DEA boundary on ~1,850 counties is set by a handful of
observations per year, so "which counties" is not a detail -- it is most of the
estimate.  Two candidate mechanisms were raised:

  (1) a MEASUREMENT mechanism.  capital_serv_q is a PIM series; a county whose
      initial stock is recorded near zero has an abnormally small input and
      therefore looks abnormally efficient, which pushes it onto the boundary.
      This is the same defect that shows up as K/M drifting ~11x across the
      panel and as a near-zero fitted capital elasticity.

  (2) a COMPOSITION mechanism.  Peri-urban districts and island units clear the
      cropland>=15% agricultural filter legitimately -- they really do farm --
      but their agriculture is intensive horticulture or fishery, with few
      labour days and little capital per unit of output.  Their high Y/L is REAL
      and belongs in the sample; the question is only how far it drags a
      best-practice boundary that is meant to describe field-crop technology.

These are separable, and this script separates them: it recomputes every
boundary on the full sample and on the sample excluding urban districts, and
reports (a) which counties define each, (b) how far the boundary MOVES, and
(c) whether the capital tilt survives once composition is held fixed.

Reported in two stages, aggregate first:
  STAGE A  the scalar Tornqvist input index against output -- the one-input
           production function, where composition and measurement both act
           through a single channel.
  STAGE B  the partial views, one input at a time (K, Land, M per labour day),
           where a single mis-recorded input can act directly on its own axis.

Nothing here changes the sample.  It reports what the sample does.

Run:  python src/59_frontier_audit.py
      python src/59_frontier_audit.py --alpha 0.95
Output: src/dq/frontier_audit_vertices.csv, frontier_audit_summary.csv
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import _common as C
import _tfp as T
import _frontier as F

XCOLS = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
SHORT = {"Laborday_impute": "L", "Land_serv_q": "Land",
         "capital_serv_q": "K", "Inter_all_real": "M"}


def hull_vertices(sub, xcol, ycol):
    """countyids on the DEA-NIRS upper hull of (xcol, ycol).

    Same construction as _frontier.dea_frontier -- origin prepended for
    sum(lambda)<=1, then free disposal in x -- but it returns the IDENTITIES of
    the defining counties rather than the drawn curve.
    """
    s = sub[[xcol, ycol, "countyid"]].replace([np.inf, -np.inf], np.nan).dropna()
    s = s[(s[xcol] > 0) & (s[ycol] > 0)].sort_values(xcol)
    if len(s) < 3:
        return []
    pts = [(0.0, 0.0, -1)] + list(map(tuple, s.values))
    H = []
    for p in pts:
        while len(H) >= 2 and F._cross(H[-2][:2], H[-1][:2], p[:2]) >= 0:
            H.pop()
        H.append(p)
    keep, best = [], -np.inf
    for h in H:
        if h[1] >= best:
            best = h[1]; keep.append(h)
    return [int(h[2]) for h in keep if h[2] > 0]


def boundary_gap(g_full, g_sub, xcol, ycol, alpha, kind):
    """How far does the boundary MOVE when the sub-sample is used instead?

    Both boundaries are evaluated on a common grid of x spanning the OVERLAP of
    the two samples' support, and the gap is reported in logs, so it reads as a
    proportional shift in maximum feasible output.  A gap of 0.10 means the full
    sample's boundary claims ~10.5% more output at the same input.
    """
    def curve(g):
        if kind == "dea":
            hx, hy, _ = F.dea_frontier(g[xcol].values, g[ycol].values, rts="nirs")
        else:
            hx, hy = F.quantile_frontier(g[xcol].values, g[ycol].values, alpha=alpha)
        return hx, hy
    ax, ay = curve(g_full)
    bx, by = curve(g_sub)
    if len(ax) < 2 or len(bx) < 2:
        return np.nan
    lo = max(np.min(ax), np.min(bx)); hi = min(np.max(ax), np.max(bx))
    if not (hi > lo):
        return np.nan
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 60))
    fa = np.interp(grid, ax, ay); fb = np.interp(grid, bx, by)
    return float(np.median(np.log(fa) - np.log(fb)))


def tilt(d, verts, cols):
    """Median within-year percentile of each variable among the vertex counties,
    plus the share of vertices in the bottom decile.  50 = no tilt, 10% = random."""
    V = d.merge(verts.assign(_v=1), on=["year", "countyid"], how="inner")
    out = {}
    for c in cols:
        out[c] = (100 * V["pr_" + c].median(), 100 * (V["pr_" + c] < .10).mean())
    return out, len(V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(C.DQ_DIR, exist_ok=True)

    d = T.load_panel()
    d["district"] = d.county_name.map(C.is_urban_district)
    d["x"], w = F.aggregate_index(d, XCOLS, dict(T.BETA_AG))
    d["YL"] = d.real_gvp / d.Laborday_impute
    for c in XCOLS + ["real_gvp"]:
        d["pr_" + c] = d.groupby("year")[c].rank(pct=True)
    d["pr_x"] = d.groupby("year")["x"].rank(pct=True)

    nm = d.groupby("countyid").county_name.last()
    STRATA = {"full": d, "no urban districts": d[~d.district]}
    print(f"panel: {len(d):,d} county-years, {d.countyid.nunique():,d} counties")
    for k, s in STRATA.items():
        print(f"  {k:20s} {s.countyid.nunique():5,d} counties, {len(s):7,d} county-years")
    print(f"index weights: " + ", ".join(f"{SHORT[c]}={w[c]:.4f}" for c in XCOLS))

    vrows, srows = [], []

    # ---------------------------------------------------------------- STAGE A
    print("\n" + "=" * 78)
    print("STAGE A -- AGGREGATE: output against the scalar input index")
    print("=" * 78)
    V = {}
    for k, s in STRATA.items():
        r = [(yr, cid) for yr, g in s.groupby("year") for cid in hull_vertices(g, "x", "real_gvp")]
        V[k] = pd.DataFrame(r, columns=["year", "countyid"])
        vrows.append(V[k].assign(stage="aggregate", stratum=k))
    for k in STRATA:
        t, n = tilt(d, V[k], XCOLS + ["real_gvp", "x"] if False else XCOLS + ["real_gvp"])
        print(f"\n[{k}]  {n} vertex-years, {V[k].countyid.nunique()} distinct counties")
        print(f"  {'variable':18s} {'median pctile':>14s} {'% in bottom decile':>20s}")
        for c in XCOLS + ["real_gvp"]:
            print(f"  {SHORT.get(c, c):18s} {t[c][0]:>14.1f} {t[c][1]:>20.1f}")
            srows.append(dict(stage="aggregate", stratum=k, variable=SHORT.get(c, c),
                              median_pctile=t[c][0], pct_bottom_decile=t[c][1],
                              n_vertex_years=n))
    a_full, a_sub = set(V["full"].countyid), set(V["no urban districts"].countyid)
    print(f"\n  vertex counties: full {len(a_full)}, no-district {len(a_sub)}, "
          f"shared {len(a_full & a_sub)}")
    lost = sorted(a_full - a_sub)
    print(f"  defined the full-sample boundary but are districts: {len(lost)}"
          + (" -> " + ", ".join(f"{c} {nm[c]}" for c in lost[:8]) if lost else ""))
    for kind in ("dea", "alpha"):
        gaps = [boundary_gap(g, g[~g.district], "x", "real_gvp", a.alpha, kind)
                for _, g in d.groupby("year")]
        gaps = np.array([x for x in gaps if np.isfinite(x)])
        print(f"  boundary shift, {kind:5s}: median {100*np.median(gaps):+.2f}% of output "
              f"(p90 {100*np.quantile(gaps,.9):+.2f}%, max {100*gaps.max():+.2f}%)")
        srows.append(dict(stage="aggregate", stratum=f"gap_{kind}", variable="ln_gap",
                          median_pctile=float(np.median(gaps)), pct_bottom_decile=np.nan,
                          n_vertex_years=len(gaps)))

    # ---------------------------------------------------------------- STAGE B
    print("\n" + "=" * 78)
    print("STAGE B -- PARTIAL: output per labour day against ONE input per labour day")
    print("=" * 78)
    for X in ["capital_serv_q", "Land_serv_q", "Inter_all_real"]:
        d["_r"] = d[X] / d.Laborday_impute
        print(f"\n--- {SHORT[X]}/L ---")
        for k, s in STRATA.items():
            s = s.assign(_r=s[X] / s.Laborday_impute)
            r = [(yr, cid) for yr, g in s.groupby("year")
                 for cid in hull_vertices(g, "_r", "YL")]
            vv = pd.DataFrame(r, columns=["year", "countyid"])
            vrows.append(vv.assign(stage=f"{SHORT[X]}/L", stratum=k))
            t, n = tilt(d, vv, XCOLS + ["real_gvp"])
            dsh = 100 * vv.merge(d[["year", "countyid", "district"]],
                                 on=["year", "countyid"]).district.mean()
            print(f"  [{k}]  {n} vertex-years, {vv.countyid.nunique()} counties, "
                  f"{dsh:.0f}% of vertex-years are districts")
            print(f"    {SHORT[X]:>5s} median pctile {t[X][0]:5.1f}   "
                  f"bottom decile {t[X][1]:5.1f}%   |  L {t['Laborday_impute'][0]:5.1f}  "
                  f"Land {t['Land_serv_q'][0]:5.1f}  M {t['Inter_all_real'][0]:5.1f}  "
                  f"Y {t['real_gvp'][0]:5.1f}")
            for c in XCOLS + ["real_gvp"]:
                srows.append(dict(stage=f"{SHORT[X]}/L", stratum=k, variable=SHORT.get(c, c),
                                  median_pctile=t[c][0], pct_bottom_decile=t[c][1],
                                  n_vertex_years=n))
        gaps = []
        for _, g in d.groupby("year"):
            gg = g.assign(_r=g[X] / g.Laborday_impute)
            gaps.append(boundary_gap(gg, gg[~gg.district], "_r", "YL", a.alpha, "dea"))
        gaps = np.array([x for x in gaps if np.isfinite(x)])
        if len(gaps):
            print(f"  boundary shift (DEA): median {100*np.median(gaps):+.2f}% of Y/L")

    pd.concat(vrows).to_csv(os.path.join(C.DQ_DIR, "frontier_audit_vertices.csv"),
                            index=False, encoding="utf-8-sig")
    pd.DataFrame(srows).to_csv(os.path.join(C.DQ_DIR, "frontier_audit_summary.csv"),
                               index=False, encoding="utf-8-sig")
    print("\nsaved -> dq/frontier_audit_vertices.csv, dq/frontier_audit_summary.csv")


if __name__ == "__main__":
    main()
