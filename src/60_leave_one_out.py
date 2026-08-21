# -*- coding: utf-8 -*-
"""
Leave-one-out sensitivity of the NIRS DEA envelope to its extreme vertices.

WHY.  59_frontier_audit.py showed the boundary is set by a handful of counties
-- on some year-panels one county carries 99% of the input range.  That is a
statement about fragility, not yet a test of it.  This script does the test: it
names the counties that define the near-origin end of each input's envelope,
audits their raw input-output vectors for the two mundane explanations (a unit
slip or a decimal error) against the third (a genuinely specialised county), and
then DELETES them and re-solves the whole thing.

If dropping one or two counties moves a large share of the other counties'
efficiency scores, the level of those scores is not a property of the technology
and cross-year TFP comparisons built on them inherit that fragility.

WHAT IS RECOMPUTED.  Two different objects, deliberately:
  * the PER-INPUT envelope f_j(x_j), which is what the figures draw and where
    the suspect vertex is identified;
  * the FULL FOUR-INPUT NIRS efficiency score
        theta_i = y_i / f(x_i),
        f(x_i) = max SUM_k lam_k y_k  s.t. SUM lam_k x_k <= x_i, SUM lam <= 1,
    which is what actually feeds TFP.  A vertex can be harmless in one and
    decisive in the other, so reporting only the first would understate the risk.

SCOPE  2002 / 2007 / 2011 / 2016, Huang-Huai-Hai Plain and Middle-lower Yangtze
Plain, each region solved on its own counties (as the by-region figures do).

Run:  python src/60_leave_one_out.py
Out:  src/figures/frontier/leave_one_out/*.png
      src/dq/loo_vertex_counties.csv, loo_efficiency_shift.csv
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C
import _tfp as T
import _frontier as F

C.set_cjk_font(plt)
OUT_FIG = os.path.join(C.FIG_DIR, "frontier", "leave_one_out")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(C.DQ_DIR, exist_ok=True)

YEARS = [2002, 2007, 2011, 2016]
REGIONS = ["Huang-Huai-Hai Plain", "Middle-lower Yangtze Plain"]
XCOLS = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
SHORT = {"Laborday_impute": "L", "Land_serv_q": "Land",
         "capital_serv_q": "K", "Inter_all_real": "M"}
INK = "#1a1a19"


# --------------------------------------------------------------------- LPs
def eff_scores(X, y, Xe=None, ye=None):
    """theta_i = y_i / f(x_i) for every evaluated point, NIRS, four inputs.

    Solved as the DUAL -- four variables and one constraint per reference point
    instead of one variable per reference point:
        f(xb) = min  xb'u + v   s.t.  x_k'u + v >= y_k for all k,  u >= 0, v >= 0
    The reference set is pruned to the vertices of conv({0} u points) first,
    which is exact because a linear objective is maximised at a vertex.
    """
    Xe = X if Xe is None else Xe
    ye = y if ye is None else ye
    P = np.column_stack([X, y])
    if len(P) > 12:
        s = np.abs(P).max(axis=0); s[s == 0] = 1.0
        try:
            v = np.unique(ConvexHull(np.vstack([P / s, np.zeros(P.shape[1])]),
                                     qhull_options="Qx").vertices)
            v = v[v < len(P)]
            if len(v) >= 6:
                X, y = X[v], y[v]
        except Exception:
            pass
    A = -np.column_stack([X, np.ones(len(X))])
    b = -y
    out = np.full(len(Xe), np.nan)
    for i in range(len(Xe)):
        r = linprog(np.append(Xe[i], 1.0), A_ub=A, b_ub=b,
                    bounds=[(0, None)] * (X.shape[1] + 1), method="highs")
        if r.success and r.fun and r.fun > 0:
            out[i] = ye[i] / r.fun
    return out


def near_origin_vertex(x, y, cid):
    """countyid of the hull vertex nearest the origin -- the county that defines
    the initial NIRS ray, i.e. the highest observed output-input ratio.  That is
    the one point the whole left-hand end of the envelope rests on."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y, cid = x[ok], y[ok], cid[ok]
    o = np.argsort(x); x, y, cid = x[o], y[o], cid[o]
    H = F._upper_hull([(0.0, 0.0, -1)] + list(zip(x, y, cid)))
    v = [h for h in H if h[2] > 0]
    return (int(v[0][2]), float(v[0][0]), float(v[0][1])) if v else (None, np.nan, np.nan)


# --------------------------------------------------------------------- main
def main():
    d = T.load_panel()
    d = d[d.year >= 1985]
    d = F.add_region(d)
    nm = d.groupby("countyid").county_name.last()

    vrows, srows = [], []
    for reg in REGIONS:
        g_all = d[d.region == reg]
        for yr in YEARS:
            g = g_all[g_all.year == yr]
            if len(g) < 80:
                print(f"[{reg} {yr}] only {len(g)} counties -- skipped"); continue
            X = g[XCOLS].values; y = g.real_gvp.values; cid = g.countyid.values

            # --- who defines the near-origin end of each input's envelope? ---
            picks = {}
            for j, c in enumerate(XCOLS):
                q, xv, yv = near_origin_vertex(X[:, j], y, cid)
                if q is None:
                    continue
                picks[c] = q
                s = g[g.countyid == q].iloc[0]
                pr = {k: float((g[k] < s[k]).mean()) for k in XCOLS + ["real_gvp"]}
                vrows.append(dict(
                    region=reg, year=yr, input_dim=SHORT[c], countyid=q,
                    county_name=str(nm.get(q, "")),
                    **{k: float(s[k]) for k in XCOLS + ["real_gvp"]},
                    **{f"pctile_{SHORT.get(k, k)}": 100 * pr[k]
                       for k in XCOLS + ["real_gvp"]},
                    ratio_y_over_x=float(s.real_gvp / s[c]),
                    ratio_rank=float((g.real_gvp / g[c] < s.real_gvp / s[c]).mean() * 100)))

            drop = sorted(set(picks.values()))
            base = eff_scores(X, y)
            keep = ~np.isin(cid, drop)
            loo = eff_scores(X[keep], y[keep], Xe=X, ye=y)   # same evaluation points
            both = np.isfinite(base) & np.isfinite(loo)
            rel = np.where(both, loo / base - 1.0, np.nan)

            srows.append(dict(
                region=reg, year=yr, n_counties=len(g), n_dropped=len(drop),
                dropped=";".join(str(q) for q in drop),
                median_abs_pct=100 * np.nanmedian(np.abs(rel)),
                mean_abs_pct=100 * np.nanmean(np.abs(rel)),
                p95_abs_pct=100 * np.nanquantile(np.abs(rel), .95),
                max_abs_pct=100 * np.nanmax(np.abs(rel)),
                share_gt1pct=100 * np.nanmean(np.abs(rel) > .01),
                share_gt5pct=100 * np.nanmean(np.abs(rel) > .05),
                mean_eff_before=np.nanmean(base), mean_eff_after=np.nanmean(loo)))
            print(f"[{reg[:16]:16s} {yr}] drop {len(drop)} county(ies) -> "
                  f"median |d eff| {100*np.nanmedian(np.abs(rel)):5.2f}%, "
                  f"{100*np.nanmean(np.abs(rel) > .01):5.1f}% of counties move >1%, "
                  f"{100*np.nanmean(np.abs(rel) > .05):4.1f}% move >5%")

            # ---------------------------------------------------------- figure
            fig, axes = plt.subplots(1, 5, figsize=(23, 4.2))
            for ax, c in zip(axes[:4], XCOLS):
                for lab, m, col, ls in [("all counties", np.ones(len(g), bool),
                                         "#1f4e79", "-"),
                                        ("vertex dropped", keep, "#b3541e", "--")]:
                    hx, hy, n, vx, vy = F.dea_frontier(X[m, XCOLS.index(c)], y[m],
                                                       return_vertices=True)
                    if len(hx) > 1:
                        ax.plot(hx, hy, ls, color=col, lw=1.9, label=f"{lab} (n={n})")
                ax.scatter(X[:, XCOLS.index(c)], y, s=6, alpha=.18, color="0.45",
                           linewidths=0, rasterized=True)
                q = picks.get(c)
                if q is not None:
                    s = g[g.countyid == q].iloc[0]
                    ax.scatter([s[c]], [s.real_gvp], s=55, facecolor="none",
                               edgecolor="#b3541e", linewidth=1.8, zorder=6)
                    ax.annotate(f"{q}", (s[c], s.real_gvp), fontsize=7,
                                color="#b3541e", xytext=(5, 4),
                                textcoords="offset points")
                ax.set_xlim(0, np.nanquantile(X[:, XCOLS.index(c)], .97))
                ax.set_ylim(0, np.nanquantile(y, .97))
                ax.set_xlabel(SHORT[c], fontsize=8)
                ax.set_ylabel("real GVP", fontsize=8)
                ax.set_title(SHORT[c], fontsize=9, loc="left", fontweight="bold")
                ax.grid(alpha=.25); ax.legend(fontsize=6.5)
            ax = axes[4]
            r = 100 * rel[np.isfinite(rel)]
            ax.hist(r, bins=60, color="#1f4e79", alpha=.85)
            ax.axvline(0, color="0.4", lw=.8)
            ax.set_xlabel("change in efficiency score, % (after − before)", fontsize=8)
            ax.set_ylabel("counties", fontsize=8)
            ax.set_title(f"4-input efficiency shift\n"
                         f"{100*np.nanmean(np.abs(rel) > .01):.1f}% of counties move >1%",
                         fontsize=9, loc="left", fontweight="bold")
            ax.grid(alpha=.25)
            fig.suptitle(f"{reg} {yr} — leave-one-out on the near-origin envelope "
                         f"vertices ({', '.join(str(q) for q in drop)})",
                         fontsize=11, fontweight="bold", color=INK)
            fig.tight_layout(rect=(0, 0, 1, .93))
            p = os.path.join(OUT_FIG, f"loo_{SHORT.get(reg, reg.split()[0])}_{yr}.png")
            fig.savefig(p, dpi=160, facecolor="white"); plt.close(fig)
            print("   saved", os.path.basename(p))

    V = pd.DataFrame(vrows); S = pd.DataFrame(srows)
    V.to_csv(os.path.join(C.DQ_DIR, "loo_vertex_counties.csv"),
             index=False, encoding="utf-8-sig")
    S.to_csv(os.path.join(C.DQ_DIR, "loo_efficiency_shift.csv"),
             index=False, encoding="utf-8-sig")
    print(f"\nsaved -> dq/loo_vertex_counties.csv ({len(V)} rows), "
          f"dq/loo_efficiency_shift.csv ({len(S)} rows)")
    print(f"figures -> {OUT_FIG}")


if __name__ == "__main__":
    main()
