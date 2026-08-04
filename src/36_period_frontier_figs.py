# -*- coding: utf-8 -*-
"""
DESCRIPTIVE period figures (OECD Fig 9 / Fig 10 analogs).  Two raw inputs only:
labour L (Laborday_impute) and capital K (capital_serv_q), output Y (real_gvp).

*** STANDALONE FRONTIER CONSTRUCTION ***
This is NOT connected to the CPS f_t(x,w) machinery in 33_cps_decomposition.py.
No weather, no Tornqvist scalar x, and the reference set is FRESH PER PERIOD
(not sequential/cumulative), because the point of these figures is to show the
inter-period shift rather than accumulate it away.

Fig 9 analog  scatter of (K/L, Y/L) by period, with a per-period DEA frontier
              f(k) = max SUM lam_i y_i  s.t.  k >= SUM lam_i k_i, SUM lam_i <= 1
              evaluated on a grid of k and joined piecewise-linearly.
Fig 10 analog per-period unit isoquant in (L, K) INPUT LEVELS at a reference
              output y* = sample median real GVP ("unit" = median output), i.e.
                 min SUM lam_i K_i  s.t. SUM lam_i Y_i >= y*,
                                         SUM lam_i L_i <= L, SUM lam_i <= 1
              plus the OBSERVED input paths of candidate counties.

Periods: 5-period Kalirajan et al. (as cited in Gong 2018 JDE) and a 6-period
variant splitting the last at 2004 (Zhang & Bruemmer 2011; national phase-out
of agricultural taxes).  Data start in 1981, so period 1 is 1981-84.

OUTPUTS  figures + CSVs of every plotted series, so the plots can be rebuilt
         in the paper's own style.
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C

C.set_cjk_font(plt)          # Chinese county names in labels

OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)

PERIODS5 = [(1981, 1984), (1985, 1989), (1990, 1993), (1994, 1997), (1998, 2016)]
PERIODS6 = [(1981, 1984), (1985, 1989), (1990, 1993), (1994, 1997),
            (1998, 2003), (2004, 2016)]
# GVP county cross-section is NOT independently measured in these years (see the
# data-quality note printed at run time): 1982 repeats the 1981 shares and
# 2013-2016 repeat one frozen cross-section, each rescaled by a provincial index.
LOCKSTEP_YEARS = [1982, 2013, 2014, 2015, 2016]


def trim_ref(df, col, q):
    """Drop the top q of the reference set on `col` before building a frontier.

    The DEA frontier here is set by a SINGLE extreme observation: max Y/K per
    period is 28-52 against a median of 0.05-0.17, so the raw unit isoquant sits
    ~100x below the observed cloud and its period ordering is dictated by which
    period happened to contain the most extreme point.  Trimming makes the
    descriptive figure readable; it is a presentation choice, reported, not a
    change to the estimator used anywhere else in the project."""
    if q <= 0 or len(df) < 50:
        return df
    return df[df[col] <= df[col].quantile(1 - q)]


def load():
    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    v = ["real_gvp", "Laborday_impute", "capital_serv_q"]
    d = d.dropna(subset=v)
    d = d[(d[v] > 0).all(axis=1)].copy()
    d = d.rename(columns={"real_gvp": "Y", "Laborday_impute": "L",
                          "capital_serv_q": "K"})
    d["KL"] = d.K / d.L
    d["YL"] = d.Y / d.L
    return d


def label(p):
    return f"{p[0]}–{p[1]}"


# --------------------------------------------------------------- Fig 9 frontier
def frontier_kl(k, y, grid):
    """f(k)=max SUM lam y s.t. SUM lam k <= k, SUM lam <=1, lam>=0.
    Solved as the 2-variable DUAL: min k*u + v s.t. k_i u + v >= y_i, u,v>=0.
    Reference points are first pruned to the vertices of conv({0} u {(k,y)}),
    which is exact (the optimum is attained at a vertex)."""
    P = np.column_stack([k, y])
    if len(P) > 4:
        s = np.abs(P).max(axis=0); s[s == 0] = 1
        try:
            v = np.unique(ConvexHull(np.vstack([P / s, np.zeros(2)]),
                                     qhull_options="Qx").vertices)
            v = v[v < len(P)]
            k, y = k[v], y[v]
        except Exception:
            pass
    A = -np.column_stack([k, np.ones_like(k)])
    b = -y
    out = []
    for kk in grid:
        r = linprog([kk, 1.0], A_ub=A, b_ub=b,
                    bounds=[(0, None), (0, None)], method="highs")
        out.append(r.fun if r.success else np.nan)
    return np.array(out)


# -------------------------------------------------------------- Fig 10 isoquant
def isoquant_LK(L, K, Y, ystar, Lgrid):
    """min SUM lam K s.t. SUM lam Y >= ystar, SUM lam L <= L, SUM lam <= 1."""
    n = len(L)
    A_ub = np.vstack([-Y, L, np.ones(n)])           # -SUM lam Y <= -ystar ; SUM lam L <= L ; SUM lam <= 1
    out = []
    for LL in Lgrid:
        r = linprog(K, A_ub=A_ub, b_ub=[-ystar, LL, 1.0],
                    bounds=[(0, None)] * n, method="highs")
        out.append(r.fun if r.success else np.nan)
    return np.array(out)


def pick_candidates(d):
    """Three candidate 'representative' counties under different criteria."""
    n = d.groupby("countyid").year.nunique()
    full = set(n[n == n.max()].index)
    f = d[d.countyid.isin(full)]
    nm = f.groupby("countyid").county_name.last()
    sid = f.groupby("countyid").SID.last()
    g = f.pivot_table(index="countyid", columns="year", values="Y")
    # growth measured to 2012: 2013-2016 carry no genuine county cross-section
    gr = np.log(g[2012] / g[1981]) / (2012 - 1981)
    med = gr.median()
    cand = []
    a = (gr - med).abs().idxmin()
    cand.append(("a_median_growth", a, f"complete panel, 1981-2012 output growth "
                                       f"closest to sample median ({gr[a]:.4f} vs {med:.4f}/yr)"))
    gb = f[f.SID.isin([23, 41])].groupby("countyid").Y.mean()
    b = gb.idxmax()
    cand.append(("b_grain_province", b, f"largest mean output among complete "
                                        f"Heilongjiang/Henan counties ({gb[b]:,.0f})"))
    mk = f.groupby("year").KL.median(); my = f.groupby("year").YL.median()
    t = f.assign(dk=np.log(f.KL) - f.year.map(np.log(mk)),
                 dy=np.log(f.YL) - f.year.map(np.log(my)))
    dist = t.groupby("countyid").apply(lambda s: np.sqrt((s.dk ** 2 + s.dy ** 2).mean()))
    c = dist.idxmin()
    cand.append(("c_typical_path", c, f"minimum RMS log-deviation from the annual "
                                      f"median (K/L, Y/L) path (d={dist[c]:.3f})"))
    return [(tag, int(cid), str(nm[cid]), int(sid[cid]), why) for tag, cid, why in cand]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", choices=["5", "6", "both"], default="both")
    ap.add_argument("--trim", type=float, default=0.05,
                    help="drop this top share of the reference set (Y/L for fig9, "
                         "Y/K for fig10) before building each period frontier; "
                         "0 = raw DEA")
    a = ap.parse_args()
    d = load()
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} counties, "
          f"{d.year.min()}-{d.year.max()}")

    print("\n" + "=" * 78)
    print("DATA-QUALITY NOTE affecting the OUTPUT axis of these figures")
    print("  County GVP has no independently measured cross-section in "
          f"{LOCKSTEP_YEARS}:")
    print("  1982 reproduces the 1981 county shares exactly, and 2013-2016 all")
    print("  reproduce ONE frozen cross-section, each rescaled by a provincial")
    print("  index (max |share_t - share_2013| ~ 1e-8).  Labour and capital are")
    print("  NOT affected.  So within-period Y/L dispersion in the final period")
    print("  is partly mechanical; county growth to 2016 is a provincial index.")
    print("  Candidate selection below therefore measures growth to 2012.")
    print("=" * 78)

    cands = pick_candidates(d)
    print("\nCANDIDATE REPRESENTATIVE COUNTIES (pick one for the final figure):")
    for tag, cid, name, sid, why in cands:
        print(f"  [{tag}] {cid} {name} (SID {sid})\n        {why}")
    pd.DataFrame(cands, columns=["tag", "countyid", "name", "SID", "criterion"]) \
      .to_csv(os.path.join(OUT, "county_candidates.csv"), index=False)

    schemes = ({"5": [("P5", PERIODS5)], "6": [("P6", PERIODS6)],
                "both": [("P5", PERIODS5), ("P6", PERIODS6)]})[a.periods]

    ystar = d.Y.median()
    for tag, PER in schemes:
        cols = plt.cm.viridis(np.linspace(0, .88, len(PER)))

        # ---------------- Fig 9 ----------------
        fig, ax = plt.subplots(figsize=(9.5, 7))
        klo, khi = d.KL.quantile(.01), d.KL.quantile(.99)
        grid = np.exp(np.linspace(np.log(klo), np.log(khi), 90))
        frows = []
        for (p, col) in zip(PER, cols):
            s = d[d.year.between(*p)]
            ax.scatter(s.KL, s.YL, s=3, alpha=.10, color=col, linewidths=0)
        for (p, col) in zip(PER, cols):
            s = d[d.year.between(*p)]
            ref = trim_ref(s, "YL", a.trim)
            fr = frontier_kl(ref.KL.values, ref.YL.values, grid)
            ax.plot(grid, fr, "-", color=col, lw=2.4, label=f"{label(p)}  (n={len(s):,d})")
            frows.append(pd.DataFrame({"scheme": tag, "period": label(p),
                                       "k_KL": grid, "f_YL": fr}))
        pd.concat(frows).to_csv(os.path.join(OUT, f"fig9_frontier_{tag}.csv"), index=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("K/L  — capital service per labour day (log)")
        ax.set_ylabel("Y/L  — real agricultural output per labour day (log)")
        ax.set_title(f"Production frontier by period ({tag}) — county-year observations\n"
                     "per-period DEA frontier, NIRS, fresh reference set per period "
                     "(not sequential)", fontsize=11)
        ax.legend(fontsize=8, title="period"); ax.grid(alpha=.25, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, f"fig9_frontier_{tag}.png"), dpi=170)
        plt.close(fig)
        print(f"  saved fig9_frontier_{tag}.png")

        # ---------------- Fig 10 ----------------
        fig, ax = plt.subplots(figsize=(9.5, 7))
        Lg = np.exp(np.linspace(np.log(d.L.quantile(.02)), np.log(d.L.quantile(.98)), 55))
        # (L grid spans the observed labour range; the isoquant is the minimum K)
        irows = []
        for (p, col) in zip(PER, cols):
            s = d[d.year.between(*p)]
            # sub-sample very large periods: the isoquant LP is O(n) per grid point
            ref = trim_ref(s.assign(YK=s.Y / s.K), "YK", a.trim)
            ss = ref.sample(min(len(ref), 4000), random_state=0)
            iso = isoquant_LK(ss.L.values, ss.K.values, ss.Y.values, ystar, Lg)
            ax.plot(Lg, iso, "-", color=col, lw=2.4, label=f"{label(p)}")
            irows.append(pd.DataFrame({"scheme": tag, "period": label(p),
                                       "L": Lg, "K_min": iso}))
        pd.concat(irows).to_csv(os.path.join(OUT, f"fig10_isoquant_{tag}.csv"), index=False)

        tr_rows = []
        styles = [("o-", "#d62728"), ("s-", "#1f77b4"), ("^-", "#2ca02c")]
        for (tagc, cid, name, sid, _), (mk, cc) in zip(cands, styles):
            t = d[d.countyid == cid].sort_values("year").copy()
            # Radially scale each year's (L,K) to the reference output y* so the
            # path is on the same footing as the y* isoquant (Farrell
            # normalisation).  Raw levels are kept in the CSV.
            f_ = ystar / t.Y
            t["L_s"], t["K_s"] = t.L * f_, t.K * f_
            ax.plot(t.L_s, t.K_s, mk, color=cc, ms=3.2, lw=1.2, alpha=.9,
                    label=f"{name} ({cid}) observed path")
            ax.annotate(str(int(t.year.iloc[0])), (t.L_s.iloc[0], t.K_s.iloc[0]),
                        fontsize=7, color=cc)
            ax.annotate(str(int(t.year.iloc[-1])), (t.L_s.iloc[-1], t.K_s.iloc[-1]),
                        fontsize=7, color=cc, fontweight="bold")
            tr_rows.append(t.assign(tag=tagc)[["tag", "countyid", "county_name",
                                               "year", "L", "K", "Y", "L_s", "K_s"]])
        pd.concat(tr_rows).to_csv(os.path.join(OUT, f"fig10_county_paths_{tag}.csv"),
                                  index=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("L — labour (man-days, log); county paths scaled to y*")
        ax.set_ylabel("K — capital service (log)")
        ax.set_title(f"Unit isoquant by period ({tag}) and observed county input paths\n"
                     f"isoquant at y* = sample median output ({ystar:,.0f}); "
                     "NIRS, fresh reference set per period", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=.25, which="both")
        ax.annotate("County lines are OBSERVED input paths (empirical expansion paths):\n"
                    "a positive trace of how each county's actual (L,K) mix moved as output\n"
                    "grew. They are NOT cost-minimising expansion paths — that would need\n"
                    "county factor prices and a cost-minimisation problem, which we do not have.\n"
                    "County paths are radially scaled to y* (Farrell) so they are\n"
                    "comparable with the isoquant; raw levels are in the CSV.\n"
                    f"Reference set trimmed at the top {a.trim:.1%} on Y/K "
                    "(the raw frontier is set by one extreme observation).\n"
                    f"Output cross-section is provincially rescaled in {LOCKSTEP_YEARS}.",
                    (0.015, 0.015), xycoords="axes fraction", fontsize=7.2, color="0.25",
                    va="bottom")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, f"fig10_isoquant_{tag}.png"), dpi=170)
        plt.close(fig)
        print(f"  saved fig10_isoquant_{tag}.png")

    print(f"\nCSVs -> {OUT}")


if __name__ == "__main__":
    main()
