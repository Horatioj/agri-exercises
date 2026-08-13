# -*- coding: utf-8 -*-
"""
STEP 41 — CPS (2020) annual four-component decomposition of adjacent-period TFP change,
on the ~1,800-county agricultural-county sample (unbalanced).

    ln TFP(t,t-1) = dT + dW + dX + dE

Frontier (sequential, NIRS, weak disposability in w, free disposability in x):

    f_t(x,w) = max  SUM_{(j,k): j<=t, k in K_j} lam_jk y_jk
        s.t.   w = SUM lam_jk w_jk        (2 EQUALITIES, weak disposability)
               x >= SUM lam_jk x_jk       (inequality, free disposability)
               1 >= SUM lam_jk , lam >= 0 (NIRS)

TWO IMPLEMENTATION POINTS THAT MAKE THIS TRACTABLE
--------------------------------------------------
1. SOLVE THE DUAL.  The primal has |R_t| columns (up to 58k) and 4 rows; the
   dual has 4 variables and |R_t| constraints:
       min  xbar*u + v + w1bar*p1 + w2bar*p2
       s.t. x_i*u + v + w1_i*p1 + w2_i*p2 >= y_i   for all i in R_t
            u,v >= 0 ; p1,p2 free
   The dual is ALWAYS feasible (take u=p=0, v=max y_i), so
       dual UNBOUNDED  <=>  primal INFEASIBLE
   which gives clean detection of the cross-period infeasibility that arises
   when w_t lies outside conv({0} u {w_jk : j<=t-1}).

2. EXACT REFERENCE-SET PRUNING.  The attainable set of (SUM lam x, SUM lam w,
   SUM lam y) over {lam>=0, SUM lam<=1} is exactly conv({0} u {(x,w,y)_i}), and
   a linear objective over it is maximised at a VERTEX.  Non-vertices are convex
   combinations of vertices, so dropping them leaves the hull -- and hence every
   f value -- unchanged.  Pruning R to the vertices of the 4-D hull takes
   58,654 points down to ~213 and is exact: verified max relative difference
   1.6e-15 against the unpruned LP, 68x faster.
   Hulls are built incrementally: conv(R_t) = conv(vertices(R_{t-1}) u K_t).

SUM-IDENTITY CHECK (step 4 of the spec)
   dT+dW+dX+dE = ln(y_t/x_t) - ln(y_{t-1}/x_{t-1}) is an ALGEBRAIC identity: the
   six cross/mixed frontier terms cancel exactly, leaving only the two
   own-period terms and x.  It therefore verifies the assembly code, but does
   NOT validate the six cross LPs -- stated so the check is not over-read.

OUTPUTS (src/clean/cps/)
  cps_county_pairs.csv     county x pair: dT,dW,dX,dE and the 8 log-frontier values
  cps_national_annual.csv  national annual series (unweighted + GVP-weighted)
  cps_infeasibility.csv    infeasibility rate by year
  cps_entry_diagnostic.csv first-appearance share of R_j by year
  cps_summary.csv          CPS Table 2/3-style mean/sd summary
"""
from __future__ import annotations
import os, sys, time, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from joblib import Parallel, delayed
import _common as C

OUT = os.path.join(C.CLEAN_DIR, "cps")
os.makedirs(OUT, exist_ok=True)
BND = [(0, None), (0, None), (None, None), (None, None)]     # u,v >=0 ; p1,p2 free


# ----------------------------------------------------------------- LP kernel
def make_dual(ref):
    """ref = (x, w1, w2, y) arrays -> (A_ub, b_ub) of the dual constraints."""
    x, w1, w2, y = ref
    return -np.column_stack([x, np.ones_like(x), w1, w2]), -y


def fval(A_ub, b_ub, xb, w1b, w2b):
    """ln-free frontier value f(xb,wb); NaN if the primal is infeasible."""
    r = linprog(np.array([xb, 1.0, w1b, w2b]), A_ub=A_ub, b_ub=b_ub,
                bounds=BND, method="highs")
    if r.status == 3:            # dual unbounded  <=>  primal infeasible
        return np.nan
    if not r.success or r.fun is None or r.fun <= 0:
        return np.nan
    return float(r.fun)


def hull_prune(x, w1, w2, y):
    """Indices of the vertices of conv({0} u pts) -- exact, see module docstring."""
    P = np.column_stack([x, w1, w2, y])
    if len(P) <= 6:
        return np.arange(len(P))
    scale = np.abs(P).max(axis=0); scale[scale == 0] = 1.0
    S = np.vstack([P / scale, np.zeros(4)])
    try:
        v = np.unique(ConvexHull(S, qhull_options="Qx").vertices)
    except Exception:
        return np.arange(len(P))
    return v[v < len(P)]


# ------------------------------------------------------------------- per pair
def do_pair(pt, tt, ref_prev, ref_cur, obs_prev, obs_cur, own_prev, own_cur):
    """Six remaining LPs per county; own-period values are passed in."""
    Ap, bp = make_dual(ref_prev)
    Ac, bc = make_dual(ref_cur)
    rows = []
    for cid, (xp, w1p, w2p, yp) in obs_prev.items():
        cur = obs_cur.get(cid)
        if cur is None:
            continue
        xt, w1t, w2t, yt = cur
        A_ = own_cur.get(cid, np.nan)          # ln f_t(x_t,w_t)  -- cached
        H_ = own_prev.get(cid, np.nan)         # ln f_{t-1}(x_{t-1},w_{t-1})
        B_ = fval(Ac, bc, xt, w1p, w2p)        # f_t (x_t,   w_p)
        Cc = fval(Ac, bc, xp, w1t, w2t)        # f_t (x_p,   w_t)
        D_ = fval(Ac, bc, xp, w1p, w2p)        # f_t (x_p,   w_p)   Malmquist cross
        E_ = fval(Ap, bp, xt, w1t, w2t)        # f_p (x_t,   w_t)   Malmquist cross
        F_ = fval(Ap, bp, xt, w1p, w2p)        # f_p (x_t,   w_p)
        G_ = fval(Ap, bp, xp, w1t, w2t)        # f_p (x_p,   w_t)
        vals = dict(A=A_, B=np.log(B_) if B_ == B_ else np.nan,
                    C=np.log(Cc) if Cc == Cc else np.nan,
                    D=np.log(D_) if D_ == D_ else np.nan,
                    E=np.log(E_) if E_ == E_ else np.nan,
                    F=np.log(F_) if F_ == F_ else np.nan,
                    G=np.log(G_) if G_ == G_ else np.nan, H=H_)
        bad = [k for k, v in vals.items() if not np.isfinite(v)]
        if bad:
            rows.append(dict(countyid=cid, year=tt, year_prev=pt,
                             infeasible=1, missing="".join(sorted(bad))))
            continue
        A_, B_, Cc, D_, E_, F_, G_, H_ = (vals[k] for k in "ABCDEFGH")
        dT = .5 * ((A_ - E_) + (D_ - H_))
        dW = .25 * ((Cc - D_) + (A_ - B_) + (G_ - H_) + (E_ - F_))
        dX = (np.log(xp) - np.log(xt)) + .25 * ((A_ - Cc) + (B_ - D_)
                                                + (E_ - G_) + (F_ - H_))
        dE = (np.log(yt) - A_) - (np.log(yp) - H_)
        lhs = dT + dW + dX + dE
        rhs = (np.log(yt) - np.log(xt)) - (np.log(yp) - np.log(xp))
        if abs(lhs - rhs) > 1e-8:
            raise AssertionError(
                f"sum identity violated: county {cid} pair {pt}->{tt}: "
                f"{lhs:.12f} vs {rhs:.12f} (diff {lhs-rhs:.3e})")
        rows.append(dict(countyid=cid, year=tt, year_prev=pt, infeasible=0,
                         dT=dT, dW=dW, dX=dX, dE=dE, lnTFP=rhs,
                         y_t=yt, x_t=xt, lnf_tt=A_, lnf_pp=H_))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    f = pd.read_csv(os.path.join(OUT, "cps_frame.csv"))
    years = sorted(f.year.unique())
    by_year = {y: g for y, g in f.groupby("year")}
    K = {y: set(g.countyid) for y, g in by_year.items()}
    print(f"frame: {len(f):,d} county-years, {f.countyid.nunique():,d} counties, "
          f"{years[0]}-{years[-1]}")

    # ---- entry-contamination diagnostic (spec step 6) -------------------
    seen, ent = set(), []
    for y in years:
        new = K[y] - seen
        ent.append(dict(year=y, n_Kj=len(K[y]), n_first_appearance=len(new),
                        share_of_Kj=len(new) / len(K[y])))
        seen |= K[y]
    ent = pd.DataFrame(ent)
    cum = 0; rows = []
    for y in years:
        cum += len(by_year[y])
        rows.append(cum)
    ent["n_Rj"] = rows
    ent["first_appearance_share_of_Rj"] = ent.n_first_appearance / ent.n_Rj
    ent.to_csv(os.path.join(OUT, "cps_entry_diagnostic.csv"), index=False)
    print("\nentry contamination (first-appearance counties in R_j):")
    print(ent[["year", "n_Kj", "n_first_appearance", "share_of_Kj",
               "n_Rj", "first_appearance_share_of_Rj"]].head(8).round(4).to_string(index=False))
    print(f"  ... peak share of K_j = {ent.share_of_Kj.max():.3f} in "
          f"{int(ent.loc[ent.share_of_Kj.idxmax(),'year'])}; "
          f"post-1985 mean share of R_j = "
          f"{ent[ent.year>1985].first_appearance_share_of_Rj.mean():.4f}")

    # ---- sequential hulls, built incrementally --------------------------
    print("\nbuilding sequential reference sets (exact hull pruning) ...")
    refs, keep = {}, None
    t0 = time.time()
    for y in years:
        g = by_year[y]
        arr = np.column_stack([g.x.values, g.w1.values, g.w2.values, g.y.values])
        cand = arr if keep is None else np.vstack([keep, arr])
        idx = hull_prune(cand[:, 0], cand[:, 1], cand[:, 2], cand[:, 3])
        keep = cand[idx]
        refs[y] = (keep[:, 0].copy(), keep[:, 1].copy(),
                   keep[:, 2].copy(), keep[:, 3].copy())
        print(f"   R_{y}: {len(cand):,d} candidates -> {len(keep):,d} vertices", flush=True)
    print(f"   hulls built in {time.time()-t0:.0f}s")

    obs = {y: {r.countyid: (r.x, r.w1, r.w2, r.y) for r in by_year[y].itertuples()}
           for y in years}

    # ---- own-period frontier values, one per county-year ----------------
    print("\nown-period f_j(x_j,w_j) ...")
    def own_year(y):
        A_, b_ = make_dual(refs[y])
        out = {}
        for cid, (x_, w1_, w2_, y_) in obs[y].items():
            v = fval(A_, b_, x_, w1_, w2_)
            out[cid] = np.log(v) if v == v else np.nan
        return y, out
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, backend="loky", verbose=0)(
        delayed(own_year)(y) for y in years)
    own = dict(res)
    n_own_bad = sum(1 for y in years for v in own[y].values() if not np.isfinite(v))
    print(f"   {sum(len(own[y]) for y in years):,d} LPs in {time.time()-t0:.0f}s; "
          f"non-finite: {n_own_bad}")

    # ---- adjacent pairs --------------------------------------------------
    pairs = [(p, t) for p, t in zip(years[:-1], years[1:]) if t == p + 1]
    print(f"\ndecomposing {len(pairs)} adjacent pairs on {a.jobs} workers ...")
    t0 = time.time()
    out = Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
        delayed(do_pair)(p, t, refs[p], refs[t], obs[p], obs[t], own[p], own[t])
        for p, t in pairs)
    rows = [r for chunk in out for r in chunk]
    print(f"   {time.time()-t0:.0f}s")

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "cps_county_pairs.csv"), index=False)
    ok = d[d.infeasible == 0].copy()
    print(f"\ncounty-pairs: {len(d):,d}; usable {len(ok):,d} "
          f"({100*len(ok)/len(d):.1f}%); infeasible {int(d.infeasible.sum()):,d}")
    print("   SUM-IDENTITY CHECK PASSED for every evaluated county-pair "
          "(assertion would have raised)")

    # ---- infeasibility by year ------------------------------------------
    inf = (d.groupby("year")
             .agg(n_pairs=("infeasible", "size"), n_infeasible=("infeasible", "sum"))
             .reset_index())
    inf["rate"] = inf.n_infeasible / inf.n_pairs
    miss = (d[d.infeasible == 1].groupby("year")["missing"]
              .apply(lambda s: s.value_counts().index[0] if len(s) else ""))
    inf = inf.merge(miss.rename("most_common_missing"), on="year", how="left")
    inf.to_csv(os.path.join(OUT, "cps_infeasibility.csv"), index=False)
    print("\ninfeasibility by year (top 8):")
    print(inf.sort_values("rate", ascending=False).head(8).round(4).to_string(index=False))

    # ---- national annual series -----------------------------------------
    comp = ["dT", "dW", "dX", "dE", "lnTFP"]
    nat = ok.groupby("year")[comp].mean().reset_index()
    wsum = ok.groupby("year").apply(
        lambda g: pd.Series({c: np.average(g[c], weights=g.y_t) for c in comp}))
    for c in comp:
        nat[c + "_w"] = wsum[c].values
    nat["n"] = ok.groupby("year").size().values
    nat.to_csv(os.path.join(OUT, "cps_national_annual.csv"), index=False)

    # ---- CPS Table 2/3-style summary ------------------------------------
    summ = pd.DataFrame({
        "component": ["dT (technology)", "dW (weather)", "dX (input/scale)",
                      "dE (efficiency/adaptation)", "ln TFP change"],
        "mean": [ok[c].mean() for c in comp],
        "sd": [ok[c].std() for c in comp],
        "mean_pct_yr": [100 * (np.exp(ok[c].mean()) - 1) for c in comp],
        "p10": [ok[c].quantile(.10) for c in comp],
        "p90": [ok[c].quantile(.90) for c in comp],
        "share_positive": [(ok[c] > 0).mean() for c in comp],
    })
    summ.to_csv(os.path.join(OUT, "cps_summary.csv"), index=False)
    print("\n=== CPS Table 2/3-style summary (county-pair level, logs) ===")
    print(summ.round(4).to_string(index=False))
    print("\nnational annual means (unweighted), first/last 5 years:")
    print(pd.concat([nat.head(5), nat.tail(5)])[["year", "n"] + comp].round(4).to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
