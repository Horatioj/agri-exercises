# -*- coding: utf-8 -*-
"""
DEA Malmquist TFP index by year (output-oriented, CRS), RAW vs CLEANED.
Output = real_gvp; inputs = Labour, Capital, Intermediate (3, matching the BC92 spec).

Malmquist (Fare et al. 1994), Farrell output expansion phi^s(a) = max phi s.t.
(x_a, phi*y_a) feasible in period-s technology:
    M_i = sqrt[ (phi^t(t)/phi^t(t+1)) * (phi^{t+1}(t)/phi^{t+1}(t+1)) ]   (>1 = TFP up)
Aggregate annual TFP growth = mean_i ln M_i (geometric mean); cumulative index,
base year = 0 (log).  Plotted like fig_tfp_byyear.png.

Speed: per year the reference set is pruned to free-disposal NON-DOMINATED DMUs
(dominated points can never enter the optimal basis), so the per-DMU LPs are small.
CAVEAT printed: CRS efficiency LEVELS are very low here (frontier = a few extreme
counties) -- the *change* index is robust, the levels are not.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

OUT = "real_gvp"
INP_CLEAN = ["Laborday_impute", "capital_serv_q", "Inter_all_real"]
EPS = 1e-9

def nondominated(X, Y):
    """indices of free-disposal non-dominated DMUs (superset of the CRS frontier)."""
    n = len(Y); keep = np.ones(n, bool)
    order = np.argsort(-Y)                       # high output first
    for a in range(n):
        i = order[a]
        if not keep[i]:
            continue
        # j dominated by i if Y_i>=Y_j and X_i<=X_j (i at least as good)
        dom = (Y <= Y[i] + EPS) & np.all(X >= X[i] - EPS, axis=1)
        dom[i] = False
        keep[dom] = False
    return np.where(keep)[0]

def phi(xo, yo, Xref, Yref):
    n = len(Yref); K = Xref.shape[1]
    c = np.zeros(n + 1); c[-1] = -1.0
    A = np.zeros((K + 1, n + 1)); b = np.zeros(K + 1)
    A[0, :n] = -Yref; A[0, -1] = yo; b[0] = 0.0
    A[1:, :n] = Xref.T; b[1:] = xo
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * n + [(0, None)], method="highs")
    return (-r.fun) if r.success else np.nan

def phi_vec(Xeval, Yeval, Xref, Yref):
    return np.array([phi(Xeval[i], Yeval[i], Xref, Yref) for i in range(len(Yeval))])

def run_arm(panel, out_col, inp_cols, tag):
    d = panel.dropna(subset=[out_col] + inp_cols).copy()
    d = d[(d[out_col] > 0) & (d[inp_cols] > 0).all(axis=1)]
    years = sorted(d.year.unique())
    # per year: arrays + non-dominated reference
    YR = {}
    for y in years:
        g = d[d.year == y]
        X = g[inp_cols].to_numpy(float); Y = g[out_col].to_numpy(float)
        cid = g.countyid.to_numpy()
        ref = nondominated(X, Y)
        YR[y] = dict(cid=cid, X=X, Y=Y, Xref=X[ref], Yref=Y[ref], idx={c: i for i, c in enumerate(cid)})
    # own-period phi
    t0 = time.time()
    for y in years:
        s = YR[y]
        s["phi_own"] = phi_vec(s["X"], s["Y"], s["Xref"], s["Yref"])
        print(f"  [{tag}] {y} own phi done  ({time.time()-t0:.0f}s)", flush=True)
    # Malmquist over transitions
    growth = {}
    for a, b in zip(years[:-1], years[1:]):
        if b != a + 1:
            continue
        sa, sb = YR[a], YR[b]
        common = [c for c in sa["cid"] if c in sb["idx"]]
        ia = np.array([sa["idx"][c] for c in common]); ib = np.array([sb["idx"][c] for c in common])
        # cross: phi^a(b) = period-b obs on period-a tech; phi^b(a) = period-a obs on period-b tech
        phi_a_b = phi_vec(sb["X"][ib], sb["Y"][ib], sa["Xref"], sa["Yref"])
        phi_b_a = phi_vec(sa["X"][ia], sa["Y"][ia], sb["Xref"], sb["Yref"])
        phi_a_a = sa["phi_own"][ia]; phi_b_b = sb["phi_own"][ib]
        M = np.sqrt((phi_a_a / phi_a_b) * (phi_b_a / phi_b_b))
        lnM = np.log(M)
        lnM = lnM[np.isfinite(lnM)]
        growth[b] = np.mean(lnM)         # aggregate log TFP growth a->b
        print(f"  [{tag}] {a}->{b} Malmquist: {len(lnM)} cty, dlnTFP={growth[b]*100:+.2f}%", flush=True)
    # cumulative index, base = first year = 0
    yrs = [years[0]] + [b for b in years[1:] if b in growth]
    lvl = [0.0]
    for b in yrs[1:]:
        lvl.append(lvl[-1] + growth[b])
    out = pd.DataFrame({"year": yrs, "lntfp": lvl})
    out.to_csv(os.path.join(C.CLEAN_DIR, f"dea_tfp_{tag}.csv"), index=False)
    # report level caveat
    me = np.nanmean(1.0 / YR[years[len(years)//2]]["phi_own"])
    print(f"[{tag}] mean own-period efficiency (mid-year) = {me:.3f}  (LEVELS unreliable; change index is the output)")
    return out

def main():
    panel = pd.read_csv(C.CLEAN_PANEL)
    clean = run_arm(panel.rename(columns={c: c for c in INP_CLEAN}), OUT, INP_CLEAN, "cleaned")
    raw_cols = [c + "_raw" for c in INP_CLEAN]
    rawp = panel.rename(columns={OUT + "_raw": "__y"})
    raw = run_arm(panel.assign(**{OUT: panel[OUT + "_raw"]}), OUT, raw_cols, "raw")

    # plot raw vs cleaned (style of fig_tfp_byyear.png)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 5))
    for df, lab, col in [(raw, "raw", "0.6"), (clean, "cleaned", "#2ca02c")]:
        g = 100 * df["lntfp"].iloc[-1] / (df.year.iloc[-1] - df.year.iloc[0])
        ax.plot(df.year, 100 * df["lntfp"], "-o", ms=4, color=col, label=f"{lab} ({g:.2f}%/yr)")
    ax.axhline(0, color="k", lw=.6, ls=":")
    ax.set_title("DEA Malmquist agricultural TFP index (CRS, output-oriented; base year=0)\n"
                 "raw vs cleaned  —  3 inputs (L,K,M), single output")
    ax.set_ylabel("ln TFP x 100 (~ % vs base)"); ax.set_xlabel("year")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig_dea_tfp_byyear.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("saved:", p)

if __name__ == "__main__":
    main()
