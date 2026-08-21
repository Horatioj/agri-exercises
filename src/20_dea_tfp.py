# -*- coding: utf-8 -*-
"""
DEA TFP on the CLEANED panel: all 4 inputs (Labour, Land, Capital, Intermediate),
single output real_gvp.

  Output-oriented, NIRS (Σλ <= 1), free-disposal inputs, SEQUENTIAL
  (non-regressive) frontier = all observations up to year t.  Sequential
  Malmquist, cumulated to a level index (base year = 0 in logs).  The
  EFFCH/TECHCH split of the same index is also written per county-year.

The parametric counterpart is NOT computed here.  It is 21_sfa_bc92.do, which
estimates the same technology by BC92 stochastic frontier under the SAME
returns-to-scale assumption (NIRS) in both Cobb-Douglas and translog form; the
two are brought together in 23_dea_sfa_framework.do.  Keeping the nonparametric
and parametric estimates in separate scripts stops a fast reduced-form residual
from being read as the parametric frontier.

Outputs: src/clean/tfp_dea_nirs_4in.csv, src/clean/dea_malmquist_county.csv,
         src/figures/fig_tfp_dea.png
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

TAG = {"v": ""}
OUT = "real_gvp"
INP = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
EPS = 1e-9

# ----------------------------------------------------------------- DEA NIRS
def nondominated(X, Y):
    n = len(Y); keep = np.ones(n, bool)
    for i in np.argsort(-Y):
        if not keep[i]:
            continue
        dom = (Y <= Y[i] + EPS) & np.all(X >= X[i] - EPS, axis=1)
        dom[i] = False
        keep[dom] = False
    return np.where(keep)[0]

def fmax(xo, Xref, Yref):
    """max lambda'Y s.t. Xref@lambda<=xo, sum lambda<=1 (NIRS), lambda>=0."""
    n = len(Yref); K = Xref.shape[1]
    c = -Yref
    A = np.vstack([Xref.T, np.ones((1, n))])
    b = np.append(xo, 1.0)
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
    return (-r.fun) if r.success else np.nan

def fmax_vec(Xeval, Xref, Yref):
    return np.array([fmax(Xeval[i], Xref, Yref) for i in range(len(Xeval))])

def dea_sequential(d):
    d = d.dropna(subset=[OUT] + INP).copy()
    d = d[(d[OUT] > 0) & (d[INP] > 0).all(axis=1)]
    years = sorted(d.year.unique())
    S = {}
    cumX = cumY = None
    t0 = time.time()
    for y in years:
        g = d[d.year == y]
        X = g[INP].to_numpy(float); Y = g[OUT].to_numpy(float)
        cid = g.countyid.to_numpy()
        cumX = X if cumX is None else np.vstack([cumX, X])
        cumY = Y if cumY is None else np.concatenate([cumY, Y])
        nd = nondominated(cumX, cumY)                 # sequential ref up to y (pruned)
        cumX, cumY = cumX[nd], cumY[nd]               # keep only non-dominated (frontier candidates)
        S[y] = dict(cid=cid, X=X, Y=Y, Rx=cumX.copy(), Ry=cumY.copy(),
                    idx={c: i for i, c in enumerate(cid)})
        # own-period frontier value f_y(x_y)
        S[y]["f_own"] = fmax_vec(X, cumX, cumY)
        print(f"  [DEA] {y}: ref={len(cumY)} pts ({time.time()-t0:.0f}s)", flush=True)
    growth = {}
    cyrows = []            # per-county Malmquist + decomposition (year=b)
    for a, b in zip(years[:-1], years[1:]):
        if b != a + 1:
            continue
        sa, sb = S[a], S[b]
        common = [c for c in sa["cid"] if c in sb["idx"]]
        ia = np.array([sa["idx"][c] for c in common]); ib = np.array([sb["idx"][c] for c in common])
        ya = sa["Y"][ia]; yb = sb["Y"][ib]
        f_a_a = sa["f_own"][ia]                       # f_a(x_a)
        f_b_b = sb["f_own"][ib]                       # f_b(x_b)
        f_a_b = fmax_vec(sb["X"][ib], sa["Rx"], sa["Ry"])   # f_a(x_b)
        f_b_a = fmax_vec(sa["X"][ia], sb["Rx"], sb["Ry"])   # f_b(x_a)
        # output distance D^s(x,y) = y / f_s(x)  (<=1 = efficiency)
        # M = EFFCH x TECHCH  (Fare et al. 1994)
        #   EFFCH  = D^b(x_b,y_b)/D^a(x_a,y_a)                    catch-up
        #   TECHCH = sqrt[ (f_b_b/f_a_b) * (f_b_a/f_a_a) ]        frontier shift
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.sqrt(((f_a_a/ya)/(f_a_b/yb)) * ((f_b_a/ya)/(f_b_b/yb)))
            EC = (yb / f_b_b) / (ya / f_a_a)
            TC = np.sqrt((f_b_b / f_a_b) * (f_b_a / f_a_a))
            lnM_all, lnEC, lnTC = np.log(M), np.log(EC), np.log(TC)
            eff_a, eff_b = ya / f_a_a, yb / f_b_b       # efficiency levels
        fin = np.isfinite(lnM_all) & np.isfinite(lnEC) & np.isfinite(lnTC)
        growth[b] = np.mean(lnM_all[fin])
        for k in np.where(fin)[0]:
            cyrows.append((b, int(common[k]), float(lnM_all[k]), float(lnEC[k]),
                           float(lnTC[k]), float(eff_a[k]), float(eff_b[k])))
        print(f"  [DEA] {a}->{b}: {int(fin.sum())} cty, dlnTFP={growth[b]*100:+.2f}% "
              f"(EC {np.mean(lnEC[fin])*100:+.2f}, TC {np.mean(lnTC[fin])*100:+.2f})", flush=True)
    yrs = [years[0]] + [b for b in years[1:] if b in growth]
    lvl = [0.0]
    for b in yrs[1:]:
        lvl.append(lvl[-1] + growth[b])
    county = pd.DataFrame(cyrows, columns=["year", "countyid", "lnM", "lnEC", "lnTC",
                                           "eff_prev", "eff_now"])
    county.to_csv(os.path.join(C.CLEAN_DIR, TAG["v"] + "dea_malmquist_county.csv"),
                  index=False)
    return pd.DataFrame({"year": yrs, "lntfp": lvl})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=1985,
                    help="capital_serv_q behaves differently before 1985 and "
                         "county coverage nearly doubles at it -- see "
                         "52_frontier_figs.py")
    ap.add_argument("--exclude-counties", default=None,
                    help="CSV with a countyid column; dropped from the SAMPLE, "
                         "hence from the sequential DEA reference set, so they "
                         "can no longer define any year's frontier")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    a = ap.parse_args()

    # the cleaned panel holds ALL resolved rural counties (reusable product);
    # the frontier is estimated on the cropland>=15% agricultural subset only
    panel = pd.read_csv(C.CLEAN_PANEL)
    n_all = panel.countyid.nunique()
    panel = panel[panel["ag_county"] == 1]
    panel = panel[panel.year >= a.from_year]
    if a.exclude_counties:
        ex = pd.read_csv(a.exclude_counties)
        ex = set(pd.to_numeric(ex["countyid"], errors="coerce").dropna().astype(int))
        nb = panel.countyid.nunique()
        panel = panel[~panel.countyid.isin(ex)]
        print(f"excluded {len(ex)} counties: {nb:,d} -> "
              f"{panel.countyid.nunique():,d} counties")
    d = panel.dropna(subset=[OUT] + INP).copy()
    d = d[(d[OUT] > 0) & (d[INP] > 0).all(axis=1)]
    print(f"sample: {d.countyid.nunique():,d} agricultural counties "
          f"(of {n_all:,d} in the cleaned panel), {len(d):,d} county-years")
    TAG["v"] = a.tag.lstrip("_") + "_" if a.tag else ""
    print("DEA sequential NIRS (4 inputs):")
    dea = dea_sequential(d)
    dea.to_csv(os.path.join(C.CLEAN_DIR, f"tfp_dea_nirs_4in{a.tag}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    g = 100 * dea["lntfp"].iloc[-1] / (dea.year.iloc[-1] - dea.year.iloc[0])
    ax.plot(dea.year, 100 * dea["lntfp"], "-o", ms=4, color="#1f77b4",
            label=f"DEA NIRS sequential ({g:.2f}%/yr)")
    ax.axhline(0, color="k", lw=.6, ls=":")
    for yr in (1, 3):
        ax.plot(dea.year, 100*np.log(1+yr/100.0)*(dea.year-dea.year.iloc[0]),
                "--", color="0.6", lw=1, label=f"lit. ref {yr}%/yr")
    ax.set_title("China agricultural TFP by year: DEA (NIRS, sequential)\n"
                 "cleaned panel, all 4 inputs (L, Land, K, M), single output")
    ax.set_ylabel("ln TFP x 100 (~ % vs base)"); ax.set_xlabel("year")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"fig_tfp_dea{a.tag}.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("saved:", p)

if __name__ == "__main__":
    main()
