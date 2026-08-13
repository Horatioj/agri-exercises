# -*- coding: utf-8 -*-
"""
Compare two TFP methodologies on the CLEANED panel, BOTH with all 4 inputs
(Labour, Land, Capital, Intermediate), single output real_gvp:

  (A) FE Solow residual  -- two-way (county+year) FE Cobb-Douglas elasticities, ln TFP = lnY - Σβ lnX, national
      mean by year.
  (B) DEA TFP            -- output-oriented, NIRS (Σλ<=1), free-disposal inputs,
      SEQUENTIAL (non-regressive) frontier = all obs up to year t.
      Sequential Malmquist, cumulative.  The EFFCH/TECHCH split of the same
      index is done in 23_dea_sfa_framework.do.

Both indexed to base year = 0 (log). One overlay figure: fig_tfp_dea_vs_fe.png.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scipy.optimize import linprog
from linearmodels.panel import PanelOLS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _common as C

OUT = "real_gvp"
INP = ["Laborday_impute", "Land_serv_q", "capital_serv_q", "Inter_all_real"]
EPS = 1e-9

# ----------------------------------------------------------------- (A) FE Solow
def fe_solow(d):
    df = d.copy()
    df["lY"] = np.log(df[OUT])
    for v in INP:
        df["l_" + v] = np.log(df[v])
    df = df.set_index(["countyid", "year"])
    X = df[["l_" + v for v in INP]]
    res = PanelOLS(df["lY"], X, entity_effects=True, time_effects=True).fit()
    b = res.params
    print("  FE elasticities:", {v: round(b["l_"+v], 3) for v in INP},
          "RTS=%.3f" % sum(b["l_"+v] for v in INP))
    resid = df["lY"] - sum(b["l_"+v] * df["l_"+v] for v in INP)
    tfp = resid.groupby("year").mean()
    tfp = tfp - tfp.iloc[0]
    return pd.DataFrame({"year": tfp.index, "lntfp": tfp.values}), b

# ----------------------------------------------------------------- (B) DEA NIRS
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
    county.to_csv(os.path.join(C.CLEAN_DIR, "dea_malmquist_county.csv"), index=False)
    return pd.DataFrame({"year": yrs, "lntfp": lvl})

def main():
    # the cleaned panel holds ALL resolved rural counties (reusable product);
    # the frontier is estimated on the cropland>=15% agricultural subset only
    panel = pd.read_csv(C.CLEAN_PANEL)
    n_all = panel.countyid.nunique()
    panel = panel[panel["ag_county"] == 1]
    d = panel.dropna(subset=[OUT] + INP).copy()
    d = d[(d[OUT] > 0) & (d[INP] > 0).all(axis=1)]
    print(f"sample: {d.countyid.nunique():,d} agricultural counties "
          f"(of {n_all:,d} in the cleaned panel), {len(d):,d} county-years")
    print("FE Solow residual (4 inputs):")
    fe, beta = fe_solow(d)
    print("DEA sequential NIRS (4 inputs):")
    dea = dea_sequential(d)
    fe.to_csv(os.path.join(C.CLEAN_DIR, "tfp_fe_solow_4in.csv"), index=False)
    dea.to_csv(os.path.join(C.CLEAN_DIR, "tfp_dea_nirs_4in.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for df, lab, col in [(fe, "FE Solow residual", "#2ca02c"), (dea, "DEA NIRS sequential", "#1f77b4")]:
        g = 100 * df["lntfp"].iloc[-1] / (df.year.iloc[-1] - df.year.iloc[0])
        ax.plot(df.year, 100 * df["lntfp"], "-o", ms=4, color=col, label=f"{lab} ({g:.2f}%/yr)")
    ax.axhline(0, color="k", lw=.6, ls=":")
    for yr in (1, 3): ax.plot(fe.year, 100*np.log(1+yr/100.0)*(fe.year-fe.year.iloc[0]),
                              "--", color="0.6", lw=1, label=f"lit. ref {yr}%/yr")
    ax.set_title("China agricultural TFP by year: DEA (NIRS, sequential) vs FE Solow residual\n"
                 "cleaned panel, all 4 inputs (L, Land, K, M), single output")
    ax.set_ylabel("ln TFP x 100 (~ % vs base)"); ax.set_xlabel("year")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig_tfp_dea_vs_fe.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print("saved:", p)

if __name__ == "__main__":
    main()
