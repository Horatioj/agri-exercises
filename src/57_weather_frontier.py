# -*- coding: utf-8 -*-
"""
Figure 9 and Figure 10 in INPUT-WEATHER space.

The conventional Fig 9/10 (53_frontier_isoquant_expansion.py) put a second
CONVENTIONAL input on the x axis.  Here the second axis is WEATHER, which is
what the Chambers-Pieralli framework actually treats as the extra argument of
the technology.  Two things change, and both matter:

1. NO FREE DISPOSABILITY.  A conventional input is freely disposable, so its
   boundary is made monotone (more input can never hurt).  Weather is WEAKLY
   disposable -- the equality constraint in the CPS programme -- and it is not
   chosen by the farmer.  Imposing monotonicity here would be wrong and would
   also destroy the shape we most want to see.

2. THE SHAPE IS EXPECTED TO BE HUMPED, NOT MONOTONE.  Too little heat and the
   crop does not develop; too much and it is damaged.  GDD should therefore
   trace an inverted U, and the soil-moisture anomaly likewise (drought on one
   side, waterlogging on the other).  A monotone envelope cannot represent that,
   so the boundary here is the raw within-bin quantile curve.

   This is also why the conventional Fig 10 comes out FLAT and this one need
   not.  There, the running-minimum free-disposal step meets an input
   requirement that RISES with labour, so it locks onto the leftmost bin and
   returns a constant.  No monotone step is applied here.

FIGURE 9W  y/L against weather, per year.
   Boundary = alpha-quantile of output per labour day within equal-count weather
   bins.  Reads as: the most output per worker achieved at this weather.

FIGURE 10W  input requirement against weather at fixed output y*, per year.
   Every county-year is radially scaled to y*, then the boundary is the
   (1-alpha)-quantile of the scaled Tornqvist aggregate input within weather
   bins.  Reads as: the least input needed to produce y* at this weather.
   A U-shape is the expected signature -- benign weather needs less input,
   extreme weather on either side needs more.  This object needs INPUT-WEATHER
   substitution, not factor-factor substitution, which is why it can be
   informative on a panel where the conventional isoquant is degenerate.

Captions are printed and written to captions_fig9w_10w.md, never drawn on the
figures.

Input : src/clean/io_weather_panel.csv  (12_county_gdd_hdd.py)
Output: src/figures/fig9w_frontier_{var}.png, fig10w_isoquant_{var}.png
        src/clean/descriptive/fig9w_*.csv, fig10w_*.csv, captions_fig9w_10w.md
Run:  python src/57_weather_frontier.py
      python src/57_weather_frontier.py --years 1986,2000,2015 --alpha 0.9
"""
from __future__ import annotations
import os, sys, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _common as C
import _tfp as T

OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)
CAPTIONS = {}

DEFAULT_YEARS = [1986, 1995, 2000, 2005, 2010, 2015]

# weather variates: column -> (short, axis label, whether a z-score)
WVARS = {
    "gdd_growing_season": ("GDD", "Growing-season GDD  (°C·day, beneficial heat 8–32 °C)", False),
    "hdd_growing_season": ("HDD", "Growing-season HDD  (°C·day, harmful heat > 32 °C)", False),
    "sm_0_28_z":          ("SMz", "Root-zone soil-moisture anomaly  (sd from the county-month normal)", True),
}


def load():
    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    need = ["real_gvp", "Laborday_impute", "Land_serv_q", "capital_serv_q",
            "Inter_all_real"] + list(WVARS)
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"io_weather_panel.csv lacks {miss} — re-run 12/13")
    d = d.dropna(subset=need)
    d = d[(d[need[:5]] > 0).all(axis=1)].copy()
    # Tornqvist aggregate input, CRS weights from the converged BC92 SFA
    b = T.BETA_AG
    s = sum(b.values())
    d["x"] = np.exp(sum((v / s) * np.log(d[k]) for k, v in b.items()))
    d["YL"] = d.real_gvp / d.Laborday_impute
    return d


def bin_quantile(w, v, alpha, nbins=14, minobs=40):
    """alpha-quantile of v within equal-count bins of w.

    NO monotone step.  Weather is weakly disposable and its effect is expected
    to be humped, so forcing the curve up or down would impose the answer.
    """
    w = np.asarray(w, float); v = np.asarray(v, float)
    ok = np.isfinite(w) & np.isfinite(v)
    w, v = w[ok], v[ok]
    if len(w) < nbins * minobs:
        nbins = max(4, len(w) // minobs)
    edges = np.unique(np.quantile(w, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(edges, w, side="right") - 1, 0, len(edges) - 2)
    bw, bv, bn = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < minobs:
            continue
        bw.append(np.median(w[m])); bv.append(np.quantile(v[m], alpha))
        bn.append(int(m.sum()))
    return np.array(bw), np.array(bv), np.array(bn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="upper quantile for Fig 9W (output frontier)")
    ap.add_argument("--iso-alpha", type=float, default=0.05,
                    help="lower quantile for Fig 10W (input requirement)")
    a = ap.parse_args()

    d = load()
    YEARS = [int(y) for y in a.years.split(",")]
    missing = [y for y in YEARS if y not in set(d.year)]
    if missing:
        raise SystemExit(f"years not in the panel: {missing}")
    ystar = d.real_gvp.median()
    cols = plt.cm.viridis(np.linspace(0, .88, len(YEARS)))
    ny = d[d.year.isin(YEARS)].groupby("year").countyid.nunique()
    print(f"{len(d):,d} county-years, {d.countyid.nunique():,d} counties")
    print("years:", ", ".join(f"{y} (n={ny[y]:,d})" for y in YEARS))
    print(f"y* = {ystar:,.0f} (2005 prices)\n")

    for wcol, (short, wlab, is_z) in WVARS.items():
        # ---------------------------------------------------- FIGURE 9W
        fig, ax = plt.subplots(figsize=(9.5, 6.6))
        rows = []
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            ax.scatter(s[wcol], s.YL, s=8, alpha=.20, color=col, linewidths=0,
                       rasterized=True)
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            bw, bv, bn = bin_quantile(s[wcol].values, s.YL.values, a.alpha)
            if len(bw) < 3:
                print(f"  [{short}] {yr}: too few bins, skipped", file=sys.stderr)
                continue
            ax.plot(bw, bv, "-o", color=col, lw=2.2, ms=4,
                    label=f"{yr}  (n={len(s):,d})")
            rows.append(pd.DataFrame({"year": yr, "wvar": short, "w": bw,
                                      "f_YL": bv, "n_bin": bn}))
            k = int(np.argmax(bv))
            print(f"  [{short}] {yr}: peak Y/L at {short}={bw[k]:,.3g} "
                  f"(bin {k+1}/{len(bw)}) -> {'INTERIOR (humped)' if 0 < k < len(bw)-1 else 'at an endpoint (monotone)'}")
        pd.concat(rows).to_csv(os.path.join(OUT, f"fig9w_frontier_{short}.csv"), index=False)
        ax.set_yscale("log")
        ax.set_xlabel(wlab); ax.set_ylabel("Y/L — real output per labour day (log)")
        ax.set_title(f"Figure 9W. Output frontier in weather space — {short}\n"
                     f"order-{chr(945)} quantile ({chr(945)}={a.alpha:g}) within weather bins; "
                     "no monotone step (weather is weakly disposable)", fontsize=11)
        ax.legend(fontsize=8, title="year"); ax.grid(alpha=.25, which="both")
        if is_z:
            ax.axvline(0, color="0.5", lw=.8, ls=":")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, f"fig9w_frontier_{short}.png"), dpi=170)
        plt.close(fig)
        print(f"  saved fig9w_frontier_{short}.png")
        CAPTIONS[f"fig9w_frontier_{short}"] = (
            f"Figure 9W ({short}). Output per labour day against {wlab.lower()}, "
            f"{', '.join(map(str, YEARS))}. Source: cleaned county agricultural panel "
            f"joined to ERA5/CACD weather over the cropland-NDVI growing season; "
            f"{d.countyid.nunique():,d} agricultural counties. A separate boundary is "
            "computed for each year using only that year's observations. The boundary "
            f"is the {a.alpha:.0%} quantile of Y/L within equal-count weather bins. "
            "Unlike the conventional Figure 9 it is NOT made monotone: weather is "
            "weakly disposable in the Chambers-Pieralli technology and is not chosen by "
            "the farmer, and its effect is expected to be humped rather than "
            "monotone -- too little heat and the crop does not develop, too much and it "
            "is damaged. An interior peak is therefore the meaningful signature, and "
            "the printed diagnostic reports whether each year's peak is interior. "
            "Output is on a log scale; weather is on its natural scale. Points are "
            "county-years.")

        # ---------------------------------------------------- FIGURE 10W
        fig, ax = plt.subplots(figsize=(9.5, 6.6))
        rows = []
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            f_ = ystar / s.real_gvp
            ax.scatter(s[wcol], s.x * f_, s=8, alpha=.20, color=col,
                       linewidths=0, rasterized=True)
        for yr, col in zip(YEARS, cols):
            s = d[d.year == yr]
            xs = (s.x * (ystar / s.real_gvp)).values
            bw, bv, bn = bin_quantile(s[wcol].values, xs, a.iso_alpha)
            if len(bw) < 3:
                print(f"  [{short}] {yr}: too few bins, skipped", file=sys.stderr)
                continue
            ax.plot(bw, bv, "-o", color=col, lw=2.2, ms=4,
                    label=f"{yr}  (n={len(s):,d})")
            rows.append(pd.DataFrame({"year": yr, "wvar": short, "w": bw,
                                      "x_min": bv, "n_bin": bn}))
            k = int(np.argmin(bv))
            print(f"  [{short}] {yr}: least input at {short}={bw[k]:,.3g} "
                  f"(bin {k+1}/{len(bw)}) -> {'INTERIOR (U-shaped)' if 0 < k < len(bw)-1 else 'at an endpoint'}")
        pd.concat(rows).to_csv(os.path.join(OUT, f"fig10w_isoquant_{short}.csv"), index=False)
        ax.set_yscale("log")
        ax.set_xlabel(wlab)
        ax.set_ylabel(f"Törnqvist aggregate input needed for y* (log)")
        ax.set_title(f"Figure 10W. Input requirement in weather space at y* — {short}\n"
                     f"{a.iso_alpha:.0%} quantile of y*-scaled input within weather bins; "
                     "weak disposability, no flat extension", fontsize=11)
        ax.legend(fontsize=8, title="year"); ax.grid(alpha=.25, which="both")
        if is_z:
            ax.axvline(0, color="0.5", lw=.8, ls=":")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIG_DIR, f"fig10w_isoquant_{short}.png"), dpi=170)
        plt.close(fig)
        print(f"  saved fig10w_isoquant_{short}.png\n")
        CAPTIONS[f"fig10w_isoquant_{short}"] = (
            f"Figure 10W ({short}). Input requirement at y* = median real GVP "
            f"({ystar:,.0f}, 2005 prices) against {wlab.lower()}, "
            f"{', '.join(map(str, YEARS))}. Every county-year is radially scaled to y* "
            "(Farrell normalisation) and the boundary is the "
            f"{a.iso_alpha:.0%} quantile of the y*-scaled Törnqvist aggregate input "
            "within equal-count weather bins, separately for each year. The aggregate "
            "input uses the CRS weights from the converged BC92 SFA. Reads as: the "
            "least input needed to produce y* at this weather. A U-shape is the "
            "expected signature -- benign weather needs less input, extreme weather on "
            "either side needs more -- and the printed diagnostic reports whether each "
            "year's minimum is interior. Weather is weakly disposable, so unlike the "
            "conventional unit isoquant the boundary carries no flat free-disposal "
            "extension. Note this object requires INPUT-WEATHER substitution, not "
            "factor-factor substitution, which is why it can be informative on a panel "
            "where the conventional labour-capital isoquant is degenerate. Input is on "
            "a log scale; weather is on its natural scale.")

    cap_md = os.path.join(OUT, "captions_fig9w_10w.md")
    with open(cap_md, "w", encoding="utf-8") as fh:
        fh.write("# Figure captions - Fig 9W / 10W (input-weather space)\n\n")
        for k in sorted(CAPTIONS):
            fh.write(f"## {k}\n\n{CAPTIONS[k]}\n\n")
    print("=" * 78)
    print("FIGURE CAPTIONS (also written to captions_fig9w_10w.md)")
    print("=" * 78)
    for k in sorted(CAPTIONS):
        print(f"\n[{k}]\n{CAPTIONS[k]}")
    print(f"\ncaptions -> {cap_md}")


if __name__ == "__main__":
    main()
