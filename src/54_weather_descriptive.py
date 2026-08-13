# -*- coding: utf-8 -*-
"""
DESCRIPTIVE weather figure — motivational only.  No DEA, no LP, no frontier.

Small-multiple scatter of county-year points, coloured by period:
    rows    = outcome on the y-axis:  K/L  and  Y/L   (both produced; pick one)
    columns = weather on the x-axis:  growing-season GDD, root-zone soil
              moisture (0-28 cm)

Period-specific fits are overlaid PURELY as a visual aid.  They are descriptive
OLS lines through a raw scatter with no controls, no fixed effects and no
identification strategy -- they do not imply a causal or frontier relationship,
and are not the estimates used anywhere in the analysis.

OUTPUT  src/figures/fig_weather_descriptive_{P5,P6}.png
        src/clean/descriptive/weather_scatter_fits_{P5,P6}.csv  (fit lines)
        src/clean/descriptive/weather_binned_{P5,P6}.csv        (binned medians)
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

C.set_cjk_font(plt)
OUT = os.path.join(C.CLEAN_DIR, "descriptive")
os.makedirs(OUT, exist_ok=True)

PERIODS5, PERIODS6 = T.PERIODS5, T.PERIODS6      # defined once in _tfp.py
WEATHER = [("gdd_growing_season", "growing-season GDD (°C·day)"),
           ("sm_0_28", "root-zone soil moisture, 0–28 cm (m³/m³)")]
OUTCOMES = [("KL", "K/L — capital service per labour day"),
            ("YL", "Y/L — real output per labour day")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", choices=["5", "6", "both"], default="both")
    a = ap.parse_args()

    d = pd.read_csv(os.path.join(C.CLEAN_DIR, "io_weather_panel.csv"))
    v = ["real_gvp", "Laborday_impute", "capital_serv_q"] + [w for w, _ in WEATHER]
    d = d.dropna(subset=v)
    d = d[(d[["real_gvp", "Laborday_impute", "capital_serv_q"]] > 0).all(axis=1)].copy()
    d["KL"] = d.capital_serv_q / d.Laborday_impute
    d["YL"] = d.real_gvp / d.Laborday_impute
    print(f"{len(d):,d} county-years with output, inputs and both weather variates")

    schemes = ({"5": [("P5", PERIODS5)], "6": [("P6", PERIODS6)],
                "both": [("P5", PERIODS5), ("P6", PERIODS6)]})[a.periods]

    for tag, PER in schemes:
        cols = plt.cm.viridis(np.linspace(0, .88, len(PER)))
        fig, axes = plt.subplots(len(OUTCOMES), len(WEATHER),
                                 figsize=(13.5, 9.5), sharex="col")
        fits, binned = [], []
        for i, (oc, oclab) in enumerate(OUTCOMES):
            for j, (wv, wlab) in enumerate(WEATHER):
                ax = axes[i, j]
                lo, hi = d[wv].quantile([.005, .995])
                for (p, col) in zip(PER, cols):
                    s = d[d.year.between(*p) & d[wv].between(lo, hi)]
                    s = s[s[oc] > 0]
                    if len(s) < 50:
                        continue
                    ax.scatter(s[wv], s[oc], s=2.5, alpha=.06, color=col, linewidths=0)
                    # descriptive fit in log(outcome) -- outcome is strongly skewed
                    b1, b0 = np.polyfit(s[wv], np.log(s[oc]), 1)
                    xs = np.linspace(s[wv].quantile(.02), s[wv].quantile(.98), 60)
                    ys = np.exp(b0 + b1 * xs)
                    ax.plot(xs, ys, "-", color=col, lw=2.2,
                            label=f"{p[0]}–{p[1]}" if (i == 0 and j == 0) else None)
                    fits.append(pd.DataFrame({"scheme": tag, "period": f"{p[0]}–{p[1]}",
                                              "outcome": oc, "weather": wv,
                                              "x": xs, "fit": ys,
                                              "slope_log_per_unit": b1}))
                    q = pd.qcut(s[wv], 12, duplicates="drop")
                    bm = s.groupby(q, observed=True).agg(
                        x=(wv, "median"), y=(oc, "median"), n=(oc, "size")).reset_index(drop=True)
                    binned.append(bm.assign(scheme=tag, period=f"{p[0]}–{p[1]}",
                                            outcome=oc, weather=wv))
                ax.set_yscale("log")
                ax.set_xlim(lo, hi)
                if i == len(OUTCOMES) - 1:
                    ax.set_xlabel(wlab, fontsize=9)
                if j == 0:
                    ax.set_ylabel(oclab, fontsize=9)
                ax.grid(alpha=.22, which="both")
        axes[0, 0].legend(fontsize=8, title="period", loc="upper left")
        fig.suptitle("Descriptive: input intensity and labour productivity against weather, "
                     f"by period ({tag})\n"
                     "raw county-year scatter with period-specific descriptive OLS fits — "
                     "no controls, no fixed effects, no causal or frontier interpretation",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p_ = os.path.join(C.FIG_DIR, f"fig_weather_descriptive_{tag}.png")
        fig.savefig(p_, dpi=170); plt.close(fig)
        pd.concat(fits).to_csv(os.path.join(OUT, f"weather_scatter_fits_{tag}.csv"), index=False)
        pd.concat(binned).to_csv(os.path.join(OUT, f"weather_binned_{tag}.csv"), index=False)
        print(f"  saved {os.path.basename(p_)}")

        sl = (pd.concat(fits).groupby(["outcome", "weather", "period"])
                .slope_log_per_unit.first().reset_index())
        print(f"\n  descriptive slopes, d ln(outcome) / d weather  ({tag}):")
        print(sl.pivot(index=["outcome", "weather"], columns="period",
                       values="slope_log_per_unit").round(5).to_string())
    print(f"\nCSVs -> {OUT}")


if __name__ == "__main__":
    main()
