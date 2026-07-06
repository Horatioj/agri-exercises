# -*- coding: utf-8 -*-
"""
fig8-style TFP-by-year comparison: RAW vs CLEANED panel.

TFP index = year effects (gamma_t) from a two-way fixed-effects Cobb-Douglas
production function estimated in Stata (areg, absorb(countyid), i.year) on the
full county panel -- see 06_estimate_tfp_fe.do.  Index is normalised to base
year = 1.  Left panel: the index for raw vs cleaned with literature reference
growth lines.  Right panel: annual TFP growth (100*Delta gamma_t) as grouped
bars so the volatility reduction from cleaning is visible.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import CLEAN_DIR, FIG_DIR, ensure_dirs

ensure_dirs()

BASE_YEAR = 2005   # GVP deflated to 2005 Hebei prices -> index the TFP path to 2005


def load(tag):
    df = pd.read_csv(os.path.join(CLEAN_DIR, f"tfp_fe_{tag}.csv")).sort_values("year")
    df = df.reset_index(drop=True)
    g0 = df.loc[df["year"] == BASE_YEAR, "gamma"].iloc[0]
    df["lntfp"] = df["gamma"] - g0
    df["index"] = np.exp(df["lntfp"])
    df["growth"] = 100.0 * df["lntfp"].diff()
    return df


clean = load("cleaned")
raw = load("raw")

yrs = clean["year"].values
n_yr = yrs.max() - yrs.min()

# average log-growth (%/yr) over the whole window
g_clean = 100.0 * clean["lntfp"].iloc[-1] / n_yr
g_raw = 100.0 * raw["lntfp"].iloc[-1] / n_yr

# volatility of annual growth
vol_clean = np.nanstd(clean["growth"].values)
vol_raw = np.nanstd(raw["growth"].values)

C_CLEAN = "#1f77b4"
C_RAW = "#ff7f0e"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

# ---------------------------------------------------------------- left panel
axL.plot(yrs, clean["index"], "-o", ms=4, lw=2, color=C_CLEAN,
         label=f"Cleaned ({g_clean:.2f}%/yr)")
axL.plot(yrs, raw["index"], "-o", ms=4, lw=2, color=C_RAW,
         label=f"Raw ({g_raw:.2f}%/yr)")

t = yrs - BASE_YEAR
axL.plot(yrs, 1.03 ** t, "--", color="0.5", lw=1.2, label="lit. ref. 3%/yr")
axL.plot(yrs, 1.04 ** t, ":", color="0.5", lw=1.2, label="lit. ref. 4%/yr")

axL.axvspan(1997, 2001, color="red", alpha=0.07)
axL.annotate("late-90s\nslowdown", xy=(1999, 1.55), color="#1f3b73",
             fontsize=8, ha="center")
axL.axvline(2004, color="green", ls=":", lw=1)
axL.annotate("post-2004 access /\ntax abol. / subsidies", xy=(2004.2, 2.4),
             color="green", fontsize=8, ha="left")

axL.set_title("FE year-dummy TFP index (1981=1)")
axL.set_ylabel("TFP index")
axL.set_xlabel("year")
axL.legend(loc="upper left", fontsize=8, framealpha=0.9)
axL.grid(alpha=0.3)

# --------------------------------------------------------------- right panel
gy = clean["year"].values[1:]
w = 0.4
axR.bar(gy - w / 2, clean["growth"].values[1:], width=w, color=C_CLEAN,
        label=f"Cleaned  (sd {vol_clean:.1f})")
axR.bar(gy + w / 2, raw["growth"].values[1:], width=w, color=C_RAW, alpha=0.85,
        label=f"Raw  (sd {vol_raw:.1f})")
axR.axhline(3.0, ls="--", color="0.4", lw=1, label="3%/yr")
axR.axhline(0.0, color="0.2", lw=0.8)

axR.set_title("Annual TFP growth: raw vs cleaned (%/yr)")
axR.set_ylabel("% per year")
axR.set_xlabel("year")
axR.legend(loc="upper right", fontsize=8, framealpha=0.9)
axR.grid(alpha=0.3, axis="y")

fig.suptitle("China county agricultural TFP by year "
             "\u2014 county-FE Cobb\u2013Douglas (year effects), raw vs cleaned",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))

png = os.path.join(FIG_DIR, "fig_tfp_byyear_compare.png")
svg = os.path.join(FIG_DIR, "fig_tfp_byyear_compare.svg")
fig.savefig(png, dpi=300)
fig.savefig(svg)
plt.close(fig)

print(f"avg growth  cleaned={g_clean:.2f}%/yr  raw={g_raw:.2f}%/yr")
print(f"volatility  cleaned sd={vol_clean:.2f}  raw sd={vol_raw:.2f}")
print(f"saved: {png}")
print(f"saved: {svg}")
