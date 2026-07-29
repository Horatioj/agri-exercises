# -*- coding: utf-8 -*-
"""
Degree days the Ortiz-Bobea / Schlenker-Roberts way (single-sine method),
plus a check against the daily-MEAN method currently used in b_step_gdd.py.

WHY THE METHOD MATTERS
----------------------
b_step_gdd.py computes one number per day from the daily MEAN:
      tmean = (tmax+tmin)/2 ;  gdd = clip(tmean-8, 0, 24)
That throws away the within-day temperature path.  A day with tmin=5, tmax=35
has tmean=20 and is scored 12 "good" degree days and ZERO harmful exposure --
even though the crop actually spent several hours above 32 C.  Extreme heat is
exactly the signal HDD is meant to capture, so it must not be averaged away.

Ortiz-Bobea (and Schlenker & Roberts 2009, following Snyder 1985) instead
reconstruct the within-day path.  Temperature over the day is modelled as a sine
wave running between tmin and tmax,
      T(t) = M + A*sin(t),      M = (tmax+tmin)/2,  A = (tmax-tmin)/2
and the time spent above a threshold is integrated analytically.

Degree days above threshold b on one day:
      tmax <= b            ->  0
      tmin >= b            ->  M - b
      tmin <  b < tmax     ->  (1/pi) * [ (M-b)*(pi/2 - theta) + A*cos(theta) ],
                               theta = arcsin( (b-M)/A )

From that single primitive both variables follow, with bounds (8 C, 32 C):
      GDD (beneficial, 8-32 C) = DD(8) - DD(32)
      HDD (harmful,   > 32 C)  = DD(32)

Note GDD is DD(8) MINUS DD(32): degrees above 32 are moved OUT of the growth
measure and INTO the harm measure, instead of being capped and silently kept.

USAGE
-----
    from degree_days import dd_above, gdd_hdd
    gdd, hdd = gdd_hdd(tmin, tmax, base=8.0, cap=32.0)   # arrays, degrees C

Run this file directly to validate against real temperature rasters:
    python degree_days.py --zip "Z:/weather data/temperature" --year 1982
"""
from __future__ import annotations
import numpy as np

BASE_T = 8.0
CAP_T = 32.0


def dd_above(tmin, tmax, b):
    """Degree days above threshold `b` (deg C * day), single-sine method.

    tmin/tmax: arrays of daily min/max temperature in deg C (NaN allowed).
    Returns an array of the same shape; NaN where either input is NaN.
    """
    tmin = np.asarray(tmin, dtype="float64")
    tmax = np.asarray(tmax, dtype="float64")
    M = (tmax + tmin) / 2.0
    A = (tmax - tmin) / 2.0

    with np.errstate(invalid="ignore", divide="ignore"):
        # clip guards against tiny numerical overshoot outside [-1, 1]
        ratio = np.clip(np.divide(b - M, A, out=np.zeros_like(M), where=A > 0), -1.0, 1.0)
        theta = np.arcsin(ratio)
        partial = (1.0 / np.pi) * ((M - b) * (np.pi / 2.0 - theta) + A * np.cos(theta))

    out = np.where(tmax <= b, 0.0,                    # never reaches threshold
          np.where(tmin >= b, M - b,                  # whole day above threshold
                   partial))                          # crosses the threshold
    # A == 0 (tmin == tmax): degenerate, the sine collapses to a constant
    out = np.where(A <= 0, np.maximum(M - b, 0.0), out)
    out = np.where(np.isnan(tmin) | np.isnan(tmax), np.nan, out)
    return out.astype("float32")


def gdd_hdd(tmin, tmax, base: float = BASE_T, cap: float = CAP_T):
    """Return (GDD between base and cap, HDD above cap) for one day."""
    dd_base = dd_above(tmin, tmax, base)
    dd_cap = dd_above(tmin, tmax, cap)
    return (dd_base - dd_cap).astype("float32"), dd_cap.astype("float32")


def gdd_daily_mean(tmin, tmax, base: float = BASE_T, cap: float = CAP_T):
    """The CURRENT b_step_gdd.py method, for comparison: piecewise on the mean."""
    tmean = (np.asarray(tmax, "float64") + np.asarray(tmin, "float64")) / 2.0
    out = np.where(tmean < base, 0.0, np.where(tmean > cap, cap - base, tmean - base))
    return np.where(np.isnan(tmean), np.nan, out).astype("float32")


# --------------------------------------------------------------------------
def _validate(zip_root: str, year: int, sample_days: int = 24):
    """Compare the two methods on real rasters, and show the HDD that the
    daily-mean method cannot see."""
    import zipfile, tempfile, os, re
    import rasterio

    zmax = os.path.join(zip_root, f"{year}_max.zip")
    zmin = os.path.join(zip_root, f"{year}_min.zip")
    tmp = tempfile.mkdtemp(prefix="dd_val_")

    def read_day(zpath, ymd, kind):
        with zipfile.ZipFile(zpath) as z:
            name = f"{ymd}_{kind}.tif"
            z.extract(name, tmp)
        with rasterio.open(os.path.join(tmp, name)) as s:
            a = s.read(1).astype("float32")
        return np.where(a > -1e37, a, np.nan)

    with zipfile.ZipFile(zmax) as z:
        days = sorted({re.match(r"(\d{8})_max\.tif", n).group(1)
                       for n in z.namelist() if n.endswith("_max.tif")})
    step = max(1, len(days) // sample_days)
    days = days[::step][:sample_days]

    tot_g_sine = tot_g_mean = tot_h = 0.0
    n_px = 0
    hot_days = 0
    print(f"validating {len(days)} days of {year}  (grid 0.1deg, China)")
    print(f"{'date':>10} {'GDDsine':>9} {'GDDmean':>9} {'diff%':>7} {'HDD':>8} {'px>32C':>8}")
    for ymd in days:
        tmax = read_day(zmax, ymd, "max")
        tmin = read_day(zmin, ymd, "min")
        g_s, h_s = gdd_hdd(tmin, tmax)
        g_m = gdd_daily_mean(tmin, tmax)
        ok = ~np.isnan(g_s)
        gs, gm, hs = np.nanmean(g_s[ok]), np.nanmean(g_m[ok]), np.nanmean(h_s[ok])
        nhot = int(np.nansum(tmax[ok] > CAP_T))
        if nhot:
            hot_days += 1
        d = 100 * (gs - gm) / gm if gm > 0 else np.nan
        print(f"{ymd:>10} {gs:9.3f} {gm:9.3f} {d:7.1f} {hs:8.3f} {nhot:8,d}")
        tot_g_sine += gs; tot_g_mean += gm; tot_h += hs; n_px = int(ok.sum())

    n = len(days)
    print(f"\nmean over sampled days: GDD sine {tot_g_sine/n:.3f} | "
          f"GDD daily-mean {tot_g_mean/n:.3f} "
          f"({100*(tot_g_sine-tot_g_mean)/tot_g_mean:+.1f}%)")
    print(f"mean HDD (>32C) per day  : {tot_h/n:.3f}  "
          f"-- the daily-mean method reports this as 0 by construction")
    print(f"days in sample with any pixel above 32C: {hot_days}/{n}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=r"Z:/weather data/temperature")
    ap.add_argument("--year", type=int, default=1982)
    ap.add_argument("--days", type=int, default=24)
    a = ap.parse_args()
    _validate(a.zip, a.year, a.days)
