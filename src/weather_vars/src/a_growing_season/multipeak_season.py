# -*- coding: utf-8 -*-
"""
Multi-peak growing seasons for China (extends step A's single-peak rule).

WHY
---
Step A follows Ortiz-Bobea: smooth the 24-half-month NDVI climatology, take
`argmax`, and use peak +/- 2 months.  That is right for single-cropping regions,
but China runs 一年一熟 (NE) / 一年二熟 (North China Plain) / 一年三熟 (South).
Where two or three crops are grown the climatology has two or three peaks, and
`argmax` keeps only one of them: in the North China Plain, winter wheat +
summer maize span ~10 months, so a single 5-month window built on one peak
misses roughly half the exposure -- and misses it asymmetrically across crops.

WHAT THIS DOES
--------------
Finds ALL prominent local maxima in the circular 24-period climatology, so that
each pixel gets a number of cropping cycles and a growing-season month set that
is the union of `peak +/- half_width` around each retained peak.

Prominence rule (keeps it from firing on noise): a candidate peak is kept if
  * it is a strict local max on the circle, AND
  * its height above the lowest point between it and a taller peak (its
    topographic prominence) is at least `prom_frac` of the pixel's full
    climatological range (max - min).

Outputs per pixel: n_peaks, peak half-months, and the union month set.

USAGE
-----
    from multipeak_season import find_peaks_circular, season_months_multipeak
    peaks = find_peaks_circular(curve24, prom_frac=0.20)   # 1-D climatology
    months = season_months_multipeak(curve24, half_width_months=2)

Run directly for the unit tests:
    python multipeak_season.py
"""
from __future__ import annotations
import numpy as np

N_HALFMONTH = 24


def _prominence(curve: np.ndarray, idx: int, peaks: list[int]) -> float:
    """Topographic prominence of peak `idx` on a circular curve."""
    n = len(curve)
    higher = [p for p in peaks if curve[p] > curve[idx]]
    if not higher:
        return curve[idx] - curve.min()          # the global max
    best_col = -np.inf
    for p in higher:                             # walk both ways to each taller peak
        for direction in (1, -1):
            j, lowest = idx, curve[idx]
            for _ in range(n):
                j = (j + direction) % n
                lowest = min(lowest, curve[j])
                if j == p:
                    break
            best_col = max(best_col, lowest)
        # note: the highest saddle is the relevant one
    return curve[idx] - best_col


def find_peaks_circular(curve: np.ndarray, prom_frac: float = 0.20,
                        max_peaks: int = 3) -> list[int]:
    """Indices of prominent local maxima on a circular curve (len 24).

    prom_frac : minimum prominence as a fraction of (max - min) of the curve.
    max_peaks : keep at most this many, tallest first (China: <= 3 croppings).
    """
    c = np.asarray(curve, dtype="float64")
    n = len(c)
    if not np.isfinite(c).all() or np.nanmax(c) <= np.nanmin(c):
        return []
    rng = c.max() - c.min()
    cand = [i for i in range(n) if c[i] >= c[(i - 1) % n] and c[i] >= c[(i + 1) % n]]
    # drop plateau duplicates (keep the first index of a flat top)
    cand = [i for i in cand if not (c[i] == c[(i - 1) % n] and (i - 1) % n in cand)]
    keep = [i for i in cand if _prominence(c, i, cand) >= prom_frac * rng]
    keep.sort(key=lambda i: -c[i])
    return sorted(keep[:max_peaks])


def season_months_multipeak(curve: np.ndarray, half_width_months: int = 2,
                            prom_frac: float = 0.20, max_peaks: int = 3):
    """Return (sorted month set, peak half-month indices).

    Each retained peak contributes `peak_month +/- half_width_months`; the
    growing season is their union (so a double-cropping pixel gets a wider,
    possibly non-contiguous window rather than one arbitrary 5-month block).
    """
    peaks = find_peaks_circular(curve, prom_frac=prom_frac, max_peaks=max_peaks)
    months = set()
    for p in peaks:
        pm = p // 2 + 1                                   # half-month -> month
        for k in range(-half_width_months, half_width_months + 1):
            months.add(((pm - 1 + k) % 12) + 1)
        # NOTE: with half_width=2 a single peak gives 5 months, matching step A
    return sorted(months), peaks


# ---------------------------------------------------------------- unit tests
def _tests():
    hm = np.arange(N_HALFMONTH)

    def bump(centre, width=3.0, amp=1.0):
        d = np.minimum(np.abs(hm - centre), N_HALFMONTH - np.abs(hm - centre))
        return amp * np.exp(-(d ** 2) / (2 * width ** 2))

    print("single peak (NE, one crop)")
    c1 = 0.1 + bump(13)                                   # peak in July
    m1, p1 = season_months_multipeak(c1)
    print(f"   peaks={p1} -> months={m1}   (expect 1 peak, 5 months)")
    assert len(p1) == 1 and len(m1) == 5

    print("double peak (North China Plain: winter wheat + summer maize)")
    c2 = 0.1 + bump(9, 2.5, 1.0) + bump(17, 2.5, 0.95)    # May and Sep
    m2, p2 = season_months_multipeak(c2)
    print(f"   peaks={p2} -> months={m2}   (expect 2 peaks, ~10 months)")
    assert len(p2) == 2 and len(m2) >= 9

    print("triple peak (South China)")
    c3 = 0.1 + bump(5, 2.0) + bump(12, 2.0, .95) + bump(19, 2.0, .9)
    m3, p3 = season_months_multipeak(c3)
    print(f"   peaks={p3} -> months={m3}   (expect 3 peaks)")
    assert len(p3) == 3

    print("noisy single peak -- a small ripple must NOT count as a second crop")
    rng = np.random.default_rng(0)
    c4 = 0.1 + bump(13) + rng.normal(0, 0.01, N_HALFMONTH)
    m4, p4 = season_months_multipeak(c4)
    print(f"   peaks={p4} (expect 1)")
    assert len(p4) == 1

    print("flat curve -> no peaks, empty season")
    m5, p5 = season_months_multipeak(np.full(N_HALFMONTH, 0.3))
    print(f"   peaks={p5} months={m5}")
    assert p5 == [] and m5 == []

    print("wrap-around peak (season straddling Dec-Jan)")
    c6 = 0.1 + bump(0, 2.5)
    m6, p6 = season_months_multipeak(c6)
    print(f"   peaks={p6} -> months={m6}   (expect months to wrap 11,12,1,2)")
    assert len(p6) == 1 and 12 in m6 and 1 in m6

    print("\nall multi-peak unit tests passed")


if __name__ == "__main__":
    _tests()
