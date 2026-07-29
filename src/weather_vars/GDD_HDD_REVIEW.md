# Review: GDD / HDD construction (`a_growing_season`, `b_gdd`)
### and how to compute them accurately, per county per year

---

## Verdict in one line

The **growing-season step (A) is faithful to Ortiz-Bobea**; the **degree-day step (B) is not**.
Step B averages the day before applying the temperature response, which (i) biases GDD and
(ii) makes **HDD impossible to recover** — it is not computed anywhere in the pipeline.

---

## 1. The critical methodological problem

`b_step_gdd.py` collapses each day to its mean *first*, then applies a piecewise function:

```python
tmean = (tmax + tmin) / 2
gdd   = 0 if tmean < 8 else (24 if tmean > 32 else tmean - 8)
```

Two consequences:

**(a) Heat above 32 °C is capped into GDD, not separated out.** A day that peaks at 33 °C and one
that peaks at 45 °C both contribute `24`. The distinction between beneficial and harmful heat —
the entire reason for having GDD *and* HDD — is destroyed at source.

**(b) The within-day temperature path is discarded.** A day with `tmin=5, tmax=35` has
`tmean=20`, so it is scored as 12 beneficial degree-days and **zero** harmful exposure, even
though the crop spent several hours above 32 °C.

Ortiz-Bobea (following Schlenker & Roberts 2009, Snyder 1985) does the opposite: apply the
nonlinear response **within the day**, then integrate. Temperature is modelled as a sine wave
running between `tmin` and `tmax`,

&nbsp;&nbsp;&nbsp;&nbsp;`T(t) = M + A·sin(t)`, &nbsp; `M = (tmax+tmin)/2`, &nbsp; `A = (tmax−tmin)/2`

and the time above a threshold is integrated analytically. **Degree days above threshold `b`:**

| case | value |
|---|---|
| `tmax ≤ b` | `0` |
| `tmin ≥ b` | `M − b` |
| `tmin < b < tmax` | `(1/π)·[ (M−b)·(π/2 − θ) + A·cos θ ]`, &nbsp; `θ = arcsin((b−M)/A)` |

From that one primitive both variables follow, with bounds 8 °C and 32 °C:

> **GDD (beneficial) = DD(8) − DD(32)**
> **HDD (harmful)  = DD(32)**

Note the **subtraction**: degrees above 32 °C are moved *out* of the growth measure and *into*
the harm measure — they are not capped and silently retained.

### Implementation + validation
`src/b_gdd/degree_days.py` (new) implements this and validates it. Unit checks pass exactly
(`tmax≤b → 0`; `tmin≥b → M−b`; both boundary cases → 0). Measured against the current method on
real China rasters (0.1°, 18 sampled days of 1982):

| | current (daily mean) | sine method |
|---|---|---|
| mean GDD/day | 5.344 | **5.582 (+4.5%)** |
| mean HDD/day | **0 by construction** | 0.025 |

The GDD bias is *not* uniform: it is small mid-summer (+0.6 to +1.8%) but large in the shoulder
seasons — **+43% (1 Jan), +48% (2 Mar), +23% (21 Jan)** — because on cool days with a wide diurnal
range the afternoon rises above 8 °C while the daily mean does not, so the current method scores
those days as zero growth. This systematically distorts the *start and end* of the growing season,
which is precisely where planting/harvest decisions bind.

On heat: **12 of 18 sampled days had pixels above 32 °C** (up to 23,463 pixels on 9 Aug 1982),
all of which the current method records as zero harmful exposure.

---

## 2. Blockers — the scripts cannot run on this machine as written

| # | Problem | Fix |
|---|---|---|
| 1 | `utils.py`: `DATA_ROOT = /Users/zhangxinzhen/Desktop/TFP/Climate data` (macOS) | point to `Z:/weather data` |
| 2 | Expects folders `01 temp_min_max/`, `04 NDVI_China/` | actual: `temperature/`, `NDVI_China/` |
| 3 | Expects **extracted** `1982_max/19820101_max.tif`; data ships as **`1982_max.zip`** (730 files each) | extract, or read from the zip (faster: open each zip once per year) |
| 4 | `get_temp_grid_info()` samples `1980_max/19800101_max.tif` — **1980 does not exist** (data starts 1981) → crashes at setup | sample the first available year |

Verified about the rasters themselves (these are all consistent with the code's assumptions):
0.1° grid, `EPSG:4326`, bounds 70–140 °E / 15–55 °N, 400×700, **units are °C** (July max 43.5 °C),
NoData `−3.4e38` — so the `SENTINEL = -1e37` masking is correct.

---

## 3. What is already correct (keep it)

- **Growing season (step A) follows Ortiz-Bobea 2021 closely**: NDVI half-month climatology over
  35 years → wrap-padded to 30 periods → 7-period centred moving average → `argmax` = peak
  half-month → **5-month window = peak ± 2 months**. The wrap-padding and the `set`-based month
  parsing (never treating the list as a range) are both right.
- **County aggregation (step E)** uses `exact_extract(..., ops=['weighted_mean'], weights=cropland_weight)`
  on the matching grid — cropland-weighted area aggregation, which is the correct approach and
  matches the literature. Keeping GDD on the temperature grid and RZSM on the soil grid (never
  mixing) is also right.
- NoData/sentinel handling and the strict NaN-propagation rule are sound and documented.

---

## 4. Accurate GDD/HDD per county per year — the recipe

**Step 1 — daily, per pixel (0.1° grid).** For each day, from `tmin`/`tmax` *(never from the mean,
and never from the `_avg` files)*:
```
dd8  = dd_above(tmin, tmax, 8)
dd32 = dd_above(tmin, tmax, 32)
gdd_day = dd8 - dd32          # beneficial, 8-32 C
hdd_day = dd32                # harmful,  > 32 C
```

**Step 2 — accumulate over the growing season.** Sum `gdd_day` and `hdd_day` over the days of the
zone's 5-month window (peak ± 2). Accumulate **both** variables over the *same* window, so they
are directly comparable in the regression. Keep the existing NaN rule.

**Step 3 — aggregate to county, cropland-weighted.** Exactly as step E does today:
```
exact_extract(gdd_raster, counties, ops=['weighted_mean'],
              weights=cropland_weight_<slice>_tempgrid.tif, include_cols=['PAC'])
```
using the LUCC slice mapped to that year. HDD uses the *same* weights and grid.

**Step 4 — join to the TFP panel** on the county code, with the same layered matching used in
`27_map_tfp.py` (code → name → 曾用名 → stem), since production codes and boundary codes differ.

**Result:** `PAC × year × {gdd_growing_season, hdd_growing_season}`, both in °C·day,
cropland-weighted, over a phenologically-defined season — the Ortiz-Bobea specification, ready to
enter the TFP regression with the expected opposite signs (GDD `+`, HDD `−`).

### Two judgement calls worth making explicitly
- **Thresholds.** 8/32 °C follows the AJAE paper. Schlenker–Roberts use 29 °C for US maize; the
  right upper bound is crop- and region-specific. Because `dd_above` is a primitive, you can emit
  several bounds cheaply and let robustness checks decide.
- **Season resolution.** The window is currently defined at **37-zone** resolution and in whole
  **calendar months**. Pixel-level `peak_month.tif` already exists, so a county-specific window is
  available at no extra cost and would remove within-zone phenology error — worth at least a
  robustness check.
