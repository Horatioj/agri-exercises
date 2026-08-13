# -*- coding: utf-8 -*-
"""
W3: growing-season soil moisture -> county-year.

Source: Z:/weather data/SM_data.nc -- ERA5 MONTHLY volumetric soil water,
432 months (1981-01 .. 2016-12, complete), 391 x 651 at 0.1 deg (72-137E,
16-55N).  Already monthly, so it plugs straight into the same season-mask
architecture as GDD/HDD; no daily pass is needed.

Variables built (ERA5 layer depths 0-7, 7-28, 28-100 cm), in LEVELS and as
STANDARDISED ANOMALIES:

  <var>          growing-season mean LEVEL, m3/m3
  <var>_z        growing-season mean STANDARDISED ANOMALY (strategy B2)

B2 -- WHY THE ANOMALY AND NOT THE LEVEL
  A level mostly measures WHERE a county is (arid Gansu vs humid Jiangxi), and
  that cross-sectional variation is absorbed by county fixed effects anyway.  It
  also carries the seasonal cycle: soil moisture is high in winter simply
  because nothing is transpiring, so a raw mean is dominated by when the season
  happens to sit rather than by whether the year was dry.  The anomaly is taken
  per pixel and per CALENDAR MONTH against that pixel-month's own 1981-2016
  distribution,

        z(i,m,t) = ( SM(i,m,t) - mean_t SM(i,m) ) / sd_t SM(i,m)

  so a December is only ever compared with other Decembers, and z is the
  year-to-year shock with the level and the seasonality removed.  This is the
  standard operational definition of a soil-moisture anomaly (Copernicus EDO
  Soil Moisture Anomaly; the Standardized Soil Moisture Index literature), and
  soil-moisture anomalies of this form are what Ortiz-Bobea, Wang, Carrillo &
  Ault (2019, ERL) use to unpack the climatic drivers of US yields.

  NOTE FOR THE FRONTIER.  z is negative about half the time, and the CPS
  weak-disposability programme needs w in conv({0} u {w_j}) with lam >= 0, where
  the origin has to mean "none of it".  So the frontier takes the within-county
  PERCENTILE RANK of z (0-1, strictly positive, monotone in z), emitted as
  <var>_pctl; the raw z is the regression variable.

B4 -- DORMANT-MONTH MASK  (OPT-IN ROBUSTNESS CHECK, NOT THE HEADLINE SPEC)

  Run with --dormant-mask to enable.  It is OFF by default because it is the one
  choice here with no agricultural-economics citation behind it: frozen-ground
  masking is standard QC in hydrology and remote sensing (SM products flag and
  withhold frozen pixels), but the agricultural-economics literature handles the
  same problem the other way -- by restricting to the growing season (A2) and
  using year-to-year deviations at each location (B2), exactly as Ortiz-Bobea,
  Wang, Carrillo & Ault (2019, ERL) do.  Measured on this panel the mask removes
  only 1.44% of season pixel-months, so A2 x B2 loses essentially nothing by
  leaving it out.
  Frozen soil water is not plant-available, and ERA5 swvl reports it as water
  all the same.  The literal fix is to mask on ERA5 soil temperature, but
  SM_data.nc carries only swvl1/2/3 -- there is no soil-temperature field to
  mask with.  We therefore mask on the THERMAL criterion we already have: a
  month is dormant where the CLIMATOLOGICAL monthly GDD (from the 10_ archive,
  base 8 C) is below DORMANT_GDD, i.e. the month essentially never rises above
  the growing base.  Using the climatology, not the individual year, keeps the
  mask time-invariant and therefore exogenous to the weather being measured --
  the same reasoning that makes the A2 season a 35-year NDVI climatology.
  Its bite is deliberately small: A2 already excludes winter almost everywhere,
  so this only trims shoulder months in the far north.

IMPORTANT: the SM grid (391x651) is NOT the temperature grid (400x700).  The
season mask and the cropland weights must both be built on the SM grid --
mixing grids silently misaligns everything by half a pixel and by extent.

OUTPUTS (src/clean/gdd_hdd/)
  cropland_frac_smgrid.tif
  sm_growing_season_1981_2016.nc     (year, lat, lon) levels and _z anomalies
  county_year_soilmoisture.csv       county x year, cropland-weighted + area mean,
                                     levels, _z anomalies and _pctl ranks
"""
from __future__ import annotations
import argparse, sys, time
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _county_match import load_boundaries

GDD_DIR = ROOT / "clean" / "gdd_hdd"
SM_NC = Path(r"Z:/weather data/SM_data.nc")
CACD = Path(r"Z:/weather data/cropland/CACD-2000.tif")
NDVI_TR = None            # filled from the season file's source grid
# ERA5 layer depths: swvl1 0-7cm, swvl2 7-28cm, swvl3 28-100cm.
# A 0-100cm depth weighting puts 0.72 on layer 3, which then DOMINATES the
# average -- but 28-100cm is not a meaningful rooting depth in much of China
# (thin soils on the plateau, and annual cereals root mostly in the top ~30cm).
# So we no longer impose one root-zone definition: every layer is emitted
# separately, plus a 0-28cm combination (the agronomically defensible root zone
# for annual crops) and the legacy 0-100cm for comparison.
W_0_100 = np.array([0.07, 0.21, 0.72])          # legacy deep root zone
W_0_28 = np.array([0.07, 0.21, 0.00]) / 0.28    # 0-28cm, renormalised

# B4: a month counts as thermally dormant where the CLIMATOLOGICAL monthly GDD
# is below this (deg C-day).  8 C is the GDD base, so ~zero monthly GDD means the
# month essentially never got above the growing threshold.
DORMANT_GDD = 5.0


def cropland_fraction(cacd_tif, dst_transform, dst_shape):
    dst = np.zeros(dst_shape, dtype="float32")
    with rasterio.open(cacd_tif) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=255,
                  dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=0,
                  resampling=Resampling.average)
    return np.clip(dst, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dormant-mask", action="store_true",
                    help="B4 robustness check: also drop thermally dormant months. "
                         "OFF by default -- the headline spec is strictly A2 x B2, "
                         "both of which are backed by the published literature.")
    a = ap.parse_args()
    from exactextract import exact_extract

    print("[1] open ERA5 monthly soil moisture")
    ds = xr.open_dataset(SM_NC)
    lat = ds["latitude"].values; lon = ds["longitude"].values
    t = pd.to_datetime(ds["valid_time"].values)
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    sm_tr = from_origin(float(lon.min()) - dlon / 2, float(lat.max()) + dlat / 2, dlon, dlat)
    sm_shape = (len(lat), len(lon))
    print(f"    {sm_shape} res={dlon:.2f}deg  {t.min().date()} .. {t.max().date()} "
          f"({len(t)} months)")

    print("[2] cropland fraction on the SM grid")
    f_cw = GDD_DIR / "cropland_frac_smgrid.tif"
    if f_cw.exists() and not a.force:
        with rasterio.open(f_cw) as s:
            cw = s.read(1)
        print("    cached")
    else:
        t0 = time.time(); cw = cropland_fraction(CACD, sm_tr, sm_shape)
        with rasterio.open(f_cw, "w", driver="GTiff", dtype="float32", nodata=-9999.0,
                           width=sm_shape[1], height=sm_shape[0], count=1,
                           crs="EPSG:4326", transform=sm_tr, compress="lzw") as d:
            d.write(cw, 1)
        print(f"    {time.time()-t0:.0f}s  mean={cw.mean():.4f}")

    print("[3] season mask -> SM grid (cropland-weighted, from the NDVI-grid mask)")
    sz = np.load(GDD_DIR / "season_mask_tempgrid.npz")
    mask_temp = sz["mask"]                       # (12,400,700) on the TEMP grid
    temp_tr = from_origin(70.0, 55.0, 0.1, 0.1)
    mask_sm = np.zeros((12,) + sm_shape, dtype=bool)
    for m in range(12):
        dst = np.zeros(sm_shape, dtype="float32")
        reproject(source=mask_temp[m].astype("float32"), destination=dst,
                  src_transform=temp_tr, src_crs="EPSG:4326",
                  dst_transform=sm_tr, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest)
        mask_sm[m] = dst > 0.5
    have = mask_sm.any(axis=0)
    print(f"    SM-grid pixels with a season: {int(have.sum()):,d}; "
          f"mean length {mask_sm.sum(axis=0)[have].mean():.2f} months")

    if not a.dormant_mask:
        print("[3b] B4 dormant-month mask: OFF (default) -- headline spec is A2 x B2")
    else:
      print("[3b] B4: drop thermally dormant months (climatological monthly GDD)")
      # pick the WIDEST archive, not the alphabetically last: both
      # monthly_gdd_hdd_1981_2016.npz and _1982_2016.npz exist and sorting puts
      # the shorter 1982 one last.
      def _span(f):
          try:
              a, b = f.stem.split("_")[-2:]
              return int(b) - int(a)
          except Exception:
              return -1
      arch = sorted(GDD_DIR.glob("monthly_gdd_hdd_*.npz"), key=_span)
      if not arch:
          print("    no monthly GDD archive found -> B4 mask SKIPPED")
      else:
          z_ = np.load(arch[-1])
          gdd_clim = z_["gdd_month"].mean(axis=0)          # (12,400,700) over years
          active_sm = np.zeros((12,) + sm_shape, dtype=bool)
          for m in range(12):
              dst = np.zeros(sm_shape, dtype="float32")
              reproject(source=gdd_clim[m].astype("float32"), destination=dst,
                        src_transform=temp_tr, src_crs="EPSG:4326",
                        dst_transform=sm_tr, dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear)
              active_sm[m] = dst >= DORMANT_GDD
          before = int(mask_sm.sum())
          mask_sm = mask_sm & active_sm
          after = int(mask_sm.sum())
          print(f"    {arch[-1].name}: dormant threshold {DORMANT_GDD:g} degC-day")
          print(f"    season pixel-months {before:,d} -> {after:,d} "
                f"({100*(before-after)/max(before,1):.2f}% dropped as dormant)")
          have = mask_sm.any(axis=0)
          if have.any():
              print(f"    pixels retaining a season: {int(have.sum()):,d}; "
                    f"mean length {mask_sm.sum(axis=0)[have].mean():.2f} months")

    print("[4] growing-season means by year")
    sw = np.stack([ds[f"swvl{i}"].values for i in (1, 2, 3)])        # (3,T,H,W)
    SERIES = {
        "sm_0_28":   np.tensordot(W_0_28, sw, axes=(0, 0)),    # root zone, annual crops
        "rzsm_gs":   np.tensordot(W_0_100, sw, axes=(0, 0)),   # legacy 0-100cm
        "swvl1_gs":  sw[0],                                    # 0-7cm  surface
        "swvl2_gs":  sw[1],                                    # 7-28cm
        "swvl3_gs":  sw[2],                                    # 28-100cm
    }
    years = sorted({int(x) for x in t.year})
    mon = t.month.values; yr = t.year.values

    # ---- B2: standardise each series per pixel per CALENDAR MONTH ----------
    # A December is compared only with other Decembers, so the seasonal cycle
    # (high winter moisture with nothing growing) cannot leak into the measure.
    print("[4a] B2: standardised anomalies (per pixel, per calendar month)")
    for name in list(SERIES):
        arr = SERIES[name]
        zarr = np.full_like(arr, np.nan, dtype="float32")
        for m in range(1, 13):
            k_ = np.where(mon == m)[0]
            if len(k_) < 3:
                continue
            sub = arr[k_]
            mu = np.nanmean(sub, axis=0)
            sd = np.nanstd(sub, axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                zarr[k_] = np.where(sd > 1e-8, (sub - mu) / sd, np.nan)
        SERIES[name + "_z"] = zarr
    print(f"    {len([k for k in SERIES if k.endswith('_z')])} anomaly series added")
    OUT = {k: [] for k in SERIES}
    for y in years:
        cntm = np.zeros(sm_shape, "float64")
        num = {k: np.zeros(sm_shape, "float64") for k in SERIES}
        for m in range(1, 13):
            k_ = np.where((yr == y) & (mon == m))[0]
            if len(k_) == 0:
                continue
            inseason = mask_sm[m - 1]
            for k, arr in SERIES.items():
                num[k] += np.where(inseason, arr[k_[0]], 0.0)
            cntm += inseason.astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in SERIES:
                OUT[k].append(np.where(cntm > 0, num[k] / cntm, np.nan).astype("float32"))
    OUT = {k: np.stack(v) for k, v in OUT.items()}
    for k, v in OUT.items():
        unit = "sd" if k.endswith("_z") else "m3/m3"
        print(f"    {k:14s} {np.nanmean(v[0]):+.4f} ({years[0]}) -> "
              f"{np.nanmean(v[-1]):+.4f} ({years[-1]}) {unit}")

    _ds_out = xr.Dataset({k: (("year", "lat", "lon"), v) for k, v in OUT.items()},
               coords={"year": years, "lat": lat, "lon": lon},
               attrs={"source": "ERA5 monthly swvl1-3 (0-7 / 7-28 / 28-100 cm)",
                      "sm_0_28": "root zone for annual crops, layers 1-2 renormalised",
                      "rzsm_gs": "legacy 0-100cm (.07/.21/.72) - layer 3 dominates",
                      "season": "cropland-NDVI growing season (A2, 11_growing_season.py)",
                      "dormant_mask": f"B4: climatological monthly GDD >= {DORMANT_GDD} degC-day",
                      "_z": "B2: standardised anomaly per pixel per calendar month, "
                            "1981-2016 base"})
    _ds_out.to_netcdf(GDD_DIR / "sm_growing_season_1981_2016.nc")

    print("[5] county aggregation (cropland-weighted AND plain area mean)")
    # ONE MULTI-BAND RASTER PER YEAR, ONE exact_extract CALL.
    # The per-series version issued len(SERIES) x 2 calls per year -- 720 passes
    # over 2,865 polygons once the B2 anomalies doubled the series count, which
    # took ~4.5 min per year and died partway through.  exact_extract reads a
    # multi-band raster in a single pass and returns band_<n>_<op> columns, so
    # the polygon geometry is rasterised once per year instead of once per
    # series-op.  Same numbers, ~20x fewer passes.
    g = load_boundaries()
    names = list(SERIES)                      # band order, 1-based in the output
    prof = dict(driver="GTiff", dtype="float32", nodata=-9999.0,
                width=sm_shape[1], height=sm_shape[0], count=len(names),
                crs="EPSG:4326", transform=sm_tr)
    tmp = GDD_DIR / "_tmp_sm.tif"
    out = []
    t0 = time.time()
    for i, y in enumerate(years):
        stack = np.stack([OUT[n][i] for n in names]).astype("float32")
        stack[np.isnan(stack)] = -9999.0
        with rasterio.open(tmp, "w", **prof) as d:
            d.write(stack)
        r = exact_extract(str(tmp), g, ops=["weighted_mean", "mean"],
                          weights=str(f_cw), include_cols=["code"], output="pandas")
        # Column names are DISCOVERED, not assumed: with a weights raster
        # exact_extract emits "band_<n>_<weightsband>_weighted_mean" (e.g.
        # band_1_weight_weighted_mean), not "band_<n>_weighted_mean".  Matching
        # on prefix+suffix keeps this robust to that naming.
        ren = {}
        for b, n in enumerate(names, start=1):
            pre = f"band_{b}_"
            wcol = [c for c in r.columns
                    if c.startswith(pre) and c.endswith("weighted_mean")]
            acol = [c for c in r.columns
                    if c.startswith(pre) and c.endswith("mean") and "weighted" not in c]
            if len(wcol) != 1 or len(acol) != 1:
                raise RuntimeError(f"cannot map band {b} ({n}); columns were "
                                   f"{list(r.columns)[:8]}")
            ren[wcol[0]] = n
            ren[acol[0]] = n + "_area"
        r = r.rename(columns=ren)[["code"] + [ren[k] for k in ren]]
        out.append(r.assign(year=int(y)))
        if (i + 1) % 6 == 0 or i == 0:
            print(f"    {y} ({i+1}/{len(years)}) {time.time()-t0:.0f}s", flush=True)
    tmp.unlink(missing_ok=True)
    cty = pd.concat(out, ignore_index=True).rename(columns={"code": "map_code"})

    # ---- within-county PERCENTILE RANK of each anomaly --------------------
    # The CPS frontier needs w >= 0 with a meaningful origin; a z-score is
    # negative half the time.  The rank is strictly positive, bounded (0,1) and
    # monotone in z, so it carries the same ordering without breaking the
    # conv({0} u {w_j}) geometry.  Ranked WITHIN county, so it is a pure
    # within-county weather shock, like the anomaly it comes from.
    zcols = [c for c in cty.columns if c.endswith("_z")]
    for c in zcols:
        cty[c.replace("_z", "_pctl")] = (cty.groupby("map_code")[c]
                                            .rank(pct=True, method="average"))
    print(f"    percentile ranks added for {len(zcols)} anomaly series")
    cty.to_csv(GDD_DIR / "county_year_soilmoisture.csv", index=False)
    print(f"    saved county_year_soilmoisture.csv ({len(cty):,d} county-years, "
          f"{cty.map_code.nunique():,d} counties)")
    print("[OK] W3 done")


if __name__ == "__main__":
    main()
