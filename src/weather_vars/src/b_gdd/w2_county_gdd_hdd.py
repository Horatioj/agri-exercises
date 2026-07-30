# -*- coding: utf-8 -*-
"""
W2: growing-season GDD/HDD -> county-year, and join into the Input-Output table.

  1. Re-derive seasonal GDD/HDD from the archived MONTHLY grids using the
     cropland-based multi-peak season mask from W1.  This is a pure array
     multiply -- no daily raster is re-read.
         GDD_season(y) = sum_m  monthly_gdd(y,m) * season_mask(m)
  2. Aggregate each year to counties with `exact_extract`, cropland-fraction
     weighted (`weighted_mean`), on the 2010 census boundaries -- the same
     polygons used for the TFP maps, so the county correspondence is identical.
  3. Join onto the cleaned Input-Output panel via the layered county matcher.

TFP is deliberately NOT included: it depends on the estimator (DEA / SFA /
Solow), so the distributed table carries inputs, output and weather only, and
any TFP variant can be merged onto it later.

OUTPUTS
  src/clean/gdd_hdd/county_year_gdd_hdd.csv   county x year x {gdd, hdd}
  src/clean/io_weather_panel.csv              I-O + GDD/HDD (no TFP)
"""
from __future__ import annotations
import argparse, sys, time
# Chinese county names must survive a redirected console
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[3]          # ...\src
sys.path.insert(0, str(ROOT))
import _common as C
from _county_match import load_boundaries, match_counties

GDD_DIR = ROOT / "clean" / "gdd_hdd"
TEMP_TR = from_origin(70.0, 55.0, 0.1, 0.1)
TEMP_SHAPE = (400, 700)

IO_VARS = ["real_gvp", "Laborday_impute", "Land_serv_q", "capital_serv_q",
           "Inter_all_real", "GVP_allagr_impute", "Inter_all_nom",
           "agr_labor_num", "PPI_CCD_2005"]


def seasonal_from_monthly(monthly, mask):
    """monthly (12,H,W) -> season total, NaN if any in-season month is NaN."""
    sel = np.where(mask[:, None, None] if monthly.ndim == 2 else mask, True, False)
    tot = np.zeros(monthly.shape[1:], dtype="float32")
    miss = np.zeros(monthly.shape[1:], dtype=bool)
    any_month = np.zeros(monthly.shape[1:], dtype=bool)
    for m in range(12):
        inseason = mask[m]
        if not inseason.any():
            continue
        v = monthly[m]
        tot = np.where(inseason, tot + np.where(np.isnan(v), 0.0, v), tot)
        miss |= inseason & np.isnan(v)
        any_month |= inseason
    return np.where(any_month & ~miss, tot, np.nan).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly", default=str(GDD_DIR / "monthly_gdd_hdd_1981_2016.npz"))
    ap.add_argument("--season", default=str(GDD_DIR / "season_mask_tempgrid.npz"))
    ap.add_argument("--weights", default=str(GDD_DIR / "cropland_frac_tempgrid.tif"))
    ap.add_argument("--force", action="store_true", help="redo the county aggregation")
    a = ap.parse_args()

    from exactextract import exact_extract

    print("[1] seasonal GDD/HDD from the monthly archive x cropland season mask")
    mz = np.load(a.monthly)
    years = mz["years"]; MG = mz["gdd_month"]; MH = mz["hdd_month"]
    sz = np.load(a.season); mask = sz["mask"]
    print(f"    monthly {MG.shape}, season mask {mask.shape}, "
          f"mean season length {mask.sum(axis=0)[mask.any(axis=0)].mean():.2f} months")

    G = np.stack([seasonal_from_monthly(MG[i], mask) for i in range(len(years))])
    H = np.stack([seasonal_from_monthly(MH[i], mask) for i in range(len(years))])
    print(f"    GDD {np.nanmean(G[0]):.0f} ({years[0]}) -> {np.nanmean(G[-1]):.0f} ({years[-1]})"
          f" | HDD {np.nanmean(H[0]):.2f} -> {np.nanmean(H[-1]):.2f}")

    print("[2] county aggregation (cropland-weighted) on the 2010 boundaries")
    g = load_boundaries()
    cty_f = GDD_DIR / "county_year_gdd_hdd.csv"
    if cty_f.exists() and not a.force:
        cty = pd.read_csv(cty_f)
        print(f"    cached {cty_f.name} ({len(cty):,d} county-years) -- --force to redo")
        _skip = True
    else:
        _skip = False
    print(f"    {len(g)} county polygons")
    if _skip:
        pass
    prof = dict(driver="GTiff", dtype="float32", nodata=-9999.0,
                width=TEMP_SHAPE[1], height=TEMP_SHAPE[0], count=1,
                crs="EPSG:4326", transform=TEMP_TR)
    tmp = GDD_DIR / "_tmp_year.tif"
    rows = []
    t0 = time.time()
    for i, y in ([] if _skip else list(enumerate(years))):
        for name, arr in (("gdd", G[i]), ("hdd", H[i])):
            b = arr.copy(); b[np.isnan(b)] = -9999.0
            with rasterio.open(tmp, "w", **prof) as dst:
                dst.write(b, 1)
            # cropland-weighted (headline) AND plain area mean.  The area mean
            # never depends on the cropland raster, so it stays defined for
            # counties with little/no mapped cropland and for years outside the
            # CACD window -- useful as a robustness check.
            wm = exact_extract(str(tmp), g, ops=["weighted_mean"],
                               weights=a.weights, include_cols=["code"], output="pandas")
            am = exact_extract(str(tmp), g, ops=["mean"],
                               include_cols=["code"], output="pandas")
            wm = wm.rename(columns={[c for c in wm.columns if c != "code"][0]: name})
            am = am.rename(columns={[c for c in am.columns if c != "code"][0]: name + "_area"})
            rows.append(wm.merge(am, on="code").assign(year=int(y))[
                ["code", "year", name, name + "_area"]])
        if (i + 1) % 5 == 0:
            print(f"    {y} ({i+1}/{len(years)})  {time.time()-t0:.0f}s", flush=True)
    tmp.unlink(missing_ok=True)

    if not _skip:
        gdd_t = pd.concat([r for r in rows if "gdd" in r.columns]).reset_index(drop=True)
        hdd_t = pd.concat([r for r in rows if "hdd" in r.columns]).reset_index(drop=True)
        cty = gdd_t.merge(hdd_t, on=["code", "year"], how="outer")
        cty = cty.rename(columns={"code": "map_code",
                                  "gdd": "gdd_growing_season", "hdd": "hdd_growing_season",
                                  "gdd_area": "gdd_growing_season_area",
                                  "hdd_area": "hdd_growing_season_area"})
        cty.to_csv(cty_f, index=False)
        print(f"    saved {cty_f.name}  ({len(cty):,d} county-years, "
              f"{cty.map_code.nunique():,d} counties)")

    print("[3] join onto the Input-Output panel (no TFP)")
    io = pd.read_csv(C.CLEAN_PANEL)
    keep = ["SID", "state", "countyid", "county_name", "year"] + \
           [v for v in IO_VARS if v in io.columns]
    io = io[keep].copy()
    roster = io.dropna(subset=["county_name"]).groupby("countyid", as_index=False) \
               .agg(county_name=("county_name", "last"))
    roster = match_counties(roster, g)
    io = io.merge(roster[["countyid", "map_code", "how"]], on="countyid", how="left")
    out = io.merge(cty, on=["map_code", "year"], how="left")
    n_w = int(out["gdd_growing_season"].notna().sum())
    print(f"    I-O rows {len(out):,d}; with weather {n_w:,d} ({100*n_w/len(out):.1f}%)")
    out.to_csv(ROOT / "clean" / "io_weather_panel.csv", index=False)
    print(f"    saved src/clean/io_weather_panel.csv")
    print("[OK] W2 done")


if __name__ == "__main__":
    main()
