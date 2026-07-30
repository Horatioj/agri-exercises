# -*- coding: utf-8 -*-
"""
W3: growing-season soil moisture -> county-year.

Source: Z:/weather data/SM_data.nc -- ERA5 MONTHLY volumetric soil water,
432 months (1981-01 .. 2016-12, complete), 391 x 651 at 0.1 deg (72-137E,
16-55N).  Already monthly, so it plugs straight into the same season-mask
architecture as GDD/HDD; no daily pass is needed.

Variables built (ERA5 layer depths 0-7, 7-28, 28-100 cm):
  rzsm_gs        root-zone soil moisture, depth-weighted mean of swvl1/2/3
                 with weights .07/.21/.72, averaged over the growing season
  sm_shallow_gs  surface layer (swvl1) over the growing season

IMPORTANT: the SM grid (391x651) is NOT the temperature grid (400x700).  The
season mask and the cropland weights must both be built on the SM grid --
mixing grids silently misaligns everything by half a pixel and by extent.

OUTPUTS (src/clean/gdd_hdd/)
  cropland_frac_smgrid.tif
  sm_growing_season_1981_2016.nc     (year, lat, lon) rzsm_gs, sm_shallow_gs
  county_year_soilmoisture.csv       county x year, cropland-weighted + area mean
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from _county_match import load_boundaries

GDD_DIR = ROOT / "clean" / "gdd_hdd"
SM_NC = Path(r"Z:/weather data/SM_data.nc")
CACD = Path(r"Z:/weather data/cropland/CACD-2000.tif")
NDVI_TR = None            # filled from the season file's source grid
LAYER_W = np.array([0.07, 0.21, 0.72])          # ERA5 0-7 / 7-28 / 28-100 cm


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

    print("[4] growing-season means by year")
    sw = np.stack([ds[f"swvl{i}"].values for i in (1, 2, 3)])        # (3,T,H,W)
    rz = np.tensordot(LAYER_W, sw, axes=(0, 0))                      # (T,H,W)
    sh = sw[0]
    years = sorted({int(x) for x in t.year})
    mon = t.month.values; yr = t.year.values
    RZ, SH = [], []
    for y in years:
        num_r = np.zeros(sm_shape, "float64"); num_s = np.zeros(sm_shape, "float64")
        cntm = np.zeros(sm_shape, "float64")
        for m in range(1, 13):
            k = np.where((yr == y) & (mon == m))[0]
            if len(k) == 0:
                continue
            inseason = mask_sm[m - 1]
            num_r += np.where(inseason, rz[k[0]], 0.0)
            num_s += np.where(inseason, sh[k[0]], 0.0)
            cntm += inseason.astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            RZ.append(np.where(cntm > 0, num_r / cntm, np.nan).astype("float32"))
            SH.append(np.where(cntm > 0, num_s / cntm, np.nan).astype("float32"))
    RZ = np.stack(RZ); SH = np.stack(SH)
    print(f"    rzsm {np.nanmean(RZ[0]):.4f} ({years[0]}) -> {np.nanmean(RZ[-1]):.4f} ({years[-1]}) m3/m3")

    xr.Dataset({"rzsm_gs": (("year", "lat", "lon"), RZ),
                "sm_shallow_gs": (("year", "lat", "lon"), SH)},
               coords={"year": years, "lat": lat, "lon": lon},
               attrs={"source": "ERA5 monthly swvl1-3",
                      "rzsm": "depth-weighted 0-100cm (.07/.21/.72)",
                      "season": "cropland-NDVI growing season (W1)"}) \
        .to_netcdf(GDD_DIR / "sm_growing_season_1981_2016.nc")

    print("[5] county aggregation (cropland-weighted AND plain area mean)")
    g = load_boundaries()
    prof = dict(driver="GTiff", dtype="float32", nodata=-9999.0,
                width=sm_shape[1], height=sm_shape[0], count=1,
                crs="EPSG:4326", transform=sm_tr)
    tmp = GDD_DIR / "_tmp_sm.tif"
    out = []
    t0 = time.time()
    for i, y in enumerate(years):
        rec = None
        for name, arr in (("rzsm_gs", RZ[i]), ("sm_shallow_gs", SH[i])):
            b = arr.copy(); b[np.isnan(b)] = -9999.0
            with rasterio.open(tmp, "w", **prof) as d:
                d.write(b, 1)
            wm = exact_extract(str(tmp), g, ops=["weighted_mean"], weights=str(f_cw),
                               include_cols=["code"], output="pandas")
            am = exact_extract(str(tmp), g, ops=["mean"],
                               include_cols=["code"], output="pandas")
            wm = wm.rename(columns={[c for c in wm.columns if c != "code"][0]: name})
            am = am.rename(columns={[c for c in am.columns if c != "code"][0]: name + "_area"})
            part = wm.merge(am, on="code")
            rec = part if rec is None else rec.merge(part, on="code")
        out.append(rec.assign(year=int(y)))
        if (i + 1) % 6 == 0:
            print(f"    {y} ({i+1}/{len(years)}) {time.time()-t0:.0f}s", flush=True)
    tmp.unlink(missing_ok=True)
    cty = pd.concat(out, ignore_index=True).rename(columns={"code": "map_code"})
    cty.to_csv(GDD_DIR / "county_year_soilmoisture.csv", index=False)
    print(f"    saved county_year_soilmoisture.csv ({len(cty):,d} county-years, "
          f"{cty.map_code.nunique():,d} counties)")
    print("[OK] W3 done")


if __name__ == "__main__":
    main()
