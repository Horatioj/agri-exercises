# -*- coding: utf-8 -*-
"""
W1: cropland-based multi-peak growing season (replaces the 37-zone window).

WHY NOT THE 熟制 ZONING
   中国农业熟制区划quhua covers only 62,136 of 280,000 grid cells and has real
   holes (Hainan, the Leizhou peninsula, unzoned desert/mountain in southern
   Xinjiang), so county polygons routinely extend past it and pixels silently
   fall outside every zone.  It also carries a NAME swap between OBJECTID_1
   3 and 4.  We therefore drop the zoning and define the season from the data:

        CACD cropland  ->  NDVI restricted to cropland  ->  peaks  ->  season

STEPS
  1. CACD 30 m binary cropland -> fraction on the NDVI grid (0.05 deg) and on
     the temperature grid (0.1 deg).  NoData=255 MUST be excluded from the
     average or the "fraction" comes out ~140 instead of 0-1.
  2. NDVI half-month climatology over 1982-2016 (24 periods), physical clip to
     [-1,1], then a circular 7-period centred moving average (step A's rule).
  3. Multi-peak detection per cropland pixel (_multipeak_season.py) -> number of
     cropping cycles + the union month set (peak +/- 2 months per peak).
  4. Season mask (12, H, W) on the TEMPERATURE grid, ready to multiply into the
     monthly GDD/HDD archive.

OUTPUTS (--out, default src/clean/gdd_hdd/)
  ndvi_climatology_halfmonth.nc   (24, latN, lonN) smoothed climatology
  cropland_frac_ndvigrid.tif      cropland fraction, 0.05 deg
  cropland_frac_tempgrid.tif      cropland fraction, 0.1 deg  (county weights)
  season_npeaks_tempgrid.tif      cropping cycles per pixel (0 = no cropland)
  season_mask_tempgrid.npz        (12, 400, 700) bool growing-season mask
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _multipeak_season import find_peaks_circular

NDVI_DIR = Path(r"Z:/weather data/NDVI_China")
CACD_DIR = Path(r"Z:/weather data/cropland")

# temperature grid (matches the GDD/HDD monthly archive)
TEMP_TR = from_origin(70.0, 55.0, 0.1, 0.1)
TEMP_SHAPE = (400, 700)


def cropland_fraction(cacd_tif: Path, dst_transform, dst_shape) -> np.ndarray:
    """Average-resample 30 m binary cropland to a coarse grid, EXCLUDING
    NoData=255 (otherwise the mean is polluted by the 255 fill)."""
    dst = np.zeros(dst_shape, dtype="float32")
    with rasterio.open(cacd_tif) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=255,
            dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=0,
            resampling=Resampling.average,
        )
    return np.clip(dst, 0.0, 1.0)


def ndvi_climatology(years, out_nc: Path):
    """35-year half-month NDVI climatology, then circular 7-period smoothing."""
    acc = None
    cnt = None
    for i, y in enumerate(years, 1):
        f = NDVI_DIR / f"Daily_Gap-filled_NDVI_{y}.nc4"
        if not f.exists():
            print(f"   {y}: missing, skipped"); continue
        t = time.time()
        ds = xr.open_dataset(f)
        a = ds["NDVI"].values
        a = np.where((a >= -1) & (a <= 1), a, np.nan)          # physical clip
        dates = pd.to_datetime(ds["Time"].values)
        hm = ((dates.month - 1) * 2 + (dates.day > 15).astype(int)).values
        if acc is None:
            acc = np.zeros((24,) + a.shape[1:], dtype="float64")
            cnt = np.zeros((24,) + a.shape[1:], dtype="int32")
        for k in range(24):
            m = hm == k
            if not m.any():
                continue
            sl = a[m]
            with np.errstate(invalid="ignore"):
                mean_k = np.nanmean(sl, axis=0)
            ok = np.isfinite(mean_k)
            acc[k][ok] += mean_k[ok]
            cnt[k][ok] += 1
        ds.close()
        print(f"   [{i:2d}/{len(years)}] {y} {time.time()-t:4.0f}s", flush=True)
    clim = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype("float32")

    # circular 7-period centred moving average (step A: pad 22-24 + 1-24 + 1-3)
    pad = np.concatenate([clim[-3:], clim, clim[:3]], axis=0)          # 30
    sm = np.empty_like(clim)
    for k in range(24):
        sm[k] = np.nanmean(pad[k:k + 7], axis=0)                        # centred on k+3
    return clim, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "clean" / "gdd_hdd"))
    ap.add_argument("--year-range", default="1982-2016")
    ap.add_argument("--cacd-year", type=int, default=2000, help="CACD slice for the cropland mask")
    ap.add_argument("--min-crop-frac", type=float, default=0.01,
                    help="a pixel counts as cropland above this fraction")
    ap.add_argument("--prom-frac", type=float, default=0.20,
                    help="peak prominence; diagnostic only")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="season threshold as a fraction of NDVI amplitude")
    ap.add_argument("--force", action="store_true", help="recompute cached steps")
    ap.add_argument("--half-width", type=int, default=2, help="months either side of each peak")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    y0, y1 = map(int, a.year_range.split("-"))
    years = list(range(y0, y1 + 1))

    # ---- NDVI grid geometry (from any year)
    ds0 = xr.open_dataset(NDVI_DIR / f"Daily_Gap-filled_NDVI_{y0}.nc4")
    lat = ds0["Lat"].values; lon = ds0["Lon"].values; ds0.close()
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    top = float(max(lat)) + dlat / 2; left = float(min(lon)) - dlon / 2
    ndvi_tr = from_origin(left, top, dlon, dlat)
    ndvi_shape = (len(lat), len(lon))
    lat_desc = lat[0] > lat[-1]
    print(f"[grid] NDVI {ndvi_shape} res={dlon:.4f}deg lat_descending={lat_desc}")

    # ---- 1. cropland fractions (cached: the 30 m resample is ~5 min per grid)
    cacd = CACD_DIR / f"CACD-{a.cacd_year}.tif"
    f_ndvi = out / "cropland_frac_ndvigrid.tif"
    f_temp = out / "cropland_frac_tempgrid.tif"
    print(f"[1] cropland fraction from {cacd.name}")
    if f_ndvi.exists() and f_temp.exists() and not a.force:
        with rasterio.open(f_ndvi) as s:
            cf_ndvi = s.read(1)
        with rasterio.open(f_temp) as s:
            cf_temp = s.read(1)
        print("    cached (use --force to recompute)")
    else:
        t = time.time(); cf_ndvi = cropland_fraction(cacd, ndvi_tr, ndvi_shape)
        print(f"    NDVI grid  {time.time()-t:4.0f}s  mean={cf_ndvi.mean():.4f}")
        t = time.time(); cf_temp = cropland_fraction(cacd, TEMP_TR, TEMP_SHAPE)
        print(f"    temp grid  {time.time()-t:4.0f}s  mean={cf_temp.mean():.4f}")
        for arr, tr, shp, nm in ((cf_ndvi, ndvi_tr, ndvi_shape, f_ndvi.name),
                                 (cf_temp, TEMP_TR, TEMP_SHAPE, f_temp.name)):
            with rasterio.open(out / nm, "w", driver="GTiff", dtype="float32",
                               nodata=-9999.0, width=shp[1], height=shp[0], count=1,
                               crs="EPSG:4326", transform=tr, compress="lzw") as dst:
                dst.write(arr, 1)
    print(f"    cropland cells>0: NDVI {int((cf_ndvi>0).sum()):,d}, "
          f"temp {int((cf_temp>0).sum()):,d}")

    # ---- 2. NDVI climatology (cached)
    clim_f = out / "ndvi_climatology_halfmonth.nc"
    print(f"[2] NDVI half-month climatology {y0}-{y1}")
    if clim_f.exists() and not a.force:
        with xr.open_dataset(clim_f) as d:
            clim = d["ndvi_clim_raw"].values; sm = d["ndvi_clim_smooth"].values
        print("    cached (use --force to recompute)")
    else:
        clim, sm = ndvi_climatology(years, clim_f)
        xr.Dataset({"ndvi_clim_raw": (("halfmonth", "lat", "lon"), clim),
                    "ndvi_clim_smooth": (("halfmonth", "lat", "lon"), sm)},
                   coords={"halfmonth": np.arange(1, 25), "lat": lat, "lon": lon}) \
            .to_netcdf(clim_f)
        print("    saved ndvi_climatology_halfmonth.nc")

    # ---- 3. SEASON = half-amplitude threshold on the climatology
    #
    # WHY NOT peak +/- 2 months.  Measured on this NDVI product: the 35-year
    # climatology is UNIMODAL even in known wheat+maize double-cropping counties
    # (德州, 周口), so peak detection returns one cycle almost everywhere
    # (11 of 176,262 cropland pixels got 2).  Single years do show two peaks at
    # 周口 (1995, 2005) but not at 德州, i.e. the 5 km "gap-filled" product plus
    # 35-year averaging cannot resolve China's cropping cycles.
    # A threshold instead of a peak sidesteps the problem: the season is every
    # month whose climatological NDVI clears half the annual amplitude, so a
    # double-cropping pixel automatically gets a LONGER season without anyone
    # having to resolve the individual peaks.
    #     threshold = min + alpha * (max - min),  alpha = 0.5 (half amplitude)
    # Validated: NE single-crop 5 months < 德州 6 < 周口/Hainan 7.
    print(f"[3] season = months with climatological NDVI >= min + "
          f"{a.alpha}*(max-min), on cropland (frac > {a.min_crop_frac})")
    crop = cf_ndvi > a.min_crop_frac
    H, W = ndvi_shape
    finite = np.isfinite(sm).all(axis=0)
    lo = np.nanmin(sm, axis=0); hi = np.nanmax(sm, axis=0)
    thr = lo + a.alpha * (hi - lo)
    inseason_hm = sm >= thr[None, :, :]                       # (24,H,W)
    mask_ndvi = np.zeros((12, H, W), dtype=bool)
    for k in range(24):
        mask_ndvi[k // 2] |= inseason_hm[k]
    valid = crop & finite & (hi > lo)
    mask_ndvi &= valid[None, :, :]
    print(f"    cropland pixels with a season: {int(valid.sum()):,d}")
    print(f"    mean season length: {mask_ndvi.sum(axis=0)[valid].mean():.2f} months")

    # cropping-cycle count kept as a DIAGNOSTIC only (see note above)
    npk = np.zeros((H, W), dtype="int8")
    ys, xs = np.where(valid)
    for r, c in zip(ys, xs):
        npk[r, c] = len(find_peaks_circular(sm[:, r, c], prom_frac=a.prom_frac, max_peaks=3))
    vals, counts = np.unique(npk[valid], return_counts=True)
    print(f"    [diagnostic] detected cycles: {dict(zip(vals.tolist(), counts.tolist()))}")

    # ---- 4. resample season mask + npeaks to the temperature grid
    print("[4] season mask -> temperature grid")
    # CROPLAND-WEIGHTED resample.  A plain average of the boolean mask asks
    # "what share of this 0.1deg cell is in-season cropland", so cells with
    # little cropland fall below any threshold and lose their season entirely
    # (44k of 69k cropland cells survived that way).  The question we actually
    # want is "of the CROPLAND in this cell, is the majority in season", i.e.
    #     sum(mask * crop) / sum(crop)
    # obtained as the ratio of two average-resamples.
    cw = np.zeros(TEMP_SHAPE, dtype="float32")
    reproject(source=cf_ndvi.astype("float32"), destination=cw,
              src_transform=ndvi_tr, src_crs="EPSG:4326",
              dst_transform=TEMP_TR, dst_crs="EPSG:4326",
              resampling=Resampling.average)
    mask_temp = np.zeros((12,) + TEMP_SHAPE, dtype=bool)
    for m in range(12):
        num = np.zeros(TEMP_SHAPE, dtype="float32")
        reproject(source=(mask_ndvi[m] * cf_ndvi).astype("float32"), destination=num,
                  src_transform=ndvi_tr, src_crs="EPSG:4326",
                  dst_transform=TEMP_TR, dst_crs="EPSG:4326",
                  resampling=Resampling.average)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(cw > 0, num / cw, 0.0)
        mask_temp[m] = share >= 0.5           # majority of the CROPLAND in season
    npk_t = np.zeros(TEMP_SHAPE, dtype="float32")
    reproject(source=npk.astype("float32"), destination=npk_t,
              src_transform=ndvi_tr, src_crs="EPSG:4326",
              dst_transform=TEMP_TR, dst_crs="EPSG:4326", resampling=Resampling.mode)
    np.savez_compressed(out / "season_mask_tempgrid.npz",
                        mask=mask_temp, npeaks=npk_t.astype("int8"))
    with rasterio.open(out / "season_npeaks_tempgrid.tif", "w", driver="GTiff",
                       dtype="int16", nodata=0, width=TEMP_SHAPE[1], height=TEMP_SHAPE[0],
                       count=1, crs="EPSG:4326", transform=TEMP_TR, compress="lzw") as dst:
        dst.write(npk_t.astype("int16"), 1)
    have = mask_temp.any(axis=0)
    print(f"    temp-grid pixels with a season: {int(have.sum()):,d}; "
          f"mean length {mask_temp.sum(axis=0)[have].mean():.2f} months")
    print("[OK] W1 done")


if __name__ == "__main__":
    main()
