# -*- coding: utf-8 -*-
"""
B step (corrected): per-pixel per-year growing-season GDD **and HDD**.

Replaces b_step_gdd.py.  Three substantive changes:

 1. DEGREE-DAY METHOD.  b_step_gdd.py applied the temperature response to the
    daily MEAN (tmean=(tmax+tmin)/2, capped at 32C).  That biases GDD (+4.5%
    overall on 1982 rasters, up to +48% in shoulder months) and makes harmful
    heat unrecoverable.  Here we use the within-day single-sine integration
    (Snyder 1985 / Schlenker & Roberts 2009, as in Ortiz-Bobea):
        GDD = DD(8) - DD(32)      HDD = DD(32)
    See degree_days.py for the closed form and its validation.

 2. HDD IS EMITTED.  The old pipeline had no harmful-heat variable at all.

 3. RUNS ON THE WINDOWS DATA LAYOUT.  Temperature ships as per-year ZIPs
    (`1982_max.zip`, 730 members); this reads them in place, no extraction.
    Also fixes the setup crash where the grid was probed from 1980 (data
    starts 1981).

Growing-season window, in order of preference:
  --season-csv  zone37_growing_season.csv + --zone-shp quhua.shp   (step A output)
  --months 4,5,6,7,8,9                                             (fixed window)
  --all-year                                                       (no window)

Outputs (into --out):
  gdd_hdd_growing_season_{y0}_{y1}.nc   vars: gdd_growing_season, hdd_growing_season
  gdd_growing_season_{year}.tif / hdd_growing_season_{year}.tif    (sample years)

Example (fixed Apr-Sep window, no step-A intermediates needed):
  python b_step_gdd_hdd.py --year-range 1982-1983 --months 4,5,6,7,8,9
"""
from __future__ import annotations
import argparse
import io
import re
import sys
import time
import zipfile
from calendar import isleap
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degree_days import dd_above, BASE_T, CAP_T

SENTINEL = -1e37          # tmax/tmin NoData is -3.4e38


# ----------------------------------------------------------------- zip reader
class TempZipYear:
    """Random access to one year of daily tmax/tmin held inside per-year ZIPs."""

    def __init__(self, temp_dir: Path, year: int):
        self.zmax = zipfile.ZipFile(temp_dir / f"{year}_max.zip")
        self.zmin = zipfile.ZipFile(temp_dir / f"{year}_min.zip")
        self._max = {m.split("/")[-1]: m for m in self.zmax.namelist() if m.endswith(".tif")}
        self._min = {m.split("/")[-1]: m for m in self.zmin.namelist() if m.endswith(".tif")}

    def close(self):
        self.zmax.close(); self.zmin.close()

    @staticmethod
    def _read(z: zipfile.ZipFile, member: str) -> np.ndarray:
        with rasterio.io.MemoryFile(z.read(member)) as mf, mf.open() as src:
            a = src.read(1).astype("float32")
        return np.where(a > SENTINEL, a, np.nan)

    def day(self, ymd: str):
        """(tmin, tmax) for YYYYMMDD, or (None, None) if that day is absent."""
        fm, fn = f"{ymd}_max.tif", f"{ymd}_min.tif"
        if fm not in self._max or fn not in self._min:
            return None, None
        return self._read(self.zmin, self._min[fn]), self._read(self.zmax, self._max[fm])

    def any_profile(self):
        """Grid metadata from the first available member (never assumes 1980)."""
        first = sorted(self._max)[0]
        with rasterio.io.MemoryFile(self.zmax.read(self._max[first])) as mf, mf.open() as src:
            return dict(transform=src.transform, crs=src.crs, shape=src.shape,
                        width=src.width, height=src.height, bounds=src.bounds)


# ------------------------------------------------------------ season handling
def build_zone_raster(zone_shp: Path, grid: dict):
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = gpd.read_file(zone_shp).to_crs(4326)
    idcol = "OBJECTID_1" if "OBJECTID_1" in gdf.columns else gdf.columns[0]
    shapes = [(g, int(o)) for g, o in zip(gdf.geometry, gdf[idcol])]
    return rasterize(shapes, out_shape=grid["shape"], transform=grid["transform"],
                     fill=0, dtype="int32")


def load_season_lookup(csv_path: Path) -> dict:
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import parse_season_months
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r.get("peak_month_main")):
            continue
        out[int(r["OBJECTID_1"])] = parse_season_months(r["green_season_months"])
    return out


# ---------------------------------------------------------------- yearly core
def year_gdd_hdd(year: int, temp_dir: Path, grid: dict,
                 zone_raster, season_lookup, fixed_months):
    """Return (gdd, hdd) 2-D arrays accumulated over the growing season."""
    H, W = grid["shape"]
    zy = TempZipYear(temp_dir, year)
    try:
        # month accumulators + "this month had a missing/NaN day" flags
        mg = np.zeros((12, H, W), dtype="float32")
        mh = np.zeros((12, H, W), dtype="float32")
        bad = np.zeros((12, H, W), dtype=bool)

        d = date(year, 1, 1)
        for _ in range(366 if isleap(year) else 365):
            tmin, tmax = zy.day(d.strftime("%Y%m%d"))
            m = d.month - 1
            if tmin is None:
                bad[m] = True                      # whole day missing from the archive
            else:
                dd8 = dd_above(tmin, tmax, BASE_T)
                dd32 = dd_above(tmin, tmax, CAP_T)
                gdd_day = dd8 - dd32               # beneficial 8-32C
                hdd_day = dd32                     # harmful  >32C
                nan = np.isnan(gdd_day)
                mg[m] += np.where(nan, 0.0, gdd_day)
                mh[m] += np.where(nan, 0.0, hdd_day)
                bad[m] |= nan
            d += timedelta(days=1)

        for m in range(12):                        # strict propagation (as before)
            mg[m] = np.where(bad[m], np.nan, mg[m])
            mh[m] = np.where(bad[m], np.nan, mh[m])

        gdd = np.full((H, W), np.nan, dtype="float32")
        hdd = np.full((H, W), np.nan, dtype="float32")

        def accumulate(mask, months):
            sg = np.zeros((H, W), "float32"); sh = np.zeros((H, W), "float32")
            miss = np.zeros((H, W), bool)
            for mo in months:
                sg += np.where(np.isnan(mg[mo - 1]), 0.0, mg[mo - 1])
                sh += np.where(np.isnan(mh[mo - 1]), 0.0, mh[mo - 1])
                miss |= np.isnan(mg[mo - 1])
            ok = mask & ~miss
            return ok, sg, sh

        if season_lookup:                          # per-zone window
            for oid, months in season_lookup.items():
                zmask = zone_raster == oid
                if not zmask.any():
                    continue
                ok, sg, sh = accumulate(zmask, months)
                gdd = np.where(ok, sg, np.where(zmask, np.nan, gdd))
                hdd = np.where(ok, sh, np.where(zmask, np.nan, hdd))
        else:                                      # one window everywhere
            allmask = np.ones((H, W), bool)
            ok, sg, sh = accumulate(allmask, fixed_months)
            gdd = np.where(ok, sg, np.nan)
            hdd = np.where(ok, sh, np.nan)
        return gdd, hdd
    finally:
        zy.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp-dir", default=r"Z:/weather data/temperature")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "clean" / "gdd_hdd"))
    ap.add_argument("--year-range", default="1982-2016")
    ap.add_argument("--season-csv", default=None, help="zone37_growing_season.csv (step A)")
    ap.add_argument("--zone-shp", default=None, help="quhua.shp for the 37 zones")
    ap.add_argument("--months", default=None, help="fixed window, e.g. 4,5,6,7,8,9")
    ap.add_argument("--all-year", action="store_true")
    a = ap.parse_args()

    temp_dir = Path(a.temp_dir)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = map(int, a.year_range.split("-"))
    years = list(range(y0, y1 + 1))

    zy0 = TempZipYear(temp_dir, years[0])
    grid = zy0.any_profile(); zy0.close()
    print(f"[setup] grid {grid['shape']} res~0.1deg crs={grid['crs']} "
          f"bounds={tuple(round(x,2) for x in grid['bounds'])}")

    season_lookup, zone_raster, fixed_months = {}, None, None
    if a.season_csv and a.zone_shp:
        zone_raster = build_zone_raster(Path(a.zone_shp), grid)
        season_lookup = load_season_lookup(Path(a.season_csv))
        print(f"[setup] per-zone growing season: {len(season_lookup)} zones")
    elif a.all_year:
        fixed_months = list(range(1, 13)); print("[setup] window = full calendar year")
    else:
        fixed_months = [int(x) for x in (a.months or "4,5,6,7,8,9").split(",")]
        print(f"[setup] fixed window = months {fixed_months}")
    print(f"[setup] GDD = DD({BASE_T:.0f}) - DD({CAP_T:.0f}); HDD = DD({CAP_T:.0f})  [single-sine]")

    G, Hh = [], []
    t0 = time.time()
    for i, y in enumerate(years, 1):
        t = time.time()
        g, h = year_gdd_hdd(y, temp_dir, grid, zone_raster, season_lookup, fixed_months)
        print(f"  [{i:2d}/{len(years)}] {y} {time.time()-t:5.1f}s "
              f"GDD med={np.nanmedian(g):7.1f} max={np.nanmax(g):7.1f} | "
              f"HDD med={np.nanmedian(h):6.2f} max={np.nanmax(h):7.1f} "
              f"| n={int(np.isfinite(g).sum()):,d}", flush=True)
        G.append(g); Hh.append(h)

    tr = grid["transform"]; H, W = grid["shape"]
    lons = np.array([tr * (c + 0.5, 0.5) for c in range(W)])[:, 0].astype("float32")
    lats = np.array([tr * (0.5, r + 0.5) for r in range(H)])[:, 1].astype("float32")
    ds = xr.Dataset(
        {"gdd_growing_season": (("year", "lat", "lon"), np.stack(G)),
         "hdd_growing_season": (("year", "lat", "lon"), np.stack(Hh))},
        coords={"year": years, "lat": lats, "lon": lons},
        attrs={"method": "single-sine within-day degree days (Snyder 1985; "
                         "Schlenker & Roberts 2009; Ortiz-Bobea)",
               "gdd": f"DD({BASE_T})-DD({CAP_T})", "hdd": f"DD({CAP_T})",
               "unit": "degC*day accumulated over the growing season",
               "season": ("per-zone (step A)" if season_lookup else f"fixed months {fixed_months}")},
    )
    f = out_dir / f"gdd_hdd_growing_season_{y0}_{y1}.nc"
    if f.exists():
        f.unlink()
    # zlib needs netCDF4/h5netcdf; fall back to the uncompressed scipy backend
    try:
        import netCDF4  # noqa: F401
        ds.to_netcdf(f, engine="netcdf4",
                     encoding={v: {"zlib": True, "complevel": 4} for v in
                               ("gdd_growing_season", "hdd_growing_season")})
    except ImportError:
        ds.to_netcdf(f)          # scipy backend: NETCDF3, no compression
        print("      (netCDF4 not installed -> uncompressed NETCDF3)")
    print(f"\n[save] {f}  ({f.stat().st_size/1024**2:.1f} MB)")

    prof = dict(driver="GTiff", dtype="float32", nodata=-9999.0, width=W, height=H,
                count=1, crs=grid["crs"], transform=tr, compress="lzw")
    for s in [x for x in (y0, (y0 + y1) // 2, y1) if x in years]:
        k = years.index(s)
        for name, arr in (("gdd", G[k]), ("hdd", Hh[k])):
            b = arr.copy(); b[np.isnan(b)] = -9999.0
            with rasterio.open(out_dir / f"{name}_growing_season_{s}.tif", "w", **prof) as dst:
                dst.write(b, 1)
    print(f"[OK] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
