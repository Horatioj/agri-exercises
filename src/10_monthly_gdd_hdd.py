# -*- coding: utf-8 -*-
"""
STEP 10 — per-pixel MONTHLY GDD/HDD grids, 1981-2016.  The one expensive pass
over the daily temperature rasters; everything downstream is array arithmetic.

DEGREE-DAY METHOD.  The temperature response is applied WITHIN the day, not to
the daily mean, via single-sine integration (Snyder 1985 / Schlenker & Roberts
2009, as in Ortiz-Bobea):
        GDD = DD(8) - DD(32)      HDD = DD(32)
Applying it to tmean=(tmax+tmin)/2 instead — as the original B step did —
biases GDD (+4.5% overall on 1982 rasters, up to +48% in shoulder months) and
makes harmful heat unrecoverable, because a day peaking at 33C and one peaking
at 45C both cap to the same number.  See _degree_days.py for the closed form,
its unit tests, and the validation against real rasters.

WHY MONTHLY.  Saving 12 grids per year makes every season definition cheap
afterwards — a growing season is just a set of months, so the multi-peak
cropland season (11_growing_season.py), a fixed window, or the whole year can
all be derived without touching the 25,550 daily rasters again.  The monthly
archive is the deliverable; the seasonal .nc written alongside it is a sanity
artefact for the fixed-window case only.

Temperature ships as per-year ZIPs (`1982_max.zip`, 730 members each); they are
read in place, no extraction.  The grid is probed from the first available
member, never assumed (the data start in 1981, not 1980).

OUTPUTS (--out, default src/clean/gdd_hdd/)
  monthly_gdd_hdd_{y0}_{y1}.npz         years, gdd_month, hdd_month  (12,H,W)/yr
  gdd_hdd_growing_season_{y0}_{y1}.nc   fixed-window season totals
  gdd_growing_season_{year}.tif / hdd_growing_season_{year}.tif   sample years

USAGE
  python src/10_monthly_gdd_hdd.py --year-range 1982-2016
  python src/10_monthly_gdd_hdd.py --extend 1981     # merge a year into the archive
"""
from __future__ import annotations
import argparse
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
from _degree_days import dd_above, BASE_T, CAP_T

ROOT = Path(__file__).resolve().parent               # ...\src
GDD_DIR = ROOT / "clean" / "gdd_hdd"
TEMP_DIR = Path(r"Z:/weather data/temperature")
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


def grid_from(temp_dir: Path, year: int) -> dict:
    zy = TempZipYear(temp_dir, year)
    try:
        return zy.any_profile()
    finally:
        zy.close()


# ---------------------------------------------------------------- yearly core
def year_monthly(year: int, temp_dir: Path, grid: dict):
    """MONTHLY GDD/HDD grids (12,H,W) for one year -- the expensive pass.

    NaN propagation is strict: if any day of a month is missing or NoData at a
    pixel, that pixel's whole month is NaN rather than a silent partial sum.
    """
    H, W = grid["shape"]
    zy = TempZipYear(temp_dir, year)
    try:
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
                gdd_day, hdd_day = dd8 - dd32, dd32
                nan = np.isnan(gdd_day)
                mg[m] += np.where(nan, 0.0, gdd_day)
                mh[m] += np.where(nan, 0.0, hdd_day)
                bad[m] |= nan
            d += timedelta(days=1)
        for m in range(12):
            mg[m] = np.where(bad[m], np.nan, mg[m])
            mh[m] = np.where(bad[m], np.nan, mh[m])
        return mg, mh
    finally:
        zy.close()


def season_from_monthly(mg, mh, months):
    """Sum the monthly grids over `months` -> (gdd, hdd) season totals."""
    _, H, W = mg.shape
    sg = np.zeros((H, W), "float32"); sh = np.zeros((H, W), "float32")
    miss = np.zeros((H, W), bool)
    for mo in months:
        sg += np.where(np.isnan(mg[mo - 1]), 0.0, mg[mo - 1])
        sh += np.where(np.isnan(mh[mo - 1]), 0.0, mh[mo - 1])
        miss |= np.isnan(mg[mo - 1])
    return np.where(miss, np.nan, sg), np.where(miss, np.nan, sh)


def save_archive(out_dir: Path, years, MG, MH) -> Path:
    order = np.argsort(years)
    years = list(np.asarray(years)[order])
    MG, MH = np.asarray(MG)[order], np.asarray(MH)[order]
    f = out_dir / f"monthly_gdd_hdd_{years[0]}_{years[-1]}.npz"
    np.savez_compressed(f, years=np.array(years), gdd_month=MG, hdd_month=MH)
    print(f"[save] {f.name} ({f.stat().st_size/1024**2:.0f} MB) — {len(years)} years "
          f"{years[0]}-{years[-1]}; reusable for any season window")
    return f


# --------------------------------------------------------------------- extend
def extend(archive: Path, add_years, temp_dir: Path, out_dir: Path):
    """Compute missing years with the same method and merge them into an
    existing archive in year order.  The I-O panel starts in 1981 while the
    archive was first built for 1982-2016, so 1981 was a coverage gap, not a
    data limit -- the temperature ZIPs do start in 1981."""
    z = np.load(archive)
    years = list(z["years"]); MG, MH = z["gdd_month"], z["hdd_month"]
    print(f"archive: {len(years)} years {years[0]}-{years[-1]}, {MG.shape}")

    grid = grid_from(temp_dir, add_years[0])
    for y in add_years:
        if y in years:
            print(f"  {y} already present, skipping"); continue
        t = time.time()
        mg, mh = year_monthly(y, temp_dir, grid)
        print(f"  {y}: {time.time()-t:.0f}s  annual GDD median "
              f"{np.nanmedian(np.nansum(mg, axis=0)):.0f}")
        years.append(y)
        MG = np.concatenate([MG, mg[None]]); MH = np.concatenate([MH, mh[None]])
    save_archive(out_dir, years, MG, MH)


# ---------------------------------------------------------------------- build
def build(years, months, temp_dir: Path, out_dir: Path):
    grid = grid_from(temp_dir, years[0])
    print(f"[setup] grid {grid['shape']} res~0.1deg crs={grid['crs']} "
          f"bounds={tuple(round(x,2) for x in grid['bounds'])}")
    print(f"[setup] GDD = DD({BASE_T:.0f}) - DD({CAP_T:.0f}); HDD = DD({CAP_T:.0f})  [single-sine]")
    print(f"[setup] sanity window = months {months}")

    G, Hh, MG, MH = [], [], [], []
    t0 = time.time()
    for i, y in enumerate(years, 1):
        t = time.time()
        mg, mh = year_monthly(y, temp_dir, grid)
        g, h = season_from_monthly(mg, mh, months)
        MG.append(mg); MH.append(mh); G.append(g); Hh.append(h)
        print(f"  [{i:2d}/{len(years)}] {y} {time.time()-t:5.1f}s "
              f"GDD med={np.nanmedian(g):7.1f} max={np.nanmax(g):7.1f} | "
              f"HDD med={np.nanmedian(h):6.2f} max={np.nanmax(h):7.1f} "
              f"| n={int(np.isfinite(g).sum()):,d}", flush=True)

    save_archive(out_dir, years, MG, MH)

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
               "unit": "degC*day accumulated over the window",
               "season": f"fixed months {months}"},
    )
    f = out_dir / f"gdd_hdd_growing_season_{years[0]}_{years[-1]}.nc"
    if f.exists():
        f.unlink()
    # zlib needs netCDF4/h5netcdf; fall back to the uncompressed scipy backend
    try:
        import netCDF4  # noqa: F401
        ds.to_netcdf(f, engine="netcdf4",
                     encoding={v: {"zlib": True, "complevel": 4} for v in
                               ("gdd_growing_season", "hdd_growing_season")})
    except ImportError:
        ds.to_netcdf(f)
        print("      (netCDF4 not installed -> uncompressed NETCDF3)")
    print(f"[save] {f}  ({f.stat().st_size/1024**2:.1f} MB)")

    prof = dict(driver="GTiff", dtype="float32", nodata=-9999.0, width=W, height=H,
                count=1, crs=grid["crs"], transform=tr, compress="lzw")
    for s in [x for x in (years[0], years[len(years) // 2], years[-1]) if x in years]:
        k = years.index(s)
        for name, arr in (("gdd", G[k]), ("hdd", Hh[k])):
            b = arr.copy(); b[np.isnan(b)] = -9999.0
            with rasterio.open(out_dir / f"{name}_growing_season_{s}.tif", "w", **prof) as dst:
                dst.write(b, 1)
    print(f"[OK] done in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp-dir", default=str(TEMP_DIR))
    ap.add_argument("--out", default=str(GDD_DIR))
    ap.add_argument("--year-range", default="1981-2016")
    ap.add_argument("--months", default="4,5,6,7,8,9",
                    help="fixed sanity window for the .nc; the archive is always monthly")
    ap.add_argument("--all-year", action="store_true")
    ap.add_argument("--extend", default=None,
                    help="comma-separated years to compute and merge into --archive")
    ap.add_argument("--archive", default=str(GDD_DIR / "monthly_gdd_hdd_1981_2016.npz"))
    a = ap.parse_args()

    temp_dir = Path(a.temp_dir)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    if a.extend:
        extend(Path(a.archive), [int(x) for x in a.extend.split(",")], temp_dir, out_dir)
        return

    y0, y1 = map(int, a.year_range.split("-"))
    months = list(range(1, 13)) if a.all_year else [int(x) for x in a.months.split(",")]
    build(list(range(y0, y1 + 1)), months, temp_dir, out_dir)


if __name__ == "__main__":
    main()
