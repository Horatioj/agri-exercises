# -*- coding: utf-8 -*-
"""
STEP 0 — county cropland share from the CACD 30 m cropland rasters, EVERY YEAR,
and the UNION agricultural-county list the rest of the pipeline filters on.

WHY EVERY YEAR
    The agricultural-county sample was originally defined from cropland >= 15%
    in 2010 ALONE.  That is a single-year snapshot used to classify a 1981-2016
    panel, and it is the obvious thing for a referee to object to: a county that
    farmed heavily through the 1980s and was built over by 2010 gets dropped
    retrospectively, and the selection is correlated with exactly the structural
    change the paper studies.  Cropland is available annually (CACD 1986-2015),
    so the classification does not have to rest on one year.

THE RULE
    A county is agricultural if cropland >= THRESHOLD in ANY available year
    (the UNION).  Union rather than intersection or a mean, because:
      * the panel is 1981-2016 and a unit that was genuinely agricultural for
        part of it belongs in a panel about agricultural productivity;
      * dropping a county because it urbanised MID-sample deletes precisely the
        transition that makes the period interesting -- it conditions the sample
        on an outcome;
      * union is the WEAKEST selection, so it is the most defensible default.
    The per-year shares are all written out, so intersection ("agricultural in
    every year"), a k-of-n rule, or a different threshold can be built from the
    same table without re-reading a single raster.

METHOD
    For each county polygon and each year: read only that polygon's window of
    the raster (rasterio geometry_window), rasterize the polygon onto that
    window, and count.  Values are 1 = cropland, 0 = other, 255 = NoData.
        crop_pct = 100 * (#1) / (#1 + #0)
    NoData is excluded from the denominator, not counted as non-cropland -- an
    unclassified pixel is unknown, not "not farmland".
    Windowed reads are ~4 min/year at FULL 30 m resolution over 2,868 counties,
    so no decimation or approximation is used anywhere.

    Boundaries are the 2010 census county set (the same polygons the TFP maps
    and the weather join use), so `code` is comparable across every step.

RESUMABLE: one CSV per year under data/ag_county/by_year/; finished years are
skipped, so the run can be interrupted and restarted.

OUTPUTS (data/ag_county/)
  by_year/cropland_share_{year}.csv   code, name, year, crop_px, valid_px, crop_pct
  cropland_share_by_year.csv          all years stacked
  ag_counties_union.csv               code, name, n_yrs_above, first/last year
                                      above, min/mean/max crop_pct, is_ag
Run:  python src/00_cropland_share.py
      python src/00_cropland_share.py --threshold 15 --years 1986-2015
"""
from __future__ import annotations
import argparse, os, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np, pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_window, geometry_mask
import _common as C
from _county_match import SHP_2010, load_boundaries

CACD_DIR = r"Z:/weather data/cropland"
OUT_DIR = os.path.join(C.DATA, "ag_county")
YEAR_DIR = os.path.join(OUT_DIR, "by_year")
NODATA = 255
CROP_VALUE = 1


class ReadFailed(Exception):
    """A raster window could not be read after retries."""


def county_share(src, geom, path, retries=4):
    """(crop_px, valid_px) for one polygon, full raster resolution.

    The rasters live on a network drive, where TIFFReadEncodedTile occasionally
    fails transiently (`IReadBlock failed at X offset ..., Y offset ...`).  That
    is not corruption -- re-reading the same tile succeeds -- but an unguarded
    read aborts a two-hour run partway through.  So the read is retried with a
    fresh handle, and a window that still fails raises ReadFailed so the caller
    can record the county as MISSING rather than silently scoring it 0% cropland
    (which would quietly drop a real agricultural county from the sample).
    """
    try:
        win = geometry_window(src, [geom.__geo_interface__])
    except Exception:                      # polygon entirely outside the raster
        return 0, 0
    if win.width <= 0 or win.height <= 0:
        return 0, 0

    a = None
    for k in range(retries):
        try:
            if k == 0:
                a = src.read(1, window=win)
            else:                          # reopen: a stale handle can stay bad
                time.sleep(1.5 * k)
                with rasterio.open(path) as s2:
                    a = s2.read(1, window=win)
            break
        except Exception as e:
            if k == retries - 1:
                raise ReadFailed(str(e)[:160])
    if a is None or a.size == 0:
        return 0, 0
    m = geometry_mask([geom.__geo_interface__], a.shape,
                      src.window_transform(win), invert=True)
    v = a[m]
    valid = v != NODATA
    return int((v[valid] == CROP_VALUE).sum()), int(valid.sum())


def run_year(year, gdf, out_csv):
    path = os.path.join(CACD_DIR, f"CACD-{year}.tif")
    if not os.path.exists(path):
        print(f"  [{year}] raster missing, skipped")
        return None
    t0 = time.time()
    rows, failed = [], []
    with rasterio.open(path) as src:
        for i, r in enumerate(gdf.itertuples(), 1):
            try:
                cp, vp = county_share(src, r.geometry, path)
                pct = 100.0 * cp / vp if vp else np.nan
            except ReadFailed as e:
                # NaN, never 0: an unreadable window is unknown, not "no cropland"
                cp, vp, pct = np.nan, np.nan, np.nan
                failed.append((int(r.code), str(e)))
            rows.append((int(r.code), r.name_cn, year, cp, vp, pct))
            if i % 500 == 0:
                print(f"      {i}/{len(gdf)} counties  {time.time()-t0:.0f}s", flush=True)
    d = pd.DataFrame(rows, columns=["code", "name", "year", "crop_px",
                                    "valid_px", "crop_pct"])
    if failed:
        # do NOT write a partial year as if it were complete
        print(f"  [{year}] {len(failed)} counties unreadable after retries "
              f"-> year NOT saved; re-run to retry. First: {failed[:3]}", flush=True)
        return None
    d.to_csv(out_csv, index=False, encoding="utf-8-sig")
    ok = d.crop_pct.notna()
    print(f"  [{year}] {time.time()-t0:.0f}s  {int(ok.sum())} counties with data; "
          f"median {d.loc[ok,'crop_pct'].median():.1f}%  "
          f">=15%: {int((d.crop_pct >= 15).sum())}", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1986-2015")
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    years = list(range(y0, y1 + 1))
    os.makedirs(YEAR_DIR, exist_ok=True)

    # the shared loader: fixes the 县级码=0 municipalities, applies SHP_CODE_FIX,
    # drops HK/Macao/Taiwan, dissolves duplicate codes, and reprojects to 4326
    # (the rasters are 4326; leaving the polygons in Web Mercator silently
    # returns empty windows)
    g = load_boundaries()
    g = g.rename(columns={"gname": "name_cn"})
    g = g[g.geometry.notna()].copy()
    print(f"boundaries: {len(g)} county polygons (2010 census set)")
    print(f"years: {years[0]}-{years[-1]}  threshold: cropland >= {a.threshold:g}%\n")

    # preflight: every raster opens and its grid matches, before spending hours
    todo = [y for y in years
            if a.overwrite or not os.path.exists(
                os.path.join(YEAR_DIR, f"cropland_share_{y}.csv"))]
    print(f"{len(years) - len(todo)} year(s) already done; {len(todo)} to do")
    bad = []
    for y in todo:
        pth = os.path.join(CACD_DIR, f"CACD-{y}.tif")
        if not os.path.exists(pth):
            bad.append((y, "missing")); continue
        try:
            with rasterio.open(pth) as s:
                _ = s.crs, s.width
        except Exception as e:
            bad.append((y, str(e)[:80]))
    if bad:
        print(f"  preflight problems: {bad}")
    print()

    incomplete = []
    for y in todo:
        f = os.path.join(YEAR_DIR, f"cropland_share_{y}.csv")
        try:
            if run_year(y, g, f) is None and os.path.exists(
                    os.path.join(CACD_DIR, f"CACD-{y}.tif")):
                incomplete.append(y)
        except Exception as e:
            # one flaky year must not abandon the other 29
            print(f"  [{y}] ABORTED: {type(e).__name__}: {str(e)[:140]}", flush=True)
            incomplete.append(y)
    if incomplete:
        print(f"\nINCOMPLETE years (re-run to retry, finished years are skipped): "
              f"{incomplete}\n")

    # ---------------------------------------------------------------- stack
    files = sorted(glob.glob(os.path.join(YEAR_DIR, "cropland_share_*.csv")))
    if not files:
        print("no per-year files produced"); return
    alld = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                     ignore_index=True)
    alld.to_csv(os.path.join(OUT_DIR, "cropland_share_by_year.csv"),
                index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- the union
    d = alld.dropna(subset=["crop_pct"]).copy()
    d["above"] = d.crop_pct >= a.threshold
    u = (d.groupby("code")
           .agg(name=("name", "last"),
                n_yrs=("year", "nunique"),
                n_yrs_above=("above", "sum"),
                first_above=("year", lambda s: s[d.loc[s.index, "above"]].min()),
                last_above=("year", lambda s: s[d.loc[s.index, "above"]].max()),
                min_pct=("crop_pct", "min"),
                mean_pct=("crop_pct", "mean"),
                max_pct=("crop_pct", "max")).reset_index())
    u["is_ag"] = u.n_yrs_above > 0                    # UNION rule
    u["is_ag_always"] = u.n_yrs_above == u.n_yrs      # intersection, for reference
    u.to_csv(os.path.join(OUT_DIR, "ag_counties_union.csv"),
             index=False, encoding="utf-8-sig")

    # Drop-in replacement for the old single-year ag list.  01_resolve_roster.py
    # needs exactly two columns from it: `code` and `地名`.
    #
    # CODING VINTAGE.  These codes come from the 2010 census boundary set, which
    # is NOT the vintage the delivered ag_counties_crop15.csv used: that file
    # carries later codes for units renamed after 2010 (蓟县 120225 -> 蓟州区
    # 120119, 宁乡县 430124 -> 宁乡市 430182, 安县 510724 -> 安州区 510705,
    # 射洪县 510922 -> 射洪市 510981).  The 2010 vintage is the right one here:
    # the production panel is 1981-2016, so it speaks the older codes, and
    # 01_resolve_roster.py matches on (province, name) as well as code, which
    # absorbs the rest.
    aglist = u.loc[u.is_ag, ["code", "name"]].rename(columns={"name": "地名"})
    aglist.to_csv(os.path.join(OUT_DIR, "ag_counties_union_aglist.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"drop-in ag list -> ag_counties_union_aglist.csv ({len(aglist):,d} counties)")

    n_union = int(u.is_ag.sum())
    n_inter = int(u.is_ag_always.sum())
    print(f"\nUNION rule (>= {a.threshold:g}% in ANY year): {n_union:,d} of {len(u):,d} counties")
    print(f"  intersection (every year), for reference : {n_inter:,d}")
    print(f"  gained by using all years vs a single-year snapshot: {n_union - n_inter:,d} counties "
          f"that qualify in some years but not all")

    # how much does the answer depend on which single year you pick?
    per_year = d.groupby("year")["above"].sum()
    print(f"  counties >= {a.threshold:g}% in a SINGLE year: min {per_year.min():,d} "
          f"({per_year.idxmin()}), max {per_year.max():,d} ({per_year.idxmax()})")
    if 2010 in per_year.index:
        print(f"  the old 2010-only rule gave {int(per_year.loc[2010]):,d}")
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'ag_counties_union.csv')}")


if __name__ == "__main__":
    main()
