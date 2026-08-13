# -*- coding: utf-8 -*-
"""
W4: (a) WHOLE-YEAR GDD/HDD alongside the growing-season version, and
    (b) county cropland share of administrative area, with maps.

(a) WHY ANNUAL AS WELL
    Du, Chen & Hou (2025, JAAEA) use exactly the sine method implemented here
    (Baskerville & Emin 1969), but accumulate over the WHOLE YEAR for aggregate
    county outcomes, reserving the crop growing season for crop-specific
    acreage.  Our outcome (county TFP) is an aggregate, so the annual version is
    defensible on its own terms AND it sidesteps the unresolved growing-season
    question entirely -- a usable first result that can be revisited once the
    season definition is settled.  It costs nothing: the monthly archive is
    already saved, so this is a 12-month sum.

(b) CROPLAND SHARE
    Cropland fraction weights only make sense if we can see how little cropland
    the vast western counties actually have.  This computes, per county,
    cropland area / administrative area, and maps it next to the growing-season
    GDD so the two can be read together.

OUTPUTS
  src/clean/gdd_hdd/county_year_gdd_hdd_annual.csv
  src/clean/gdd_hdd/county_cropland_share.csv
  src/figures/fig_map_cropland_share.png
"""
from __future__ import annotations
import argparse, sys, time
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import _common as C
from _county_match import load_boundaries

GDD_DIR = ROOT / "clean" / "gdd_hdd"
TEMP_TR = from_origin(70.0, 55.0, 0.1, 0.1)
TEMP_SHAPE = (400, 700)
ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-annual", action="store_true")
    a = ap.parse_args()
    from exactextract import exact_extract
    g = load_boundaries()

    # ---------------------------------------------------------- (b) crop share
    print("[b] county cropland share of administrative area")
    cw_f = GDD_DIR / "cropland_frac_tempgrid.tif"
    # cropland AREA per cell = fraction * cell area; sum over county / county area
    with rasterio.open(cw_f) as s:
        cf = s.read(1)
    lat_c = 55.0 - (np.arange(TEMP_SHAPE[0]) + 0.5) * 0.1
    # 0.1deg cell area in km^2, varies with latitude
    cell_km2 = (111.32 * 0.1) * (111.32 * 0.1) * np.cos(np.radians(lat_c))
    crop_km2 = cf * cell_km2[:, None]
    tmp = GDD_DIR / "_tmp_crop.tif"
    prof = dict(driver="GTiff", dtype="float32", nodata=-9999.0,
                width=TEMP_SHAPE[1], height=TEMP_SHAPE[0], count=1,
                crs="EPSG:4326", transform=TEMP_TR)
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(crop_km2.astype("float32"), 1)
    cs = exact_extract(str(tmp), g, ops=["sum"], include_cols=["code"], output="pandas")
    cs = cs.rename(columns={[c for c in cs.columns if c != "code"][0]: "cropland_km2"})
    tmp.unlink(missing_ok=True)
    ga = g.to_crs(ALBERS)
    cs["admin_km2"] = (ga.geometry.area / 1e6).values
    cs["cropland_share"] = (cs.cropland_km2 / cs.admin_km2).clip(0, 1)
    cs.to_csv(GDD_DIR / "county_cropland_share.csv", index=False)
    print(f"    median share {cs.cropland_share.median():.3f}; "
          f"counties <5% cropland: {int((cs.cropland_share<0.05).sum()):,d}")
    prov = {65: "Xinjiang", 54: "Tibet", 63: "Qinghai", 62: "Gansu", 15: "InnerMongolia",
            41: "Henan", 37: "Shandong", 32: "Jiangsu"}
    cs["prov"] = cs.code // 10000
    r = cs.groupby("prov").cropland_share.median()
    print("    median cropland share by province:")
    for p, nm in prov.items():
        if p in r.index:
            print(f"      {nm:14s} {r[p]*100:5.1f}%")

    # ---------------------------------------------------------- (a) annual G/H
    if not a.skip_annual:
        print("[a] whole-year GDD/HDD (Du et al. 2025 aggregate-outcome convention)")
        z = np.load(GDD_DIR / "monthly_gdd_hdd_1981_2016.npz")
        years = z["years"]; MG = z["gdd_month"]; MH = z["hdd_month"]
        # ONE MULTI-BAND RASTER PER YEAR, ONE exact_extract CALL.
        # The per-series version issued four passes per year (gdd/hdd x
        # weighted_mean/mean) over 2,865 polygons -- ~145 s/year, and the run
        # died partway through.  Stacking the two variables as bands and asking
        # for both ops at once rasterises the polygons once per year instead of
        # four times.  Same numbers, ~4x fewer passes.
        names = ["gdd_annual", "hdd_annual"]
        prof_mb = dict(prof); prof_mb["count"] = len(names)
        rows = []
        t0 = time.time()
        for i, y in enumerate(years):
            stack = []
            for mm in (MG[i], MH[i]):
                tot = np.where(np.isnan(mm).any(axis=0), np.nan, np.nansum(mm, axis=0))
                stack.append(tot.astype("float32"))
            arr = np.stack(stack); arr[np.isnan(arr)] = -9999.0
            with rasterio.open(tmp, "w", **prof_mb) as d:
                d.write(arr)
            r = exact_extract(str(tmp), g, ops=["weighted_mean", "mean"],
                              weights=str(cw_f), include_cols=["code"], output="pandas")
            # column names are DISCOVERED: with a weights raster exact_extract
            # emits band_<n>_<weightsband>_weighted_mean, not band_<n>_weighted_mean
            ren = {}
            for b_, n_ in enumerate(names, start=1):
                pre = f"band_{b_}_"
                wcol = [c for c in r.columns if c.startswith(pre) and c.endswith("weighted_mean")]
                acol = [c for c in r.columns if c.startswith(pre) and c.endswith("mean")
                        and "weighted" not in c]
                if len(wcol) != 1 or len(acol) != 1:
                    raise RuntimeError(f"cannot map band {b_} ({n_}); columns were "
                                       f"{list(r.columns)[:8]}")
                ren[wcol[0]] = n_
                ren[acol[0]] = n_ + "_area"
            r = r.rename(columns=ren)[["code"] + [ren[k] for k in ren]]
            rows.append(r.assign(year=int(y)))
            if (i + 1) % 6 == 0 or i == 0:
                print(f"    {y} ({i+1}/{len(years)}) {time.time()-t0:.0f}s", flush=True)
        tmp.unlink(missing_ok=True)
        ann = pd.concat(rows, ignore_index=True).rename(columns={"code": "map_code"})
        ann.to_csv(GDD_DIR / "county_year_gdd_hdd_annual.csv", index=False)
        print(f"    saved county_year_gdd_hdd_annual.csv ({len(ann):,d} rows)")

    # ---------------------------------------------------------- map
    print("[map] cropland share")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gm = g.merge(cs[["code", "cropland_share"]], on="code", how="left").to_crs(ALBERS)
    fig, ax = plt.subplots(figsize=(11, 8))
    gm.plot(column="cropland_share", ax=ax, cmap="YlGn", vmin=0, vmax=0.8,
            linewidth=0, missing_kwds=dict(color="0.9"))
    gm.dissolve(by=gm.code // 10000).boundary.plot(ax=ax, color="white", linewidth=.35)
    sm = plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(0, 0.8))
    cb = fig.colorbar(sm, ax=ax, shrink=.6, pad=.01)
    cb.set_label("cropland / administrative area", fontsize=9)
    ax.set_title("County cropland share of administrative area (CACD 2000)\n"
                 "western counties are vast but barely cropped — which is why GDD/HDD "
                 "must be cropland-weighted", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    p = ROOT / "figures" / "fig_map_cropland_share.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("    saved", p.name)
    print("[OK] W4 done")


if __name__ == "__main__":
    main()
