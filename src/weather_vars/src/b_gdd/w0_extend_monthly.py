# -*- coding: utf-8 -*-
"""
Extend the monthly GDD/HDD archive backwards to 1981.

The I-O panel starts in 1981 but the archive was built for 1982-2016, so every
1981 row had blank weather.  The temperature ZIPs do start in 1981, so this is
purely a coverage gap, not a data limit.  Computes the missing year(s) with the
same single-sine method and merges them into the archive in year order.

  python w0_extend_monthly.py --years 1981
"""
from __future__ import annotations
import argparse, sys, time
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_step_gdd_hdd import TempZipYear, year_monthly

ROOT = Path(__file__).resolve().parents[3]
GDD_DIR = ROOT / "clean" / "gdd_hdd"
TEMP_DIR = Path(r"Z:/weather data/temperature")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1981")
    ap.add_argument("--archive", default=str(GDD_DIR / "monthly_gdd_hdd_1982_2016.npz"))
    a = ap.parse_args()
    add = [int(x) for x in a.years.split(",")]

    z = np.load(a.archive)
    years = list(z["years"]); MG = z["gdd_month"]; MH = z["hdd_month"]
    print(f"archive: {len(years)} years {years[0]}-{years[-1]}, {MG.shape}")

    zy = TempZipYear(TEMP_DIR, add[0]); grid = zy.any_profile(); zy.close()
    for y in add:
        if y in years:
            print(f"  {y} already present, skipping"); continue
        t = time.time()
        mg, mh = year_monthly(y, TEMP_DIR, grid)
        print(f"  {y}: {time.time()-t:.0f}s  annual GDD median "
              f"{np.nanmedian(np.nansum(mg, axis=0)):.0f}")
        years.append(y); MG = np.concatenate([MG, mg[None]]); MH = np.concatenate([MH, mh[None]])

    order = np.argsort(years)
    years = list(np.array(years)[order]); MG = MG[order]; MH = MH[order]
    out = GDD_DIR / f"monthly_gdd_hdd_{years[0]}_{years[-1]}.npz"
    np.savez_compressed(out, years=np.array(years), gdd_month=MG, hdd_month=MH)
    print(f"saved {out.name} ({out.stat().st_size/1024**2:.0f} MB): "
          f"{len(years)} years {years[0]}-{years[-1]}")


if __name__ == "__main__":
    main()
