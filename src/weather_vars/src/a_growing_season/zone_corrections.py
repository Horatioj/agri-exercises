# -*- coding: utf-8 -*-
"""
Known defects in 中国农业熟制区划quhua (the 37-zone cropping-system shapefile),
and how to handle them.

DEFECT 1 -- OBJECTID_1 = 3 and 4 have their NAME fields SWAPPED in the source
            data.  Any logic that keys off the zone NAME (e.g. reading the
            cropping system 一熟/二熟/三熟 out of the name) is therefore wrong
            for those two zones until the swap is undone.  The geometry and the
            OBJECTID_1 codes themselves are fine, so anything keyed purely on
            OBJECTID_1 (the GDD/HDD accumulation) is unaffected.

DEFECT 2 -- The shapefile OMITS Hainan province and the Leizhou peninsula.
            Those pixels fall in no zone, so with a per-zone growing-season
            window they silently receive NaN GDD/HDD and drop out of the panel.
            This is not a rounding issue: it removes tropical China, which is
            the highest-cropping-intensity part of the country (三熟).

            Two ways out, in order of preference:
              (a) define the growing season PER PIXEL from the NDVI climatology
                  (peak_month.tif from step A) and skip the zoning entirely --
                  Hainan and Leizhou have NDVI, so they are covered for free,
                  and this also removes within-zone phenology error;
              (b) if the zoning must be kept, fall back to nearest-zone
                  assignment for unzoned cropland pixels (`fill_unzoned`), which
                  attaches Hainan/Leizhou to the adjacent 华南 triple-cropping
                  zone.  Cruder, but keeps the counties in the panel.

Never leave them unzoned: silently dropping a region is the worst option.
"""
from __future__ import annotations
import numpy as np

# OBJECTID_1 values whose NAME fields must be exchanged
SWAPPED_NAME_IDS = (3, 4)


def fix_zone_names(gdf, id_col: str = "OBJECTID_1", name_col: str = "NAME"):
    """Undo the NAME swap between OBJECTID_1 = 3 and 4. Returns a copy."""
    out = gdf.copy()
    a, b = SWAPPED_NAME_IDS
    ia = out.index[out[id_col] == a]
    ib = out.index[out[id_col] == b]
    if len(ia) == 1 and len(ib) == 1:
        va = out.loc[ia[0], name_col]
        vb = out.loc[ib[0], name_col]
        out.loc[ia[0], name_col] = vb
        out.loc[ib[0], name_col] = va
        print(f"[zone_corrections] swapped NAME of OBJECTID_1 {a} <-> {b}: "
              f"{va!r} <-> {vb!r}")
    else:
        print(f"[zone_corrections] WARNING: could not find unique rows for "
              f"OBJECTID_1 {a}/{b}; NAME swap NOT applied")
    return out


def fill_unzoned(zone_raster: np.ndarray, valid_mask: np.ndarray | None = None,
                 max_dist_px: int = 40) -> np.ndarray:
    """Assign a zone to pixels the shapefile misses (Hainan, Leizhou peninsula).

    zone_raster : int array, 0 = no zone.
    valid_mask  : where a zone is actually required (e.g. cropland > 0). If
                  None, every 0-pixel inside the raster is filled.
    Nearest-neighbour fill via a distance transform, capped at `max_dist_px`
    so that ocean / far-outside pixels are not given a spurious zone.
    """
    from scipy.ndimage import distance_transform_edt
    need = (zone_raster == 0)
    if valid_mask is not None:
        need &= valid_mask
    if not need.any():
        return zone_raster
    # indices of the nearest non-zero zone pixel
    dist, (iy, ix) = distance_transform_edt(zone_raster == 0, return_indices=True)
    filled = zone_raster.copy()
    take = need & (dist <= max_dist_px)
    filled[take] = zone_raster[iy[take], ix[take]]
    print(f"[zone_corrections] filled {int(take.sum()):,d} unzoned pixels "
          f"from the nearest zone (<= {max_dist_px} px)")
    return filled


def report_unzoned(zone_raster: np.ndarray, cropland: np.ndarray | None = None):
    """Diagnostic: how much (cropland) area the zoning misses."""
    n0 = int((zone_raster == 0).sum())
    print(f"[zone_corrections] pixels with no zone: {n0:,d} "
          f"({100*n0/zone_raster.size:.1f}% of grid)")
    if cropland is not None:
        m = (zone_raster == 0) & (cropland > 0)
        print(f"[zone_corrections] CROPLAND pixels with no zone: {int(m.sum()):,d} "
              f"-- these would become NaN (expect Hainan + Leizhou)")
