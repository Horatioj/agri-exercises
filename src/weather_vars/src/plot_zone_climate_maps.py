"""
Zone-level climate maps at 4 time points (1982, 1991, 2001, 2016).
One figure for GDD, one figure for rzsm_gs.

Spatial unit: 37 Chinese agricultural cropping-system zones (zone37_main / OBJECTID_1).
For each year, the zone value is the cropland-area-weighted mean of all counties
that fall in that zone (using climate_panel_1982_2016.csv).

Note: the user asked for 1981 but climate variables start in 1982, so 1982 is used
in place of 1981 and the panel is labelled "1982 (1981 unavailable)".
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from pathlib import Path

CLIM_CSV = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/climate_panel_1982_2016.csv'
ZONE_SHP = '/Users/zhangxinzhen/Desktop/TFP/Climate data/05 agricultural division/01 中国农业熟制区划quhua/quhua.shp'
PROV_SHP = '/Users/zhangxinzhen/Desktop/TFP/Climate data/00 county/省.shp'
OUT_DIR  = Path('/Users/zhangxinzhen/Desktop/TFP/Code/output/zone_climate_maps')
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [1982, 1991, 2001, 2016]
YEAR_LABEL = {1982: '1982*', 1991: '1991', 2001: '2001', 2016: '2016'}
# *1981 unavailable, 1982 substituted

CLIP_PCT = {  # GDD uses full min-max (initial-version look); rzsm mildly clipped
    'gdd_growing_season': (0, 100),
    'rzsm_gs':            (5, 95),
}

# ------------ load ------------
clim = pd.read_csv(CLIM_CSV)
gdf  = gpd.read_file(ZONE_SHP)[['OBJECTID_1', 'geometry']]
gdf  = gdf.rename(columns={'OBJECTID_1': 'zone'})
prov = gpd.read_file(PROV_SHP)[['geometry']].to_crs(gdf.crs)
# national outline = unary union of provinces
nation = gpd.GeoSeries([prov.geometry.union_all()], crs=gdf.crs)
print(f'shapefile: {len(gdf)} zones, crs={gdf.crs.name};  {len(prov)} provinces')

# Clean: keep rows where the climate variable is valid
valid = clim['valid_climate_sample'] == True
clim_v = clim.loc[valid & clim['zone37_main'].notna()].copy()
clim_v['zone'] = clim_v['zone37_main'].astype(int)


def zone_mean(year, var):
    """Cropland-weighted mean of `var` per zone for `year`. Returns dict {zone: value}."""
    s = clim_v[(clim_v['year'] == year) & clim_v[var].notna() & clim_v['cropland_area_km2'].notna()]
    s = s[s['cropland_area_km2'] > 0]
    g = s.groupby('zone').apply(
        lambda d: np.average(d[var], weights=d['cropland_area_km2']),
        include_groups=False,
    )
    return g.to_dict()


def render(var, title_var, cmap, out_name, mode='continuous'):
    # Compute zone means for all 4 years -> single colour scale
    year_means = {y: zone_mean(y, var) for y in YEARS}
    all_vals = np.array([v for d in year_means.values() for v in d.values()])

    if mode == 'continuous':
        lo_pct, hi_pct = CLIP_PCT[var]
        vmin = float(np.nanpercentile(all_vals, lo_pct))
        vmax = float(np.nanpercentile(all_vals, hi_pct))
        norm = Normalize(vmin=vmin, vmax=vmax)
        scale_label = f'colour clipped to {lo_pct}-{hi_pct}th percentile'
    elif mode == 'quantile_hi_emphasis':
        # 8 quantile bins: bottom 30% all in lightest shade, then progressively
        # darker shades with finer resolution at the top (85, 90, 95, 100)
        boundaries_pct = [0, 30, 50, 65, 75, 85, 90, 95, 100]
        boundary_vals  = [float(np.nanpercentile(all_vals, p)) for p in boundaries_pct]
        palette = ['#ffffcc', '#ffeda0', '#fed976', '#fd8d3c',
                   '#fc4e2a', '#e31a1c', '#bd0026', '#67000d']
        cmap = ListedColormap(palette)
        norm = BoundaryNorm(boundary_vals, ncolors=cmap.N, clip=True)
        scale_label = ('quantile bins: 0-30, 30-50, 50-65, 65-75, 75-85, 85-90, 90-95, 95-100 '
                       'percentile (top three bands emphasise high-temperature spread)')
    else:
        raise ValueError(mode)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, year in zip(axes.flat, YEARS):
        zone_val = year_means[year]
        gplot = gdf.copy()
        gplot[var] = gplot['zone'].map(zone_val)
        gplot.plot(column=var, ax=ax, cmap=cmap, norm=norm,
                   edgecolor='white', linewidth=0.25,
                   missing_kwds={'color': 'lightgrey', 'edgecolor': 'white', 'linewidth': 0.25})
        prov.boundary.plot(ax=ax, color='#555555', linewidth=0.35, alpha=0.6)
        nation.boundary.plot(ax=ax, color='black', linewidth=0.9)
        ax.set_title(YEAR_LABEL[year], fontsize=14, fontweight='bold', pad=4)
        ax.set_axis_off()

    # shared colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.018])
    if mode == 'continuous':
        cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
    else:
        cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                          spacing='proportional', ticks=boundary_vals)
        cb.ax.set_xticklabels([f'{v:.0f}' for v in boundary_vals], rotation=0)
    cb.set_label(f'{title_var}   ({scale_label})', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(f'{title_var} by Chinese agricultural cropping-system zone (37 zones)',
                 fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10,
                        wspace=0.0, hspace=0.08)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f'saved {out_path.name}   data-range=[{np.nanmin(all_vals):.3g}, {np.nanmax(all_vals):.3g}]')


render('gdd_growing_season',
       title_var='Growing-season GDD',
       cmap=None,  # palette built inside render() for quantile_hi_emphasis
       out_name='map_gdd_1982_1991_2001_2016.png',
       mode='quantile_hi_emphasis')

render('rzsm_gs',
       title_var='Growing-season root-zone soil moisture (rzsm_gs)',
       cmap='YlGnBu',
       out_name='map_rzsm_1982_1991_2001_2016.png')
