"""
Zone-level production maps at 4 time points (1981, 1991, 2001, 2016).
Six figures (one per variable):
   1. Total input  (geometric mean of 4 inputs, county-level then zone mean)
   2. Total output (GVP_allagr_impute, zone mean per county)
   3. Labor       (Laborday_impute)
   4. Land        (Land_serv_q)
   5. Capital     (capital_serv_q)
   6. Intermediate (Inter_all_real)

Spatial unit: 37 Chinese agricultural cropping-system zones.
Colour scale: same 8-bin quantile-hi-emphasis palette as the GDD map
(bottom 30% lightest, top 5/10/15% progressively darker).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path

CLIM_CSV = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/climate_panel_1982_2016.csv'
PROD_DTA = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/cty_prod_account_agg_forAXN.dta'
ZONE_SHP = '/Users/zhangxinzhen/Desktop/TFP/Climate data/05 agricultural division/01 中国农业熟制区划quhua/quhua.shp'
PROV_SHP = '/Users/zhangxinzhen/Desktop/TFP/Climate data/00 county/省.shp'
OUT_DIR  = Path('/Users/zhangxinzhen/Desktop/TFP/Code/output/zone_production_maps')
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [1982, 2004, 2010, 2016]

# ---------- load shapefiles ----------
gdf  = gpd.read_file(ZONE_SHP)[['OBJECTID_1', 'geometry']].rename(columns={'OBJECTID_1': 'zone'})
prov = gpd.read_file(PROV_SHP)[['geometry']].to_crs(gdf.crs)
nation = gpd.GeoSeries([prov.geometry.union_all()], crs=gdf.crs)

# ---------- county -> zone mapping (from climate panel) ----------
clim = pd.read_csv(CLIM_CSV)
cty_zone = (clim.dropna(subset=['zone37_main'])
                .groupby('PAC')['zone37_main']
                .agg(lambda s: int(s.mode().iat[0]))
                .to_dict())
print(f'county->zone mapping: {len(cty_zone)} counties')

# ---------- load production, attach zone, build total-input index ----------
prod = pd.read_stata(PROD_DTA)
prod['zone'] = prod['countyid'].map(cty_zone)
prod = prod.dropna(subset=['zone'])
prod['zone'] = prod['zone'].astype(int)

inputs = ['Laborday_impute', 'Land_serv_q', 'capital_serv_q', 'Inter_all_real']
# Per-county-year geometric mean of the 4 inputs (only when all >0)
mask_pos = (prod[inputs] > 0).all(axis=1)
prod['total_input_geom'] = np.nan
prod.loc[mask_pos, 'total_input_geom'] = np.exp(np.log(prod.loc[mask_pos, inputs]).mean(axis=1))

# ---------- aggregation: unweighted zone mean of county-year values ----------
def zone_mean(df, year, var):
    s = df[(df['year'] == year) & df[var].notna() & (df[var] > 0)]
    return s.groupby('zone')[var].mean().to_dict()


# ---------- 8-bin quantile-hi-emphasis colour scale ----------
PCTILES = [0, 30, 50, 65, 75, 85, 90, 95, 100]

# Inputs: pink -> magenta -> deep purple (RdPu family)
PALETTE_INPUT = ['#fff7f3', '#fde0dd', '#fcc5c0', '#fa9fb5',
                 '#f768a1', '#dd3497', '#ae017e', '#7a0177']
# Output: pale yellow-green -> deep forest green (YlGn family)
PALETTE_OUTPUT = ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e',
                  '#78c679', '#41ab5d', '#238443', '#005a32']


def fmt(v):
    """Compact axis tick formatter."""
    if v >= 1e9: return f'{v/1e9:.1f}B'
    if v >= 1e6: return f'{v/1e6:.1f}M'
    if v >= 1e3: return f'{v/1e3:.0f}k'
    return f'{v:.1f}'


def render(var, title_var, out_name, palette):
    year_means = {y: zone_mean(prod, y, var) for y in YEARS}
    all_vals = np.array([v for d in year_means.values() for v in d.values()])
    boundary_vals = [float(np.nanpercentile(all_vals, p)) for p in PCTILES]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(boundary_vals, ncolors=cmap.N, clip=True)

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
        ax.set_title(str(year), fontsize=14, fontweight='bold', pad=4)
        ax.set_axis_off()

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.20, 0.05, 0.6, 0.018])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                      spacing='proportional', ticks=boundary_vals)
    cb.ax.set_xticklabels([fmt(v) for v in boundary_vals])
    cb.set_label(f'{title_var}   (quantile bins: {", ".join(f"{p}" for p in PCTILES[1:-1])} '
                 'percentile — top three bands emphasise high-intensity spread)',
                 fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(f'{title_var} by Chinese agricultural cropping-system zone (37 zones)',
                 fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10,
                        wspace=0.0, hspace=0.08)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f'saved {out_path.name}   range=[{np.nanmin(all_vals):.3g}, {np.nanmax(all_vals):.3g}]')


PLOTS = [
    ('total_input_geom',   'Total Input (geometric mean of 4 inputs)',           '1_map_total_input.png',   PALETTE_INPUT),
    ('GVP_allagr_impute',  'Total Agricultural Output (GVP)',                    '2_map_total_output.png',  PALETTE_OUTPUT),
    ('Laborday_impute',    'Labor (Laborday_impute, days)',                      '3_map_labor.png',         PALETTE_INPUT),
    ('Land_serv_q',        'Land services (Land_serv_q)',                        '4_map_land.png',          PALETTE_INPUT),
    ('capital_serv_q',     'Capital services (capital_serv_q)',                  '5_map_capital.png',       PALETTE_INPUT),
    ('Inter_all_real',     'Intermediate inputs (Inter_all_real)',               '6_map_intermediate.png',  PALETTE_INPUT),
]
for var, title, fname, palette in PLOTS:
    render(var, title, fname, palette)
