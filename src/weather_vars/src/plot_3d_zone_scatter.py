"""
3D scatter per agricultural zone (zone37_main).
  x = total input  (geometric mean of Laborday, Land_serv_q, capital_serv_q, Inter_all_real; log scale)
  y = total output (GVP_allagr_impute; log scale)
  z = soil moisture (rzsm_gs)
Each point = one county-year. Coloured by year.

Outputs
  output/3d_zone/zone_XX.png        — one file per zone (37 total)
  output/3d_zone/_overview_grid.png — 6x7 grid overview of all zones
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path

CLIM = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/climate_panel_1982_2016.csv'
PROD = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/cty_prod_account_agg_forAXN.dta'
OUT  = Path('/Users/zhangxinzhen/Desktop/TFP/Code/output/3d_zone')
OUT.mkdir(parents=True, exist_ok=True)

# ---------- load + merge ----------
clim = pd.read_csv(CLIM)
prod = pd.read_stata(PROD)
df = clim.merge(prod, left_on=['PAC', 'year'], right_on=['countyid', 'year'], how='inner')

inputs = ['Laborday_impute', 'Land_serv_q', 'capital_serv_q', 'Inter_all_real']
need = inputs + ['GVP_allagr_impute', 'rzsm_gs', 'zone37_main']
mask = (df[inputs] > 0).all(axis=1) & (df['GVP_allagr_impute'] > 0) \
       & df['rzsm_gs'].notna() & df['zone37_main'].notna() & df['valid_climate_sample']
df = df.loc[mask, need + ['year']].copy()

# composite input: geometric mean of the 4 inputs (= exp of mean of logs)
df['total_input'] = np.exp(np.log(df[inputs]).mean(axis=1))

# Pre-compute global colour normalisation across years
year_min, year_max = int(df['year'].min()), int(df['year'].max())
norm = Normalize(vmin=year_min, vmax=year_max)
cmap = cm.viridis

zones = sorted(df['zone37_main'].unique())
print(f'plotting {len(zones)} zones, total points = {len(df)}')


def plot_zone(ax, sub, title, *, label=True):
    sc = ax.scatter(np.log10(sub['total_input']),
                    np.log10(sub['GVP_allagr_impute']),
                    sub['rzsm_gs'],
                    c=sub['year'], cmap=cmap, norm=norm,
                    s=6, alpha=0.55, depthshade=False, edgecolors='none')
    ax.set_title(title, fontsize=10)
    if label:
        ax.set_xlabel('log10 Total input', fontsize=8, labelpad=2)
        ax.set_ylabel('log10 Output', fontsize=8, labelpad=2)
        ax.set_zlabel('rzsm_gs', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=22, azim=-58)
    return sc


# ---------- 1. individual files ----------
for z in zones:
    sub = df[df['zone37_main'] == z]
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = plot_zone(ax, sub, f'Zone {int(z)}   (n = {len(sub)} county-years)', label=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label('year', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f'zone_{int(z):02d}.png', dpi=170)
    plt.close(fig)
    print(f'  saved zone_{int(z):02d}.png')

# ---------- 2. overview grid (6 rows x 7 cols, 37 panels + 5 blanks) ----------
ROWS, COLS = 6, 7
fig = plt.figure(figsize=(COLS * 3.0, ROWS * 2.7))
for i, z in enumerate(zones):
    ax = fig.add_subplot(ROWS, COLS, i + 1, projection='3d')
    sub = df[df['zone37_main'] == z]
    sc = plot_zone(ax, sub, f'Zone {int(z)} (n={len(sub)})', label=False)
fig.suptitle('Input (log10) × Output (log10) × rzsm_gs, per Agricultural Zone\n'
             'colour = year', fontsize=14, y=0.995)
# single shared colourbar
cbar_ax = fig.add_axes([0.92, 0.06, 0.012, 0.88])
fig.colorbar(sc, cax=cbar_ax, label='year')
fig.tight_layout(rect=[0, 0, 0.91, 0.97])
fig.savefig(OUT / '_overview_grid.png', dpi=140)
plt.close(fig)
print('saved _overview_grid.png')
print('done, files in', OUT)
