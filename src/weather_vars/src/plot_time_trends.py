"""
Plot 4 mean time-trend figures:
  1. mean GDD (growing season)
  2. mean root-zone soil moisture (rzsm_gs)
  3. mean total agricultural output (GVP_allagr_impute)
  4. mean total input + 4 input components (labor, land, capital, intermediate)
     All input series indexed to base year = 100 because units differ.
Output: /Users/zhangxinzhen/Desktop/TFP/Code/output/trends/*.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

CLIM_CSV = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/climate_panel_1982_2016.csv'
PROD_DTA = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/cty_prod_account_agg_forAXN.dta'
OUT_DIR  = Path('/Users/zhangxinzhen/Desktop/TFP/Code/output/trends')
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'figure.dpi': 130,
    'savefig.dpi': 200,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# --------------- Load ---------------
clim = pd.read_csv(CLIM_CSV)
prod = pd.read_stata(PROD_DTA)

# Keep climate rows that are valid (cropland-weighted sample)
clim_valid = clim[clim['valid_climate_sample'] == True].copy()

# --------------- 1. mean GDD ---------------
gdd_yr = clim_valid.groupby('year')['gdd_growing_season'].mean()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(gdd_yr.index, gdd_yr.values, color='#d1495b', lw=2, marker='o', ms=3)
ax.set_xlabel('Year')
ax.set_ylabel('Mean GDD (growing season)')
ax.set_title('Mean Growing-Season GDD across counties, 1982-2016')
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT_DIR / '1_mean_gdd_trend.png')
plt.close(fig)

# --------------- 2. mean rzsm_gs ---------------
rz_yr = clim_valid.groupby('year')['rzsm_gs'].mean()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(rz_yr.index, rz_yr.values, color='#2e86ab', lw=2, marker='o', ms=3)
ax.set_xlabel('Year')
ax.set_ylabel('Mean root-zone soil moisture (rzsm_gs)')
ax.set_title('Mean Growing-Season Root-Zone Soil Moisture, 1982-2016')
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT_DIR / '2_mean_rzsm_gs_trend.png')
plt.close(fig)

# --------------- 3. mean total output ---------------
out_yr = prod.groupby('year')['GVP_allagr_impute'].mean()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(out_yr.index, out_yr.values, color='#386641', lw=2, marker='o', ms=3)
ax.set_xlabel('Year')
ax.set_ylabel('Mean GVP_allagr_impute (per county)')
ax.set_title('Mean Total Agricultural Output across counties, 1981-2016')
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT_DIR / '3_mean_output_trend.png')
plt.close(fig)

# --------------- 4. inputs: 2x2 subplots (each in original units) ---------------
input_cols = {
    'Laborday_impute':  ('Labor (days)',          '#bc4749'),
    'Land_serv_q':      ('Land services (q)',     '#6a994e'),
    'capital_serv_q':   ('Capital services (q)',  '#386fa4'),
    'Inter_all_real':   ('Intermediate (real)',   '#f4a261'),
}

# Replace 0 with NaN before mean so empty-input counties do not drag the mean down to 0
inp_yr = prod[['year'] + list(input_cols)].copy()
for c in input_cols:
    inp_yr.loc[inp_yr[c] == 0, c] = np.nan
inp_yr = inp_yr.groupby('year').mean()

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (col, (label, color)) in zip(axes.flat, input_cols.items()):
    ax.plot(inp_yr.index, inp_yr[col], color=color, lw=1.8, marker='o', ms=3)
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.set_xlabel('Year')
    ax.grid(alpha=.3)
fig.suptitle('Mean Input Trends across counties (each in original units), 1981-2016',
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(OUT_DIR / '4_mean_input_subplots.png')
plt.close(fig)

# --------------- 5. total input (indexed avg) on its own ---------------
base = inp_yr.iloc[0]
inp_idx = inp_yr.divide(base) * 100
total_idx = inp_idx[list(input_cols)].mean(axis=1)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(total_idx.index, total_idx.values, color='black', lw=2.2, marker='s', ms=3.5)
ax.axhline(100, color='grey', lw=0.8, ls='--', alpha=.6)
ax.set_xlabel('Year')
ax.set_ylabel(f'Index (base year {int(total_idx.index[0])} = 100)')
ax.set_title('Mean Total Input across counties\n'
             '(simple avg of 4 input indices, units harmonised via base-year normalisation)',
             fontsize=12)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT_DIR / '5_mean_total_input_trend.png')
plt.close(fig)

# --------------- 6. combined 2x2 panel: GDD / rzsm / Output / Total input ---------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

ax = axes[0, 0]
ax.plot(gdd_yr.index, gdd_yr.values, color='#d1495b', lw=2, marker='o', ms=3)
ax.set_xlabel('Year'); ax.set_ylabel('Mean GDD (growing season)')
ax.set_title('Mean Growing-Season GDD, 1982-2016')
ax.grid(alpha=.3)

ax = axes[0, 1]
ax.plot(rz_yr.index, rz_yr.values, color='#2e86ab', lw=2, marker='o', ms=3)
ax.set_xlabel('Year'); ax.set_ylabel('Mean rzsm_gs')
ax.set_title('Mean Growing-Season Root-Zone Soil Moisture, 1982-2016')
ax.grid(alpha=.3)

ax = axes[1, 0]
ax.plot(out_yr.index, out_yr.values, color='#386641', lw=2, marker='o', ms=3)
ax.set_xlabel('Year'); ax.set_ylabel('Mean GVP_allagr_impute (per county)')
ax.set_title('Mean Total Agricultural Output, 1981-2016')
ax.grid(alpha=.3)

ax = axes[1, 1]
ax.plot(total_idx.index, total_idx.values, color='black', lw=2.2, marker='s', ms=3.5)
ax.axhline(100, color='grey', lw=0.8, ls='--', alpha=.6)
ax.set_xlabel('Year'); ax.set_ylabel(f'Index (base year {int(total_idx.index[0])} = 100)')
ax.set_title('Mean Total Input (avg of 4 indices), 1981-2016')
ax.grid(alpha=.3)

fig.suptitle('Climate and Production Time Trends across Chinese counties',
             fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / '6_combined_trends_2x2.png')
plt.close(fig)

# --------------- Sanity print ---------------
print('Saved 4 figures to', OUT_DIR)
print('GDD years:', gdd_yr.index.min(), '-', gdd_yr.index.max(), 'n=', gdd_yr.notna().sum())
print('rzsm years:', rz_yr.index.min(), '-', rz_yr.index.max(), 'n=', rz_yr.notna().sum())
print('Output years:', out_yr.index.min(), '-', out_yr.index.max())
print('Input base year:', int(inp_idx.index[0]))
print('Input index endpoints:')
print(inp_idx.iloc[[0, -1]].round(1))
