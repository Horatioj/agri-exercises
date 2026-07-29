"""
Time trends per 9 major Chinese agricultural macro-regions.
Two figures:
  A) 4-input subplots (Labor, Land, Capital, Intermediate)
  B) 4 combined trends (GDD, rzsm_gs, Total Output, Total Input index)

Each subplot shows:
  * 9 zone lines (one per macro-region)
  * National mean (thick black, on top)
Legend is placed at the bottom of the figure.

The 37 cropping-system zones (zone37_main) are aggregated into the standard
9 macro-regions used in Chinese agro-geography textbooks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

CLIM_CSV = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/climate_panel_1982_2016.csv'
PROD_DTA = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/cty_prod_account_agg_forAXN.dta'
OUT_DIR  = Path('/Users/zhangxinzhen/Desktop/TFP/Code/output/trends')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 37 zones -> 9 macro-regions ----------
MACRO = [
    ('Northeast',                  '#1f77b4', [3, 4, 25, 30]),
    ('Inner Mongolia & Great Wall','#aec7e8', [9, 17]),
    ('Huang-Huai-Hai (N China Plain)', '#ff7f0e', [12, 13, 19, 27, 35]),
    ('Loess Plateau',              '#d62728', [8, 14, 15, 18, 26, 28]),
    ('Middle-Lower Yangtze',       '#2ca02c', [7, 16, 33, 34]),
    ('Southwest',                  '#9467bd', [2, 5, 6, 10, 22, 23, 24, 29]),
    ('South China',                '#8c564b', [21, 32, 37]),
    ('Gansu-Xinjiang (NW arid)',   '#e377c2', [20, 31, 36]),
    ('Qinghai-Tibet',              '#7f7f7f', [1, 11]),
]
ZONE2MACRO = {z: name for name, _, zs in MACRO for z in zs}
assert sum(len(zs) for _, _, zs in MACRO) == 37

mpl.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ---------- load ----------
clim = pd.read_csv(CLIM_CSV)
prod = pd.read_stata(PROD_DTA)

# Attach macro region
clim = clim.dropna(subset=['zone37_main']).copy()
clim['macro'] = clim['zone37_main'].astype(int).map(ZONE2MACRO)
clim = clim[clim['valid_climate_sample']]

# county -> zone -> macro mapping for production
cty2zone = (clim.groupby('PAC')['zone37_main']
                .agg(lambda s: int(s.mode().iat[0]))
                .to_dict())
prod = prod.copy()
prod['zone37_main'] = prod['countyid'].map(cty2zone)
prod = prod.dropna(subset=['zone37_main'])
prod['macro'] = prod['zone37_main'].astype(int).map(ZONE2MACRO)


# ---------- aggregation helpers ----------
def macro_yearly_mean(df, var, weighted_by=None):
    """Return DataFrame [year x macro] of mean `var`, optionally cropland-weighted."""
    s = df[df[var].notna()].copy()
    if weighted_by is not None:
        s = s[s[weighted_by].notna() & (s[weighted_by] > 0)]
        s['_w'] = s[weighted_by]
        s['_wx'] = s['_w'] * s[var]
        g = (s.groupby(['year', 'macro'])
               .apply(lambda d: d['_wx'].sum() / d['_w'].sum(), include_groups=False))
    else:
        # replace 0 with NaN before mean so empty inputs do not drag the mean
        s.loc[s[var] == 0, var] = np.nan
        s = s.dropna(subset=[var])
        g = s.groupby(['year', 'macro'])[var].mean()
    return g.unstack('macro')


def national_yearly_mean(df, var, weighted_by=None):
    s = df[df[var].notna()].copy()
    if weighted_by is not None:
        s = s[s[weighted_by].notna() & (s[weighted_by] > 0)]
        return (s.groupby('year')
                  .apply(lambda d: np.average(d[var], weights=d[weighted_by]),
                         include_groups=False))
    s.loc[s[var] == 0, var] = np.nan
    return s.dropna(subset=[var]).groupby('year')[var].mean()


# ---------- shared legend handles ----------
HANDLES = [plt.Line2D([0], [0], color=col, lw=2.0, label=name)
           for name, col, _ in MACRO]
HANDLES.append(plt.Line2D([0], [0], color='black', lw=2.6, label='National mean'))


def draw_panel(ax, macro_df, national_s, title, ylabel):
    for name, col, _ in MACRO:
        if name in macro_df.columns:
            ax.plot(macro_df.index, macro_df[name], color=col, lw=1.2, alpha=0.85)
    ax.plot(national_s.index, national_s.values, color='black', lw=2.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Year')
    ax.grid(alpha=.3)


# ================================================================
# Figure A: 4-input subplots
# ================================================================
input_cols = {
    'Laborday_impute': 'Labor (days)',
    'Land_serv_q':     'Land services (q)',
    'capital_serv_q':  'Capital services (q)',
    'Inter_all_real':  'Intermediate (real)',
}

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
for ax, (col, label) in zip(axes.flat, input_cols.items()):
    macro_df    = macro_yearly_mean(prod, col)
    national_s  = national_yearly_mean(prod, col)
    draw_panel(ax, macro_df, national_s, label, label)

fig.legend(handles=HANDLES, loc='lower center', ncol=5, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('Mean Input Trends across counties — 9 macro-regions (each in original units), 1981-2016',
             fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
fig.savefig(OUT_DIR / '7_mean_input_subplots_by_zone.png')
plt.close(fig)
print('saved 7_mean_input_subplots_by_zone.png')

# ================================================================
# Figure B: combined 2x2 (GDD / rzsm / Output / Total Input)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

# GDD (cropland-weighted)
ax = axes[0, 0]
m_df = macro_yearly_mean(clim, 'gdd_growing_season', weighted_by='cropland_area_km2')
n_s  = national_yearly_mean(clim, 'gdd_growing_season', weighted_by='cropland_area_km2')
draw_panel(ax, m_df, n_s, 'Mean Growing-Season GDD, 1982-2016', 'Mean GDD (growing season)')

# rzsm (cropland-weighted)
ax = axes[0, 1]
m_df = macro_yearly_mean(clim, 'rzsm_gs', weighted_by='cropland_area_km2')
n_s  = national_yearly_mean(clim, 'rzsm_gs', weighted_by='cropland_area_km2')
draw_panel(ax, m_df, n_s, 'Mean Growing-Season Root-Zone Soil Moisture, 1982-2016', 'Mean rzsm_gs')

# Output (unweighted mean per county)
ax = axes[1, 0]
m_df = macro_yearly_mean(prod, 'GVP_allagr_impute')
n_s  = national_yearly_mean(prod, 'GVP_allagr_impute')
draw_panel(ax, m_df, n_s, 'Mean Total Agricultural Output, 1981-2016', 'Mean GVP per county')

# Total Input index: per-macro indexed to base year (=1981) of that macro
ax = axes[1, 1]
input_list = list(input_cols.keys())

def index_total_input(df):
    """Per (year, macro) mean of 4 input vars indexed to first year per macro = 100,
    then averaged across the 4 indices."""
    out = {}
    for col in input_list:
        m = macro_yearly_mean(df, col)               # year x macro
        base = m.iloc[0]
        out[col] = m.divide(base) * 100
    # average the 4 indices
    idx = sum(out.values()) / len(out)
    return idx

macro_idx = index_total_input(prod)
# National mean total input index (same construction at country level)
def index_national(df):
    parts = []
    for col in input_list:
        n = national_yearly_mean(df, col)
        parts.append(n / n.iloc[0] * 100)
    return sum(parts) / len(parts)

national_idx = index_national(prod)
draw_panel(ax, macro_idx, national_idx,
           'Mean Total Input (avg of 4 indices, base=first year=100), 1981-2016',
           'Index (base year=100)')
ax.axhline(100, color='grey', lw=0.7, ls='--', alpha=.5)

fig.legend(handles=HANDLES, loc='lower center', ncol=5, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('Climate and Production Time Trends across Chinese counties — 9 macro-regions',
             fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
fig.savefig(OUT_DIR / '8_combined_trends_by_zone.png')
plt.close(fig)
print('saved 8_combined_trends_by_zone.png')
