"""
3D production surface per agricultural zone.
  x = log10 total input
  y = gdd_gs (root-zone soil moisture, growing season)
  z = log10 output (surface height)

For each zone, fit  log10_Y = b0 + b1*log_X + b2*rzsm + b3*log_X*rzsm + b4*log_X^2 + b5*rzsm^2
on the county-year sample, then evaluate on a regular (log_X, rzsm) grid and plot a
smooth coloured surface. Scatter of raw points is overlaid faintly for sanity.

Outputs:
  output/3d_zone_surface/zone_XX.png         (37 individual figures)
  output/3d_zone_surface/_overview_grid.png  (6x7 grid overview)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from pathlib import Path

CLIM = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/climate_panel_1982_2016.csv'
PROD = '/Users/zhangxinzhen/Desktop/TFP/TFP Data/cty_prod_account_agg_forAXN.dta'
OUT  = Path('/Users/zhangxinzhen/Desktop/TFP/Code/output/3d_zone_surface_gdd')
OUT.mkdir(parents=True, exist_ok=True)

# zone37_main -> (short English label, long English label)
# Source: TFP/Code/intermediate/growing_season/zone37_growing_season.csv (Chinese names)
ZONE_NAMES = {
    1:  ('SE Tibet / W Sichuan Valleys',         'SE Tibet & W Sichuan Valleys — cool-loving crops, single-cropping'),
    2:  ('Sichuan-Hubei-Hunan-Guizhou Plateau',  'Sichuan-Hubei-Hunan-Guizhou Low Plateau & Mts — paddy/dryland double or single'),
    3:  ('Greater & Lesser Hinggan Piedmont',    'Greater/Lesser Hinggan Range piedmont — cool-temperate crops, single-cropping'),
    4:  ('Sanjiang Plain & Changbai Mts',        'Sanjiang Plain & Changbai Mts — temperate-cool crops, single-cropping'),
    5:  ('S Yunnan Mountains',                   'S Yunnan Mts — dryland/paddy double or triple-cropping'),
    6:  ('Yunnan-Guizhou Border Plateau',        'Yunnan-Guizhou border plateau-mt-valleys — dryland 1-2 + paddy 2 cropping'),
    7:  ('Hubei-Henan-Anhui Hills & Plains',     'Hubei-Henan-Anhui hills & plains — paddy/dryland double + early-triple'),
    8:  ('Fen-Wei Valley',                       'Fen-Wei Valley — irrigated double + dryland 1-2 cropping'),
    9:  ('Houshan-Bashang-N Shanxi Plateau',     'Houshan-Bashang & N Shanxi plateau-mts — semi-arid cool, single-cropping'),
    10: ('Guizhou Plateau',                      'Guizhou Plateau — paddy/dryland double or single-cropping'),
    11: ('Haibei-Gannan Plateau',                'Haibei-Gannan Plateau — cool-loving crops, single-cropping with fallow'),
    12: ('Heilonggang Low Plain',                'Heilonggang water-scarce low plain — irrigated double + dryland single'),
    13: ('Huang-Huai Plain & Nanyang Basin',     'Huang-Huai Plain & Nanyang Basin — dryland/irrigated double-cropping'),
    14: ('E Loess Plateau',                      'E Loess Plateau — drought-prone warm-loving, single-cropping'),
    15: ('E Shanxi Sub-humid',                   'E Shanxi sub-humid drought-prone — single-cropping with catch crop'),
    16: ('Liang-Hu (Hubei-Hunan) Plain',         'Liang-Hu Plain & hills — paddy triple/double cropping'),
    17: ('Liao-Jilin-SE Inner Mongolia',         'W Liao-Jilin / SE Inner Mongolia / N Hebei — semi-arid warm, single-cropping'),
    18: ('C Gansu-E Qinghai Loess Hills',        'C Gansu / E Qinghai / SC Ningxia Loess hills — semi-arid cool, single-cropping'),
    19: ('NW Shandong-N Henan Low Plain',        'NW Shandong & N Henan low plain — irrigated grain double + cotton single'),
    20: ('S & E Xinjiang Oases',                 'S & E Xinjiang oases — double or single-cropping'),
    21: ('Nanling Hills',                        'Nanling hills & mountains — paddy/dryland triple-double cropping'),
    22: ('E Sichuan Basin Hills',                'E Sichuan Basin hills & low mts — paddy/dryland double-triple cropping'),
    23: ('W Sichuan Basin Plain (Chengdu)',      'W Sichuan Basin plain — wheat-rice double + catch crop'),
    24: ('Qinba Mountains',                      'Qinba Mts — dryland double-single + paddy double cropping'),
    25: ('Songnen Plain',                        'Songnen Plain — warm-loving crops, single-cropping'),
    26: ('N Wei-E Gansu Loess',                  'N Wei valley & E Gansu sub-humid drought-prone — winter wheat single + catch'),
    27: ('Yan-Taihang Piedmont',                 'Yan-Taihang piedmont plain — irrigated relay-double + dryland single'),
    28: ('W Henan Hills',                        'W Henan hills & mts — dryland slope single + irrigated double'),
    29: ('Yunnan Plateau',                       'Yunnan Plateau — paddy/dryland double-single cropping'),
    30: ('Liaohe Plain',                         'Liaohe Plain & hills — warm-loving crops, single-cropping with catch'),
    31: ('Hetao & Hexi Corridor',                'Hetao & Hexi Corridor irrigated — single-cropping with catch crop'),
    32: ('Zhejiang-Fujian Hills',                'Zhejiang-Fujian hills & mts — paddy/dryland triple-double cropping'),
    33: ('Lower Yangtze Plain',                  'Lower Yangtze plain & hills — paddy early-triple/double cropping'),
    34: ('Jiang-Huai Plain',                     'Jiang-Huai Plain — wheat-rice double + dryland triple'),
    35: ('Shandong Hills',                       'Shandong hills — irrigated double + dryland peanut/cotton single'),
    36: ('N Xinjiang Oasis',                     'N Xinjiang irrigated — single-cropping with catch crop'),
    37: ('S China Coastal Plain',                'S China low coastal plain — late triple-cropping'),
}

# Macro-groups by climate × cropping system. Order = display order in overview grid.
GROUPS = [
    ('A. NE cold, single-cropping',                 '#4575b4', [3, 4, 25, 30]),
    ('B. N China semi-arid, single-cropping',       '#74add1', [9, 11, 17, 18]),
    ('C. NW arid oasis, irrigated single',          '#abd9e9', [20, 31, 36]),
    ('D. Loess Plateau / Fen-Wei, drought-prone',   '#fee090', [8, 14, 15, 26, 28]),
    ('E. N China Plain, irrigated double-cropping', '#f46d43', [12, 13, 19, 27, 35]),
    ('F. Yangtze & Sichuan, paddy multi-cropping',  '#a6d96a', [7, 16, 22, 23, 24, 33, 34]),
    ('G. SW Plateau & Tibet, cool single/double',   '#66bd63', [1, 2, 5, 6, 10, 29]),
    ('H. S China subtropical, triple-cropping',     '#1a9850', [21, 32, 37]),
]
# build zone -> (group_label, group_color) lookup
ZONE_GROUP = {z: (lbl, col) for lbl, col, zs in GROUPS for z in zs}
ZONE_ORDER = [z for _, _, zs in GROUPS for z in zs]  # display order
assert len(ZONE_ORDER) == 37 and len(set(ZONE_ORDER)) == 37

# ---------- load + merge ----------
clim = pd.read_csv(CLIM)
prod = pd.read_stata(PROD)
df = clim.merge(prod, left_on=['PAC', 'year'], right_on=['countyid', 'year'], how='inner')
df = df.rename(columns={'gdd_growing_season': 'gdd_gs'})

inputs = ['Laborday_impute', 'Land_serv_q', 'capital_serv_q', 'Inter_all_real']
mask = (df[inputs] > 0).all(axis=1) & (df['GVP_allagr_impute'] > 0) \
       & df['gdd_gs'].notna() & df['zone37_main'].notna() & df['valid_climate_sample']
df = df.loc[mask, inputs + ['GVP_allagr_impute', 'gdd_gs', 'zone37_main', 'year']].copy()

df['log_input']  = np.log10(np.exp(np.log(df[inputs]).mean(axis=1)))
df['log_output'] = np.log10(df['GVP_allagr_impute'])

GRID = 32  # grid resolution per axis


def fit_and_grid(sub):
    """Return (Xg, Yg, Zg) — fitted log-output surface over a (log_X, rzsm) grid."""
    x = sub['log_input'].values
    y = sub['gdd_gs'].values
    z = sub['log_output'].values

    # design matrix for full quadratic with interaction
    X = np.column_stack([np.ones_like(x), x, y, x * y, x ** 2, y ** 2])
    # OLS
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)

    # use 5%-95% range to avoid extrapolation artefacts at the edges
    xlo, xhi = np.quantile(x, [.025, .975])
    ylo, yhi = np.quantile(y, [.025, .975])
    xg = np.linspace(xlo, xhi, GRID)
    yg = np.linspace(ylo, yhi, GRID)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = (beta[0]
          + beta[1] * Xg
          + beta[2] * Yg
          + beta[3] * Xg * Yg
          + beta[4] * Xg ** 2
          + beta[5] * Yg ** 2)
    # R^2 for diagnostic
    z_hat = X @ beta
    r2 = 1 - np.var(z - z_hat) / np.var(z)
    return Xg, Yg, Zg, r2, beta


def draw(ax, sub, title, *, label=True, show_pts=True, cmap=cm.plasma):
    Xg, Yg, Zg, r2, beta = fit_and_grid(sub)

    x_min, x_max = Xg.min(), Xg.max()
    y_min, y_max = Yg.min(), Yg.max()
    z_min, z_max = Zg.min(), Zg.max()
    pad = 0.15
    x_lo = x_min - (x_max - x_min) * pad
    y_hi = y_max + (y_max - y_min) * pad
    z_lo = z_min - (z_max - z_min) * pad
    ax.set_xlim(x_lo, x_max)
    ax.set_ylim(y_min, y_hi)
    ax.set_zlim(z_lo, z_max)

    xg = np.linspace(x_min, x_max, 80)
    yg = np.linspace(y_min, y_max, 80)

    def quad_fit(a, b):
        A_ = np.column_stack([np.ones_like(a), a, a ** 2])
        c, *_ = np.linalg.lstsq(A_, b, rcond=None)
        return c

    xx = sub['log_input'].values
    yy = sub['gdd_gs'].values
    zz = sub['log_output'].values

    # Three input-output slices of the surface at p10, p50, p90 of GDD.
    # Parallel lines -> no moderation. Fanning lines -> climate moderates the
    # input elasticity (steeper at higher GDD = climate amplifies input).
    slice_specs = [
        (np.percentile(yy, 10), '#fdcc8a', 'low GDD (p10)'),
        (np.percentile(yy, 50), '#fc8d59', 'mid GDD (p50)'),
        (np.percentile(yy, 90), '#b30000', 'high GDD (p90)'),
    ]
    for y_s, col, _ in slice_specs:
        z_s = (beta[0] + beta[1] * xg + beta[2] * y_s
               + beta[3] * xg * y_s + beta[4] * xg ** 2 + beta[5] * y_s ** 2)
        ax.plot(xg, np.full_like(xg, y_s), z_s, color=col, lw=2.2, zorder=0)

    # Left wall (yz at x=x_lo): Output ~ GDD, fit on projected (y, z) data
    c = quad_fit(yy, zz)
    z_y = c[0] + c[1] * yg + c[2] * yg ** 2
    ax.plot(np.full_like(yg, x_lo), yg, z_y, color='#1f77b4', lw=2.0, zorder=0)

    # Bottom (xy at z=z_lo): Input ~ GDD   (causal direction: GDD -> input)
    # Treat GDD as the regressor so the curve gives a single input for each
    # temperature level (input is endogenous to climate, not vice versa).
    c = quad_fit(yy, xx)
    x_pred = c[0] + c[1] * yg + c[2] * yg ** 2
    ax.plot(x_pred, yg, np.full_like(yg, z_lo), color='#2ca02c', lw=2.0, zorder=0)

    # surface drawn LAST so it overlays the back-wall curves
    surf = ax.plot_surface(Xg, Yg, Zg, cmap=cmap,
                           edgecolor='none', alpha=0.45, antialiased=True,
                           rstride=1, cstride=1, zorder=10)
    if show_pts:
        s = sub.sample(min(len(sub), 600), random_state=0)
        ax.scatter(s['log_input'], s['gdd_gs'], s['log_output'],
                   s=3, color='black', alpha=0.18, depthshade=False)

    ax.set_title(title, fontsize=9, pad=2)
    if label:
        ax.set_xlabel('Input',  fontsize=7, labelpad=-4)
        ax.set_ylabel('GDD',    fontsize=7, labelpad=-4)
        ax.set_zlabel('Output', fontsize=7, labelpad=-4)
    else:
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=24, azim=-62)
    return surf, r2


data_zones = set(int(z) for z in df['zone37_main'].unique())
zones_display = [z for z in ZONE_ORDER if z in data_zones]  # respect group order
print(f'fitting + plotting {len(zones_display)} zones (grouped order)')

# ---------- 1. individual figures ----------
for zi in zones_display:
    short, long_name = ZONE_NAMES.get(zi, (f'Zone {zi}', f'Zone {zi}'))
    g_label, g_color = ZONE_GROUP[zi]
    sub = df[df['zone37_main'] == zi]
    fig = plt.figure(figsize=(8.2, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    surf, r2 = draw(ax, sub,
                    f'Zone {zi}: {long_name}\n(n = {len(sub)} county-years)',
                    label=True, show_pts=True)
    # group banner above plot
    fig.text(0.5, 0.965, g_label, ha='center', va='center',
             fontsize=10, weight='bold', color='white',
             bbox=dict(facecolor=g_color, edgecolor='none', boxstyle='round,pad=0.4'))
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label('log10 Output (fitted)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / f'zone_{zi:02d}.png', dpi=170)
    plt.close(fig)
    print(f'  zone {zi:2d}: n={len(sub):4d}  R2={r2:.2f}  [{g_label[:1]}]  {short}')

# ---------- 2. one figure PER group ----------
# layout (rows, cols) by number of zones in the group
LAYOUTS = {3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (2, 4)}

# remove old overview if present
old_overview = OUT / '_overview_grid.png'
if old_overview.exists():
    old_overview.unlink()

for gi, (g_label, g_color, g_zones) in enumerate(GROUPS, start=1):
    g_zones_present = [z for z in g_zones if z in data_zones]
    n = len(g_zones_present)
    rows, cols = LAYOUTS[n]
    fig = plt.figure(figsize=(cols * 4.8, rows * 4.6))
    for i, zi in enumerate(g_zones_present):
        short, _ = ZONE_NAMES.get(zi, (f'Zone {zi}', f'Zone {zi}'))
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        sub = df[df['zone37_main'] == zi]
        draw(ax, sub, short, label=True, show_pts=True)

    letter, climate_desc = g_label.split('. ', 1)
    fig.suptitle(climate_desc, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02,
                        wspace=0.0, hspace=0.05)
    out_path = OUT / f'group_{gi}_{letter}.png'
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f'  group {letter}: {n} zones -> {out_path.name}')

print('done, files in', OUT)
