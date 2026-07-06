"""
DEA-based TFP Decomposition for China Agricultural Production
Replication of Chambers, Pieralli & Sheng (2020, AJAE)

Model (Option C):
  q = 1  output:  real GVP (deflated by PPP_2005)
  p = 3  conventional inputs: Labor, Land, Capital+Intermediate
  s = 2  weakly disposable: GDD, RZSM

Decomposition:
  Dln TFP(t,t-1) = DeltaT + DeltaW + DeltaX + DeltaE
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from itertools import product
import time
import sys
import os

# ============================================================
# DEA LP solver
# ============================================================

def dea_output(Y_ref, X_conv_ref, X_weak_ref, x_conv, x_weak, p, s):
    """
    Output-oriented DEA with weakly disposable inputs under NIRS.

    Solves: max lambda' @ Y_ref  s.t.
      X_conv_ref @ lambda <= x_conv   (free disposal)
      X_weak_ref @ lambda  = x_weak   (weak disposability)
      sum(lambda) <= 1                 (NIRS)
      lambda >= 0

    Returns frontier output value, or NaN if infeasible.
    """
    q, n = Y_ref.shape

    # Objective: min -Y_ref @ lambda
    c = -Y_ref.flatten()  # (n,)

    # Inequality: X_conv_ref @ lam <= x_conv; sum(lam) <= 1
    A_ub = np.vstack([X_conv_ref, np.ones((1, n))])  # (p+1, n)
    b_ub = np.append(x_conv, 1.0)                     # (p+1,)

    # Equality: X_weak_ref @ lam = x_weak
    A_eq = X_weak_ref  # (s, n)
    b_eq = x_weak      # (s,)

    bounds = [(0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs',
                  options={'maxiter': 50000, 'presolve': True,
                           'dual_feasibility_tolerance': 1e-9,
                           'primal_feasibility_tolerance': 1e-9})

    if res.success:
        return -res.fun  # frontier output = lambda' @ Y_ref
    else:
        return np.nan


# ============================================================
# Decomposition engine
# ============================================================

def compute_decomposition_year(Y_all, X_conv_all, X_weak_all, N, t_idx, p, s):
    """
    Compute DEA decomposition for year t_idx vs t_idx-1.

    Y_all, X_conv_all, X_weak_all are (var x N*T) matrices.
    t_idx is 1-indexed (first decomposition at t_idx=2).

    Returns: TECH, WEATHER, INPUT, EFF, TFP for N counties.
    """
    K = p + s
    n_combos = 2**K

    # Cumulative technologies
    N00 = (t_idx - 1) * N  # T0: years 1..t-1
    N01 = t_idx * N         # T1: years 1..t

    Y0 = Y_all[:, :N00]
    X0_conv = X_conv_all[:, :N00]
    X0_weak = X_weak_all[:, :N00]

    Y1 = Y_all[:, :N01]
    X1_conv = X_conv_all[:, :N01]
    X1_weak = X_weak_all[:, :N01]

    # Evaluation units
    x0_conv = X_conv_all[:, (t_idx-2)*N:(t_idx-1)*N]  # period 0
    x1_conv = X_conv_all[:, (t_idx-1)*N:t_idx*N]      # period 1
    x0_weak = X_weak_all[:, (t_idx-2)*N:(t_idx-1)*N]
    x1_weak = X_weak_all[:, (t_idx-1)*N:t_idx*N]
    y0 = Y_all[:, (t_idx-2)*N:(t_idx-1)*N]
    y1 = Y_all[:, (t_idx-1)*N:t_idx*N]

    # All bit patterns
    bits_all = np.array(list(product([0, 1], repeat=K)))  # (2^K, K)

    # Result matrices
    TECH = np.full(N, np.nan)
    WEATHER = np.full(N, np.nan)
    INPUT = np.full(N, np.nan)
    EFF = np.full(N, np.nan)
    TFP = np.full(N, np.nan)

    for i in range(N):
        xc0 = x0_conv[:, i]
        xc1 = x1_conv[:, i]
        xw0 = x0_weak[:, i]
        xw1 = x1_weak[:, i]

        # Evaluate all combos on T1 and T0
        fvals_T1 = np.zeros(n_combos)
        fvals_T0 = np.zeros(n_combos)
        valid = True

        for c_idx in range(n_combos):
            b = bits_all[c_idx]
            xc = np.where(b[:p] == 0, xc0, xc1)
            xw = np.where(b[p:] == 0, xw0, xw1)

            fvals_T1[c_idx] = dea_output(Y1, X1_conv, X1_weak, xc, xw, p, s)
            fvals_T0[c_idx] = dea_output(Y0, X0_conv, X0_weak, xc, xw, p, s)

            if np.isnan(fvals_T1[c_idx]) or np.isnan(fvals_T0[c_idx]):
                valid = False
                break
            if fvals_T1[c_idx] <= 0 or fvals_T0[c_idx] <= 0:
                valid = False
                break

        if not valid:
            continue

        # --- Technical change (paper Eq. 8) ---
        # ΔT = 0.5*[ln f_T1(x0,w0) - ln f_T0(x0,w0) + ln f_T1(x1,w1) - ln f_T0(x1,w1)]
        # Evaluated at ALL period 0 and ALL period 1 endpoints
        all0_idx = 0               # (0,0,0,0,0) → all period 0
        all1_idx = n_combos - 1   # (1,1,1,1,1) → all period 1

        TECH[i] = 0.5 * (np.log(fvals_T1[all0_idx]) - np.log(fvals_T0[all0_idx]) +
                          np.log(fvals_T1[all1_idx]) - np.log(fvals_T0[all1_idx]))

        # --- Weather contribution (Bennet average for each weather variable) ---
        # ΔW_k = (1/2^K) * Σ_c [sgn_k(c) * (ln f_T1(c) + ln f_T0(c))]
        # where sgn_k(c) = +1 if bit_k=1 (period 1 weather), -1 if bit_k=0
        # This measures the average frontier response to changing weather k from w0 to w1
        weather_sum = 0.0
        for k in range(p, K):
            bennet_k = 0.0
            for c_idx in range(n_combos):
                sgn = 2.0 * bits_all[c_idx, k] - 1.0  # +1 if bit=1, -1 if bit=0
                bennet_k += sgn * (np.log(fvals_T1[c_idx]) + np.log(fvals_T0[c_idx]))
            weather_sum += bennet_k / (2 * n_combos)
        WEATHER[i] = weather_sum

        # --- Efficiency change ---
        eff_t = y1[0, i] / fvals_T1[all1_idx]   # E_t = y_t / f_T1(x1,w1)
        eff_t1 = y0[0, i] / fvals_T0[all0_idx]  # E_{t-1} = y_0 / f_T0(x0,w0)

        if eff_t > 0 and eff_t1 > 0:
            EFF[i] = np.log(eff_t) - np.log(eff_t1)
        else:
            continue

        # --- Input/scale as residual (ensures exact identity) ---
        # TFP = ΔT + ΔW + ΔX + ΔE  →  ΔX = TFP - ΔT - ΔW - ΔE

        # --- TFP change ---
        TFP[i] = (np.log(y1[0, i]) - np.log(y0[0, i]) -
                  np.sum(np.log(xc1) - np.log(xc0)))

        # Input/scale = residual
        INPUT[i] = TFP[i] - TECH[i] - WEATHER[i] - EFF[i]

    return TECH, WEATHER, INPUT, EFF, TFP


# ============================================================
# Main execution
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("DEA TFP Decomposition for China (Option C)")
    print("p=3 (Labor, Land, Cap+Inter), s=2 (GDD, RZSM)")
    print("="*60)

    p, q, s = 3, 1, 2
    K = p + s

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'dea_panel_data.csv')
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df)}")

    unique_years = sorted(df['year'].unique())
    unique_counties = sorted(df['countyid'].unique())
    T = len(unique_years)
    N = len(unique_counties)
    print(f"  Panel: N={N} counties, T={T} years ({unique_years[0]}-{unique_years[-1]})")

    # Sort and reshape
    df = df.sort_values(['year', 'countyid']).reset_index(drop=True)

    X_conv_all = df[['laborday_impute', 'land_serv_q', 'capital_inter']].values.T  # (3, N*T)
    Y_all = df[['real_gvp']].values.T  # (1, N*T)
    X_weak_all = df[['gdd_growing_season', 'rzsm_gs']].values.T  # (2, N*T)

    # --- TEST MODE: small subset ---
    test_mode = '--full' not in sys.argv
    if test_mode:
        N_test = 50
        T_test = 4
        print(f"\n*** TEST MODE: first {N_test} counties, {T_test} years ***")
        print("  (Run with --full for complete computation)")

        test_counties = unique_counties[:N_test]
        mask = df['countyid'].isin(test_counties) & df['year'].isin(unique_years[:T_test])
        df_sub = df[mask].sort_values(['year', 'countyid']).reset_index(drop=True)

        X_conv_all = df_sub[['laborday_impute', 'land_serv_q', 'capital_inter']].values.T
        Y_all = df_sub[['real_gvp']].values.T
        X_weak_all = df_sub[['gdd_growing_season', 'rzsm_gs']].values.T
        N = N_test
        T = T_test
        unique_years = unique_years[:T_test]
        unique_counties = test_counties

    n_periods = T - 1
    print(f"\n  Decomposition periods: {n_periods}")
    print(f"  LPs per year: {N} counties x {2 * 2**K + 2} evaluations = {N * (2 * 2**K + 2)}")

    # Storage
    all_TECH = np.full((N, n_periods), np.nan)
    all_WEATHER = np.full((N, n_periods), np.nan)
    all_INPUT = np.full((N, n_periods), np.nan)
    all_EFF = np.full((N, n_periods), np.nan)
    all_TFP = np.full((N, n_periods), np.nan)

    total_start = time.time()

    for t_idx in range(2, T + 1):
        yr = unique_years[t_idx - 1]
        print(f"\n  Year {yr} (period {t_idx-1}/{n_periods})...", end='', flush=True)
        t0 = time.time()

        TECH, WEATHER, INPUT, EFF, TFP = compute_decomposition_year(
            Y_all, X_conv_all, X_weak_all, N, t_idx, p, s)

        elapsed = time.time() - t0
        n_valid = np.sum(~np.isnan(TECH))
        print(f" {elapsed:.1f}s ({n_valid}/{N} valid)")

        all_TECH[:, t_idx-2] = TECH
        all_WEATHER[:, t_idx-2] = WEATHER
        all_INPUT[:, t_idx-2] = INPUT
        all_EFF[:, t_idx-2] = EFF
        all_TFP[:, t_idx-2] = TFP

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if test_mode:
        per_county_sec = total_elapsed / (N * n_periods)
        est_full = per_county_sec * 1319 * 33 / 3600
        print(f"Estimated full run (1319 counties x 33 years): {est_full:.1f} hours")

    # --- Identity check ---
    residual = all_TFP - all_TECH - all_WEATHER - all_INPUT - all_EFF
    valid_mask = ~np.isnan(residual)
    if np.any(valid_mask):
        max_res = np.max(np.abs(residual[valid_mask]))
        mean_res = np.mean(np.abs(residual[valid_mask]))
        print(f"\nDecomposition identity check:")
        print(f"  Max |residual|: {max_res:.2e}")
        print(f"  Mean |residual|: {mean_res:.2e}")
        print(f"  Identity {'HOLDS' if max_res < 1e-6 else 'VIOLATED (check formulas)'}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary statistics:")
    print(f"{'Component':<15} {'Mean':>10} {'Std':>10}")
    for name, arr in [('TFP', all_TFP), ('Tech (DT)', all_TECH),
                      ('Weather (DW)', all_WEATHER), ('Input (DX)', all_INPUT),
                      ('Effic (DE)', all_EFF)]:
        v = arr[~np.isnan(arr)]
        if len(v) > 0:
            print(f"{name:<15} {np.mean(v):>10.5f} {np.std(v):>10.5f}")
        else:
            print(f"{name:<15} {'N/A':>10} {'N/A':>10}")

    # Save results
    if not test_mode:
        out_path = os.path.join(os.path.dirname(__file__), 'dea_results_china.npz')
        np.savez(out_path,
                 TECH=all_TECH, WEATHER=all_WEATHER, INPUT=all_INPUT,
                 EFF=all_EFF, TFP=all_TFP,
                 unique_years=np.array(unique_years),
                 unique_counties=np.array(unique_counties))
        print(f"\nResults saved to {out_path}")
