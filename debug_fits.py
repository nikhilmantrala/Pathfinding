#!/usr/bin/env python3
"""
Debug script to check what fits are being selected for actual data
"""
import os
import pandas as pd
import numpy as np
import sys

# Add path
sys.path.insert(0, '/'.join(os.path.abspath(__file__).split('/')[:-1]))

from visualize_data import fit_linear, fit_exponential, fit_power_law, choose_best_fit

# Load data
data_dir = "processed_data"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("Testing fit selection on actual data...\n")

# Test on nodes vs path cost for ML algorithm
df_ml = df[df['Algorithm'].str.lower().str.contains('ml', na=False) & 
           ~df['Algorithm'].str.lower().str.contains('dynamic', na=False)].copy()

if len(df_ml) > 0:
    x = df_ml['Path Length'].to_numpy()
    y = df_ml['Nodes Visited'].to_numpy()
    
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    print(f"Testing ML (static) on {len(x)} samples")
    print(f"X (Path Length) range: {x.min():.2f} - {x.max():.2f}")
    print(f"Y (Nodes Visited) range: {y.min():.2f} - {y.max():.2f}\n")
    
    # Try each fit
    print("=" * 60)
    print("FIT COMPARISON")
    print("=" * 60)
    
    coef_lin, r2_lin, _ = fit_linear(x, y)
    print(f"\nLINEAR: y = {coef_lin[0]:.4f}*x + {coef_lin[1]:.4f}")
    print(f"  R² = {r2_lin:.6f}")
    
    if np.all(y > 0):
        coef_exp, r2_exp, _ = fit_exponential(x, y)
        a, b = coef_exp
        print(f"\nEXPONENTIAL: y = {a:.4f} * exp({b:.4f}*x)")
        print(f"  R² = {r2_exp:.6f}")
        print(f"  Better than linear by: {(r2_exp - r2_lin)*100:.2f}%")
    
    if np.all(x > 0) and np.all(y > 0):
        coef_pow, r2_pow, _ = fit_power_law(x, y)
        a, b = coef_pow
        print(f"\nPOWER LAW: y = {a:.4f} * x^{b:.4f}")
        print(f"  R² = {r2_pow:.6f}")
        print(f"  Better than linear by: {(r2_pow - r2_lin)*100:.2f}%")
    
    print("\n" + "=" * 60)
    best_fit = choose_best_fit(x, y, prefer_nonlinear=True)
    fit_type, fit_params, r2_best = best_fit
    print(f"SELECTED FIT: {str(fit_type).upper()}")
    print(f"  R² = {r2_best:.6f}")
    print("=" * 60)

# Test on A* for comparison
print("\n\nTesting A* (static) on nodes vs path cost...\n")
df_astar = df[df['Algorithm'].str.lower().str.contains('astar|a\\*|a star', na=False, regex=True) & 
              ~df['Algorithm'].str.lower().str.contains('dynamic', na=False)].copy()

if len(df_astar) > 0:
    x = df_astar['Path Length'].to_numpy()
    y = df_astar['Nodes Visited'].to_numpy()
    
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    print(f"Testing A* (static) on {len(x)} samples")
    print(f"X (Path Length) range: {x.min():.2f} - {x.max():.2f}")
    print(f"Y (Nodes Visited) range: {y.min():.2f} - {y.max():.2f}\n")
    
    # Try each fit
    print("=" * 60)
    print("FIT COMPARISON")
    print("=" * 60)
    
    coef_lin, r2_lin, _ = fit_linear(x, y)
    print(f"\nLINEAR: y = {coef_lin[0]:.4f}*x + {coef_lin[1]:.4f}")
    print(f"  R² = {r2_lin:.6f}")
    
    if np.all(y > 0):
        coef_exp, r2_exp, _ = fit_exponential(x, y)
        a, b = coef_exp
        print(f"\nEXPONENTIAL: y = {a:.4f} * exp({b:.4f}*x)")
        print(f"  R² = {r2_exp:.6f}")
        print(f"  Better than linear by: {(r2_exp - r2_lin)*100:.2f}%")
    
    if np.all(x > 0) and np.all(y > 0):
        coef_pow, r2_pow, _ = fit_power_law(x, y)
        a, b = coef_pow
        print(f"\nPOWER LAW: y = {a:.4f} * x^{b:.4f}")
        print(f"  R² = {r2_pow:.6f}")
        print(f"  Better than linear by: {(r2_pow - r2_lin)*100:.2f}%")
    
    print("\n" + "=" * 60)
    best_fit = choose_best_fit(x, y, prefer_nonlinear=True)
    fit_type, fit_params, r2_best = best_fit
    print(f"SELECTED FIT: {str(fit_type).upper()}")
    print(f"  R² = {r2_best:.6f}")
    print("=" * 60)
