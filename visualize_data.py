# visualize_data_unique_names.py
# Same functionality as your enhanced visualization pipeline but every saved file has a unique name.
# Uses RUN_ID (timestamp) to make filenames unique across runs.
# Requirements: pandas, numpy, matplotlib, seaborn. Optional: scipy for p-values.

import os
import re
import ast
import math
from datetime import datetime
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional stats (pearson / mannwhitneyu)
try:
    from scipy.stats import pearsonr, mannwhitneyu
    SCIPY = True
except Exception:
    SCIPY = False

# --------------------------
# RUN ID (unique per run)
# --------------------------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# --------------------------
# CONFIG
# --------------------------
MIN_PAIRED_SAMPLES = 3

PLOT_CONFIG = {
    'figure_sizes': {
        'bar': (9, 6),
        'box': (9, 6),
        'violin': (9, 6),
        'scatter': (10, 7),
        'line': (10, 6),
        'hist': (9, 6),
        'heatmap': (8, 6)
    },
    'fonts': {
        'title': 14,
        'axis_label': 12,
        'tick_label': 10,
        'legend': 10
    },
    'colors': {
        'astar': '#3498db',
        'ml': '#2ecc71',
        'astar_variable_cost': '#2980b9',
        'ml_variable_cost': '#27ae60'
    },
    'display': {
        'astar': 'A*',
        'ml': 'ML',
        'astar_variable_cost': 'A* (Variable Cost)',
        'ml_variable_cost': 'ML (Variable Cost)'
    }
}

ALGO_KEYMAP = {
    'astar': ['a*', 'astar', 'a_star', 'a star', 'a-star'],
    'astar_variable_cost': ['a* variable cost', 'astar variable cost', 'a* dynamic', 'astar dynamic', 'a*dynamic', 'a_star_dynamic', 'a star dynamic'],
    'ml': ['ml', 'ml heuristic', 'ml_static', 'ml-static', 'ml_static_heuristic'],
    'ml_variable_cost': ['ml variable cost', 'ml-variable cost', 'ml_variable_cost', 'ml variable_cost', 'ml dynamic', 'ml-dynamic', 'ml_dynamic', 'ml_dynamic_heuristic']
}

PAIRWISE_PAIRS = [('astar', 'ml'), ('astar_variable_cost', 'ml_variable_cost')]

# --------------------------
# UTILITIES
# --------------------------
def sanitize_filename(s: Optional[str]) -> str:
    if s is None:
        return "all"
    s = str(s)
    s = s.strip()
    s = re.sub(r'[^A-Za-z0-9\-_\. ]+', '_', s)
    s = s.replace(' ', '_')
    if s == '':
        s = 'unknown'
    return s

def unique_path(dirpath: str, basename: str, ext: str = 'png') -> str:
    """Return a unique file path using RUN_ID; does not create directories."""
    name = sanitize_filename(basename)
    filename = f"{name}_{RUN_ID}.{ext}"
    return os.path.join(dirpath, filename)

def unique_csv_path(dirpath: str, basename: str) -> str:
    return unique_path(dirpath, basename, ext='csv')

def create_output_directory(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, 'visuals')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_processed_data(data_dir: str) -> pd.DataFrame:
    files = []
    for f in os.listdir(data_dir):
        if f.endswith('_processed_full.csv') or (f.endswith('.csv') and not f.startswith('.')):
            files.append(os.path.join(data_dir, f))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def normalize_algorithm_name(s: str) -> str:
    if not isinstance(s, str):
        return str(s).lower()
    s0 = s.strip().lower()
    for key, variants in ALGO_KEYMAP.items():
        for v in variants:
            if v == s0:
                return key
    if 'ml' in s0:
        if 'dyn' in s0 or 'dynamic' in s0 or 'variable' in s0:
            return 'ml_variable_cost'
        return 'ml'
    if 'a*' in s0 or 'astar' in s0 or 'a star' in s0:
        if 'dyn' in s0 or 'dynamic' in s0 or 'variable' in s0:
            return 'astar_variable_cost'
        return 'astar'
    return s0

def ensure_algorithm_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'Algorithm' not in df.columns:
        raise KeyError("Data must include 'Algorithm' column")
    df = df.copy()
    df['__algo_norm'] = df['Algorithm'].apply(normalize_algorithm_name)
    df['__algo_display'] = df['__algo_norm'].apply(lambda k: PLOT_CONFIG['display'].get(k, k))
    return df

def get_metric_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of candidates found: {candidates}. Available columns: {list(df.columns)}")

def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return float('nan'), float('nan')
    x2 = x[mask]; y2 = y[mask]
    if SCIPY:
        try:
            r, p = pearsonr(x2, y2)
            return float(r), float(p)
        except Exception:
            pass
    try:
        r = np.corrcoef(x2, y2)[0, 1]
        return float(r), float('nan')
    except Exception:
        return float('nan'), float('nan')

def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """Linear fit: y = mx + b. Returns (coefficients, R², fit_type)"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.array([0, 0]), float('nan'), 'linear'
    x2, y2 = x[mask], y[mask]
    try:
        coef = np.polyfit(x2, y2, 1)
        y_pred = np.polyval(coef, x2)
        ss_res = np.sum((y2 - y_pred) ** 2)
        ss_tot = np.sum((y2 - np.mean(y2)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
        return coef, float(r2), 'linear'
    except Exception:
        return np.array([0, 0]), float('nan'), 'linear'

def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], float, str]:
    """Exponential fit: y = a * exp(b*x). Returns ((a, b), R², fit_type)"""
    mask = ~(np.isnan(x) | np.isnan(y)) & (y > 0)
    if mask.sum() < 2:
        return (1.0, 0.0), float('nan'), 'exponential'
    x2, y2 = x[mask], y[mask]
    try:
        # Fit log(y) = log(a) + b*x
        # Note: R² on log-transformed data is NOT comparable to original space R²
        log_y = np.log(y2)
        coef = np.polyfit(x2, log_y, 1)
        b, log_a = float(coef[0]), float(coef[1])
        a = np.exp(log_a)
        
        # Calculate R² in ORIGINAL space for proper comparison
        y_pred = a * np.exp(b * x2)
        ss_res = np.sum((y2 - y_pred) ** 2)
        ss_tot = np.sum((y2 - np.mean(y2)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
        
        # Handle case where fit is terrible (negative R²)
        if r2 < -1:
            r2 = float('nan')
        
        return (a, b), float(r2), 'exponential'
    except Exception:
        return (1.0, 0.0), float('nan'), 'exponential'

def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], float, str]:
    """Power law fit: y = a * x^b. Returns ((a, b), R², fit_type)"""
    mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return (1.0, 1.0), float('nan'), 'power'
    x2, y2 = x[mask], y[mask]
    try:
        # Fit log(y) = log(a) + b*log(x)
        # Note: R² on log-transformed data is NOT comparable to original space R²
        log_x = np.log(x2)
        log_y = np.log(y2)
        coef = np.polyfit(log_x, log_y, 1)
        b, log_a = float(coef[0]), float(coef[1])
        a = np.exp(log_a)
        
        # Calculate R² in ORIGINAL space for proper comparison
        y_pred = a * np.power(x2, b)
        ss_res = np.sum((y2 - y_pred) ** 2)
        ss_tot = np.sum((y2 - np.mean(y2)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
        
        # Handle case where fit is terrible (negative R²)
        if r2 < -1:
            r2 = float('nan')
        
        return (a, b), float(r2), 'power'
    except Exception:
        return (1.0, 1.0), float('nan'), 'power'

def choose_best_fit(x: np.ndarray, y: np.ndarray, prefer_nonlinear=True) -> Tuple:
    """Choose between linear, exponential, and power law fits based on R²
    
    Args:
        x, y: Data arrays
        prefer_nonlinear: If True, prefer exponential/power if R² is within 0.02 of linear
    
    Returns:
        (fit_type, fit_params, r2) where fit_type is 'linear'/'exponential'/'power'
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return ('linear', np.array([0, 0]), float('nan'))
    
    results = {}
    
    # Try linear
    coef_lin, r2_lin, _ = fit_linear(x, y)
    results['linear'] = (coef_lin, r2_lin)
    
    # Try exponential (if all y > 0)
    if np.all(y[mask] > 0):
        coef_exp, r2_exp, _ = fit_exponential(x, y)
        results['exponential'] = (coef_exp, r2_exp)
    
    # Try power law (if all x > 0 and y > 0)
    if np.all(x[mask] > 0) and np.all(y[mask] > 0):
        coef_pow, r2_pow, _ = fit_power_law(x, y)
        results['power'] = (coef_pow, r2_pow)
    
    # Select best fit
    if not results:
        return ('linear', np.array([0, 0]), float('nan'))
    
    # Get the best by R²
    best_fit = max(results.items(), key=lambda r: r[1][1] if not np.isnan(r[1][1]) else -np.inf)
    fit_type = best_fit[0]
    coef, r2 = best_fit[1]
    
    # If prefer_nonlinear is True and we have exponential/power options,
    # prefer them if they're within tolerance of best linear fit
    if prefer_nonlinear and fit_type == 'linear' and r2_lin is not None and not np.isnan(r2_lin):
        tolerance = 0.02  # Allow 2% R² loss for better model fit
        
        if 'exponential' in results:
            r2_exp = results['exponential'][1]
            if not np.isnan(r2_exp) and r2_exp >= (r2_lin - tolerance):
                return ('exponential', results['exponential'][0], r2_exp)
        
        if 'power' in results:
            r2_pow = results['power'][1]
            if not np.isnan(r2_pow) and r2_pow >= (r2_lin - tolerance):
                return ('power', results['power'][0], r2_pow)
    
    return (fit_type, coef, r2)

def compute_obstacle_density_from_grid_column(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if 'Obstacle Density' in df2.columns:
        return df2
    if 'grid' not in df2.columns:
        return df2
    densities = []
    for val in df2['grid'].fillna('').astype(str):
        dens = np.nan
        try:
            parsed = ast.literal_eval(val)
            arr = np.array(parsed).flatten()
            numeric = pd.to_numeric(arr, errors='coerce')
            total = numeric.size
            if total == 0:
                dens = np.nan
            else:
                obstacles = np.sum(numeric == 1)
                dens = float(obstacles) / float(total)
        except Exception:
            digits = re.findall(r'[01]', val)
            if digits:
                total = len(digits)
                obstacles = digits.count('1')
                dens = float(obstacles) / float(total)
            else:
                dens = np.nan
        densities.append(dens)
    df2['Obstacle Density'] = densities
    print("Computed 'Obstacle Density' from 'grid' column where possible.")
    return df2

# --------------------------
# Pairing helper
# --------------------------
def make_instance_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'instance_id' in df.columns:
        return df
    df2 = df.copy()
    # check for Test Case or similar first
    for candidate in ['test case', 'test_case', 'testcase', 'test-id', 'test id', 'test']:
        for col in df2.columns:
            if col.strip().lower() == candidate.replace('_', ' ').replace('-', ' '):
                df2['instance_id'] = df2[col].astype(str)
                print(f"Using column '{col}' as instance_id (exact match).")
                return df2
    # fallback heuristics
    present = {col.lower(): col for col in df2.columns}
    seed_terms = ['seed_layout', 'seed', 'seed_id', 'map_seed', 'grid_seed', 'map_id', 'grid_id']
    pair_terms = ['pair_id', 'start_goal_id', 'pair', 'pairid']
    layout_terms = ['layout', 'layout_type', 'map', 'map_name', 'maptype', 'layouttype']
    coord_candidates = ['start_r', 'start_c', 'start_row', 'start_col', 'start_x', 'start_y', 'sx', 'sy',
                        'goal_r', 'goal_c', 'goal_row', 'goal_col', 'goal_x', 'goal_y', 'gx', 'gy', 'end_r', 'end_c']
    selected = []
    for t in seed_terms:
        if t in present:
            selected.append(present[t]); break
    for t in pair_terms:
        if t in present:
            selected.append(present[t]); break
    for t in layout_terms:
        if t in present and present[t] not in selected:
            selected.append(present[t]); break
    coords = [present[c] for c in coord_candidates if c in present]
    if coords:
        selected += coords
    if selected:
        df2['instance_id'] = df2[selected].astype(str).agg('_'.join, axis=1)
        print(f"Built instance_id from columns: {selected[:6]}")
        return df2
    print("Warning: could not build paired 'instance_id' automatically.")
    print("Columns present:", df2.columns.tolist())
    print("First 3 rows:")
    print(df2.head(3).to_string())
    df2 = df2.reset_index().rename(columns={'index': '_orig_index'})
    df2['instance_id'] = df2['_orig_index'].astype(str)
    return df2

# --------------------------
# PLOT HELPERS (reusable) with unique naming
# --------------------------
def save_current_figure_unique(dirpath: str, basename: str, ext: str = 'png'):
    ensure_dir(dirpath)
    path = unique_path(dirpath, basename, ext)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path

def save_dataframe_unique_csv(df: pd.DataFrame, dirpath: str, basename: str):
    ensure_dir(dirpath)
    path = unique_csv_path(dirpath, basename)
    df.to_csv(path, index=False)
    return path

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# --------------------------
# MAIN PLOTS: pairwise + per-layout + overall
# --------------------------
def plot_pairwise_comparisons(df: pd.DataFrame, output_dir: str):
    print("plot_pairwise_comparisons: start")
    out_dir = os.path.join(output_dir, 'pairs')
    ensure_dir(out_dir)

    metric_nodes = None; metric_time = None; metric_cost = None
    try:
        metric_nodes = get_metric_column(df, ['Nodes Visited', 'Nodes Expanded', 'nodes_visited', 'nodesvisited'])
    except KeyError:
        pass
    try:
        metric_time = get_metric_column(df, ['Time (ms)', 'Runtime (ms)', 'time_ms', 'runtime_ms'])
    except KeyError:
        pass
    try:
        metric_cost = get_metric_column(df, ['Path Length', 'Path Cost', 'path_length', 'path_cost'])
    except KeyError:
        pass

    if metric_nodes is None and metric_time is None and metric_cost is None:
        raise KeyError("No recognized metric columns found (Nodes/Time/PathCost).")

    df2 = ensure_algorithm_column(df)
    has_layouts = 'Layout Type' in df2.columns

    for a_key, b_key in PAIRWISE_PAIRS:
        present = set(df2['__algo_norm'].unique())
        if not ({a_key, b_key} <= present):
            print(f"Skipping pair {a_key} vs {b_key} — not both present.")
            continue
        pair_base_dir = os.path.join(out_dir, f"{a_key}_vs_{b_key}")
        ensure_dir(pair_base_dir)
        df_pair_all = df2[df2['__algo_norm'].isin([a_key, b_key])].copy()

        # overall plots (all layouts)
        _produce_pairwise_plots_for_subset(df_pair_all, a_key, b_key, metric_nodes, metric_time, metric_cost, pair_base_dir)

        # per-layout
        if has_layouts:
            layouts = df_pair_all['Layout Type'].dropna().unique()
            for layout in layouts:
                layout_safe = sanitize_filename(layout)
                layout_dir = os.path.join(pair_base_dir, layout_safe)
                ensure_dir(layout_dir)
                df_pair_layout = df_pair_all[df_pair_all['Layout Type'] == layout]
                if df_pair_layout.empty:
                    continue
                print(f"  Pair {a_key} vs {b_key} — layout: {layout}  rows: {len(df_pair_layout)}")
                _produce_pairwise_plots_for_subset(df_pair_layout, a_key, b_key, metric_nodes, metric_time, metric_cost, layout_dir)

    print("plot_pairwise_comparisons: done")

def _produce_pairwise_plots_for_subset(df_pair, a_key, b_key, metric_nodes, metric_time, metric_cost, pair_dir):
    # barplot (mean ± sd)
    for metric_col, nice in [(metric_nodes, 'Nodes Expanded'), (metric_time, 'Runtime (ms)'), (metric_cost, 'Path Cost')]:
        if metric_col is None:
            continue
        plt.figure(figsize=PLOT_CONFIG['figure_sizes']['bar'])
        pal = {PLOT_CONFIG['display'][a_key]: PLOT_CONFIG['colors'][a_key], PLOT_CONFIG['display'][b_key]: PLOT_CONFIG['colors'][b_key]}
        sns.barplot(data=df_pair, x='__algo_display', y=metric_col, errorbar='sd', palette=pal)
        plt.title(f"{PLOT_CONFIG['display'].get(a_key)} vs {PLOT_CONFIG['display'].get(b_key)} — mean {nice}")
        plt.xlabel("Algorithm"); plt.ylabel(nice)
        save_current_figure_unique(pair_dir, f"pair_bar_{sanitize_filename(metric_col)}")

    # boxplots distribution
    for metric_col in [metric_nodes, metric_time, metric_cost]:
        if metric_col is None:
            continue
        plt.figure(figsize=PLOT_CONFIG['figure_sizes']['box'])
        sns.boxplot(data=df_pair, x='__algo_display', y=metric_col, palette=[PLOT_CONFIG['colors'].get(a_key), PLOT_CONFIG['colors'].get(b_key)])
        plt.title(f"{PLOT_CONFIG['display'].get(a_key)} vs {PLOT_CONFIG['display'].get(b_key)} — distribution ({metric_col})")
        plt.xlabel("Algorithm"); plt.ylabel(metric_col)
        plt.xticks(rotation=20)
        save_current_figure_unique(pair_dir, f"pair_box_{sanitize_filename(metric_col)}")

    # nodes vs path cost scatter with colored points + best-fit per algo (exponential/power/linear)
    if metric_nodes is not None and metric_cost is not None:
        plt.figure(figsize=PLOT_CONFIG['figure_sizes']['scatter'])
        palette_map = {PLOT_CONFIG['display'][a_key]: PLOT_CONFIG['colors'][a_key], PLOT_CONFIG['display'][b_key]: PLOT_CONFIG['colors'][b_key]}
        sns.scatterplot(data=df_pair, x=metric_cost, y=metric_nodes, hue='__algo_display', palette=palette_map, alpha=0.7, s=45)
        metrics_summary = []
        for algo in [a_key, b_key]:
            sub = df_pair[df_pair['__algo_norm'] == algo].dropna(subset=[metric_nodes, metric_cost])
            if sub.shape[0] >= 2:
                x = sub[metric_cost].to_numpy(); y = sub[metric_nodes].to_numpy()
                
                # Choose best fit (exponential/power/linear)
                fit_type, fit_params, r2 = choose_best_fit(x, y)
                
                mask = ~(np.isnan(x) | np.isnan(y))
                xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                
                if fit_type == 'exponential':
                    a, b = fit_params
                    ys = a * np.exp(b * xs)
                    fit_label = f"exp(fit, R²={r2:.3f})"
                elif fit_type == 'power':
                    a, b = fit_params
                    ys = a * np.power(xs, b)
                    fit_label = f"pow(fit, R²={r2:.3f})"
                else:  # linear
                    slope, intercept = fit_params
                    ys = slope * xs + intercept
                    fit_label = f"lin(fit, R²={r2:.3f})"
                
                plt.plot(xs, ys, linestyle='--', color=PLOT_CONFIG['colors'][algo], label=f"{PLOT_CONFIG['display'][algo]} {fit_label}")
                r, p = safe_pearson(x, y)
                metrics_summary.append({'algorithm': PLOT_CONFIG['display'][algo], 'fit_type': fit_type, 'r2': r2, 'pearson_r': r, 'n': int(mask.sum())})
        plt.xlabel(metric_cost); plt.ylabel(metric_nodes)
        plt.title(f"{PLOT_CONFIG['display'].get(a_key)} vs {PLOT_CONFIG['display'].get(b_key)} — Nodes vs Path Cost")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        txt_lines = [f"{m['algorithm']}: {m['fit_type']} (R²={m['r2']:.3f}, r={m['pearson_r']:.3f}, n={m['n']})" for m in metrics_summary]
        if txt_lines:
            plt.gcf().text(0.02, 0.02, "\n".join(txt_lines), fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        save_current_figure_unique(pair_dir, "pair_tradeoff_nodes_vs_path_colored")

    # Save numeric summary per pair subset
    rows = []
    for algo in [a_key, b_key]:
        sub = df_pair[df_pair['__algo_norm'] == algo]
        if sub.empty:
            continue
        row = {'algorithm': PLOT_CONFIG['display'].get(algo), 'count': len(sub)}
        for m in [metric_nodes, metric_time, metric_cost]:
            if m is None:
                continue
            row[f'{m}_mean'] = float(sub[m].mean()); row[f'{m}_median'] = float(sub[m].median()); row[f'{m}_std'] = float(sub[m].std())
        rows.append(row)
    if rows:
        df_sum = pd.DataFrame(rows)
        csv_path = save_dataframe_unique_csv(df_sum, pair_dir, f"summary_{sanitize_filename(pair_dir)}")
        print("Saved pair summary CSV:", csv_path)

# --------------------------
# Per-algo efficiency & per-layout
# --------------------------
def plot_per_algorithm_efficiency(df: pd.DataFrame, output_dir: str):
    print("plot_per_algorithm_efficiency: start")
    out_dir = os.path.join(output_dir, 'per_algorithm_efficiency')
    ensure_dir(out_dir)
    df2 = ensure_algorithm_column(df)
    try:
        metric_nodes = get_metric_column(df2, ['Nodes Visited', 'Nodes Expanded', 'nodes_visited'])
    except KeyError:
        print("No 'Nodes Visited' column; skipping per-algo efficiency.")
        return
    try:
        metric_cost = get_metric_column(df2, ['Path Length', 'Path Cost', 'path_length', 'path_cost'])
    except KeyError:
        metric_cost = None
    try:
        metric_time = get_metric_column(df2, ['Time (ms)', 'Runtime (ms)', 'time_ms', 'runtime_ms'])
    except KeyError:
        metric_time = None

    has_layouts = 'Layout Type' in df2.columns
    for algo in df2['__algo_norm'].unique():
        sub_all = df2[df2['__algo_norm'] == algo].dropna(subset=[metric_nodes])
        if sub_all.empty:
            continue
        algo_base_dir = os.path.join(out_dir, algo)
        ensure_dir(algo_base_dir)

        # overall
        _produce_per_algo_plots_for_subset(sub_all, algo, metric_nodes, metric_cost, metric_time, algo_base_dir)

        # per-layout
        if has_layouts:
            for layout in sub_all['Layout Type'].dropna().unique():
                layout_safe = sanitize_filename(layout)
                algo_layout_dir = os.path.join(algo_base_dir, layout_safe)
                ensure_dir(algo_layout_dir)
                sub = sub_all[sub_all['Layout Type'] == layout]
                if sub.empty:
                    continue
                print(f"  Algo {algo} — layout: {layout} rows: {len(sub)}")
                _produce_per_algo_plots_for_subset(sub, algo, metric_nodes, metric_cost, metric_time, algo_layout_dir)

    print("plot_per_algorithm_efficiency: done")

def _produce_per_algo_plots_for_subset(sub, algo, metric_nodes, metric_cost, metric_time, algo_dir):
    # Nodes vs Path Cost scatter + best-fit (exponential/power/linear)
    if metric_cost is not None:
        plt.figure(figsize=PLOT_CONFIG['figure_sizes']['scatter'])
        hue_field = 'Layout Type' if 'Layout Type' in sub.columns else None
        sns.scatterplot(data=sub, x=metric_cost, y=metric_nodes, hue=hue_field, palette='muted', alpha=0.6, s=40, legend='brief')
        x = sub[metric_cost].to_numpy(); y = sub[metric_nodes].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 2:
            fit_type, fit_params, r2 = choose_best_fit(x, y)
            xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
            
            if fit_type == 'exponential':
                a, b = fit_params
                ys = a * np.exp(b * xs)
                fit_label = f"{fit_type.upper()} (a={a:.3f}, b={b:.3f}, R²={r2:.3f})"
            elif fit_type == 'power':
                a, b = fit_params
                ys = a * np.power(xs, b)
                fit_label = f"{fit_type.upper()} (a={a:.3f}, b={b:.3f}, R²={r2:.3f})"
            else:  # linear
                slope, intercept = fit_params
                ys = slope * xs + intercept
                fit_label = f"LINEAR (m={slope:.3f}, b={intercept:.3f}, R²={r2:.3f})"
            
            plt.plot(xs, ys, color='red', linestyle='--', label=fit_label, linewidth=2)
            r, p = safe_pearson(x, y)
            info_text = f"{fit_type.upper()}: R²={r2:.3f}, Pearson r={r:.3f}, n={int(mask.sum())}"
            plt.gcf().text(0.02, 0.02, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        plt.xlabel(metric_cost); plt.ylabel(metric_nodes)
        plt.title(f"{PLOT_CONFIG['display'].get(algo)} — Nodes vs Path Cost")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        save_current_figure_unique(algo_dir, f"{algo}_nodes_vs_path_regression")

    # nodes_per_cost box
    if metric_cost is not None:
        sub2 = sub.dropna(subset=[metric_cost]).copy()
        sub2['nodes_per_cost'] = sub2[metric_nodes] / sub2[metric_cost].replace({0: np.nan})
        plt.figure(figsize=(7, 5))
        sns.boxplot(x=sub2['nodes_per_cost'])
        plt.title(f"{PLOT_CONFIG['display'].get(algo)} — nodes per path cost")
        save_current_figure_unique(algo_dir, f"{algo}_nodes_per_cost_box")

    # nodes_per_ms box
    if metric_time is not None:
        sub3 = sub.dropna(subset=[metric_time]).copy()
        sub3['nodes_per_ms'] = sub3[metric_nodes] / sub3[metric_time].replace({0: np.nan})
        plt.figure(figsize=(7, 5))
        sns.boxplot(x=sub3['nodes_per_ms'])
        plt.title(f"{PLOT_CONFIG['display'].get(algo)} — nodes per ms")
        save_current_figure_unique(algo_dir, f"{algo}_nodes_per_ms_box")

# --------------------------
# Unpaired paired analysis
# --------------------------
def plot_paired_scatter_and_stats(df: pd.DataFrame, baseline_key: str, test_key: str, metric_col: str, output_dir: str):
    out_dir = os.path.join(output_dir, 'pairs_efficiency'); ensure_dir(out_dir)
    df2 = make_instance_id(df.copy()); df2 = ensure_algorithm_column(df2)
    try:
        wide = df2.pivot_table(index='instance_id', columns='__algo_norm', values=metric_col)
    except Exception as e:
        print("Pivot error (paired scatter):", e); return
    if baseline_key not in wide.columns or test_key not in wide.columns:
        print(f"Paired scatter skip: missing {baseline_key} or {test_key}. Falling back to unpaired with regression.")
        unpaired_efficiency_analysis(df, baseline_key, test_key, metric_col, output_dir)
        return
    x = wide[baseline_key].to_numpy(); y = wide[test_key].to_numpy()
    mask = ~(pd.isna(x) | pd.isna(y))
    x = x[mask]; y = y[mask]
    if len(x) < MIN_PAIRED_SAMPLES:
        print(f"Not enough paired samples ({len(x)}) for {baseline_key} vs {test_key} on {metric_col}; falling back to unpaired analysis.")
        unpaired_efficiency_analysis(df, baseline_key, test_key, metric_col, output_dir)
        return

    plt.figure(figsize=(9, 7))
    sns.scatterplot(x=x, y=y, alpha=0.7, s=45)
    mn = min(np.nanmin(x), np.nanmin(y)); mx = max(np.nanmax(x), np.nanmax(y))
    plt.plot([mn, mx], [mn, mx], color='k', linestyle='--', linewidth=1)
    try:
        fit_type, fit_params, r2 = choose_best_fit(x, y)
        xs = np.linspace(mn, mx, 100)
        
        if fit_type == 'exponential':
            a, b = fit_params
            ys = a * np.exp(b * xs)
            fit_label = f'{fit_type.upper()} (a={a:.3f}, b={b:.3f}, R²={r2:.3f})'
        elif fit_type == 'power':
            a, b = fit_params
            ys = a * np.power(xs, b)
            fit_label = f'{fit_type.upper()} (a={a:.3f}, b={b:.3f}, R²={r2:.3f})'
        else:
            slope, intercept = fit_params
            ys = slope * xs + intercept
            fit_label = f'LINEAR (m={slope:.3f}, b={intercept:.3f}, R²={r2:.3f})'
        
        plt.plot(xs, ys, color='red', linestyle='-', linewidth=1.6, alpha=0.8, label=fit_label)
        r, p = safe_pearson(x, y)
    except Exception as e:
        fit_type, r2, r, p = 'error', float('nan'), float('nan'), float('nan')
        print(f"Fit error: {e}")
    plt.xlabel(f"{PLOT_CONFIG['display'].get(baseline_key)} {metric_col}")
    plt.ylabel(f"{PLOT_CONFIG['display'].get(test_key)} {metric_col}")
    plt.title(f"Paired: {PLOT_CONFIG['display'].get(baseline_key)} vs {PLOT_CONFIG['display'].get(test_key)} — {metric_col}")
    plt.legend(fontsize=9)
    pct_imp = 100.0 * (x - y) / np.where(x == 0, np.nan, x)
    mean_pct = np.nanmean(pct_imp); median_pct = np.nanmedian(pct_imp)
    stats_text = f"n={len(x)}  mean%={mean_pct:.1f}%  median%={median_pct:.1f}%\n{fit_type.upper()}: R²={r2:.3f}, Pearson r={r:.3f}"
    plt.gcf().text(0.02, 0.02, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
    save_current_figure_unique(out_dir, f"paired_scatter_with_fit_{baseline_key}_vs_{test_key}_{sanitize_filename(metric_col)}")

    try:
        metric_cost = get_metric_column(df, ['Path Length', 'Path Cost', 'path_length', 'path_cost'])
        df_pair = df[df['Algorithm'].apply(normalize_algorithm_name).isin([baseline_key, test_key])].copy()
        _produce_pairwise_plots_for_subset(df_pair, baseline_key, test_key, metric_col, None, metric_cost, out_dir)
    except KeyError:
        pass

def unpaired_efficiency_analysis(df: pd.DataFrame, baseline_key: str, test_key: str, metric_nodes: str, output_dir: str):
    out_dir = os.path.join(output_dir, 'pairs_efficiency_unpaired'); ensure_dir(out_dir)
    df2 = ensure_algorithm_column(df.copy())

    sub = df2[df2['__algo_norm'].isin([baseline_key, test_key])].copy()
    if sub.empty:
        print("Unpaired analysis skip: no rows for these algorithms.")
        return

    metric_cost = None; metric_time = None
    try:
        metric_cost = get_metric_column(sub, ['Path Length', 'Path Cost', 'path_length', 'path_cost'])
    except KeyError:
        pass
    try:
        metric_time = get_metric_column(sub, ['Time (ms)', 'Runtime (ms)', 'time_ms', 'runtime_ms'])
    except KeyError:
        pass

    # regression scatter nodes vs path cost
    if metric_cost is not None:
        plot_regression_nodes_vs_cost(sub, baseline_key, test_key, metric_nodes, metric_cost, out_dir)

    # nodes boxplot
    plt.figure(figsize=PLOT_CONFIG['figure_sizes']['box'])
    sns.boxplot(data=sub, x='__algo_display', y=metric_nodes)
    plt.title(f"Unpaired: {PLOT_CONFIG['display'].get(test_key)} vs {PLOT_CONFIG['display'].get(baseline_key)} — Nodes Visited")
    plt.xticks(rotation=20)
    save_current_figure_unique(out_dir, f"unpaired_box_nodes_{baseline_key}_vs_{test_key}")

    # numeric summaries and Mann-Whitney U
    summary_rows = []
    for algo in [baseline_key, test_key]:
        ssub = sub[sub['__algo_norm'] == algo]
        if ssub.empty:
            continue
        summary_rows.append({
            'algorithm': PLOT_CONFIG['display'].get(algo),
            'count': len(ssub),
            'nodes_median': float(ssub[metric_nodes].median()) if metric_nodes in ssub.columns else float('nan'),
            'nodes_mean': float(ssub[metric_nodes].mean()) if metric_nodes in ssub.columns else float('nan'),
            'nodes_std': float(ssub[metric_nodes].std()) if metric_nodes in ssub.columns else float('nan'),
            'path_median': float(ssub[metric_cost].median()) if metric_cost and metric_cost in ssub.columns else float('nan')
        })
    mw_p = float('nan'); median_diff = float('nan')
    if SCIPY and metric_nodes in sub.columns:
        a_vals = sub[sub['__algo_norm'] == baseline_key][metric_nodes].dropna().to_numpy()
        b_vals = sub[sub['__algo_norm'] == test_key][metric_nodes].dropna().to_numpy()
        if len(a_vals) >= 5 and len(b_vals) >= 5:
            try:
                stat, mw_p = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
            except Exception:
                mw_p = float('nan')
        if len(a_vals) > 0 and len(b_vals) > 0:
            median_diff = float(np.median(a_vals) - np.median(b_vals))
    else:
        a_med = sub[sub['__algo_norm'] == baseline_key][metric_nodes].median() if metric_nodes in sub.columns else np.nan
        b_med = sub[sub['__algo_norm'] == test_key][metric_nodes].median() if metric_nodes in sub.columns else np.nan
        if not (np.isnan(a_med) or np.isnan(b_med)):
            median_diff = float(a_med - b_med)

    percent_median_improvement = float('nan')
    try:
        a_med = sub[sub['__algo_norm'] == baseline_key][metric_nodes].median()
        b_med = sub[sub['__algo_norm'] == test_key][metric_nodes].median()
        if a_med and not math.isnan(a_med):
            percent_median_improvement = 100.0 * (a_med - b_med) / a_med
    except Exception:
        percent_median_improvement = float('nan')

    stats_out = {
        'baseline': PLOT_CONFIG['display'].get(baseline_key),
        'test': PLOT_CONFIG['display'].get(test_key),
        'median_diff_nodes': median_diff,
        'percent_median_improvement': percent_median_improvement,
        'mannwhitney_p': mw_p
    }
    summary_df = pd.DataFrame(summary_rows)
    csv1 = save_dataframe_unique_csv(summary_df, out_dir, f"{baseline_key}_vs_{test_key}_unpaired_summary_rows")
    pd.DataFrame([stats_out]).to_csv(unique_csv_path(out_dir, f"{baseline_key}_vs_{test_key}_unpaired_stats"), index=False)
    print(f"Unpaired summary saved for {baseline_key} vs {test_key} in {out_dir}; CSV: {csv1}")

def plot_regression_nodes_vs_cost(sub_df: pd.DataFrame, baseline_key: str, test_key: str, metric_nodes: str, metric_cost: str, out_dir: str):
    if metric_cost is None or metric_nodes is None:
        return
    try:
        plt.figure(figsize=PLOT_CONFIG['figure_sizes']['scatter'])
        sns.scatterplot(data=sub_df, x=metric_cost, y=metric_nodes, hue='__algo_display', alpha=0.6, s=30)
        for algo in sub_df['__algo_norm'].unique():
            part = sub_df[sub_df['__algo_norm'] == algo].dropna(subset=[metric_cost, metric_nodes])
            if len(part) >= 2:
                x = part[metric_cost].to_numpy(); y = part[metric_nodes].to_numpy()
                mask = ~(np.isnan(x) | np.isnan(y))
                coef = np.polyfit(x[mask], y[mask], 1)
                slope = float(coef[0])
                xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                plt.plot(xs, slope * xs + coef[1], linestyle='--', color=PLOT_CONFIG['colors'].get(algo, 'k'), label=f"{PLOT_CONFIG['display'].get(algo)} slope={slope:.3f}")
        plt.xlabel(metric_cost); plt.ylabel(metric_nodes)
        plt.title("Nodes vs Path Cost (unpaired regression)")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        save_current_figure_unique(out_dir, "unpaired_nodes_vs_cost_regression")
    except Exception as e:
        print("plot_regression_nodes_vs_cost error:", e)

# --------------------------
# Overall visualizations (heatmap, violin, tradeoffs)
# --------------------------
def plot_overall_violin_and_hist(df: pd.DataFrame, output_dir: str):
    out_dir = os.path.join(output_dir, 'overall')
    ensure_dir(out_dir)
    df2 = ensure_algorithm_column(df)
    metric_nodes = None; metric_time = None; metric_cost = None
    try: metric_nodes = get_metric_column(df2, ['Nodes Visited','Nodes Expanded','nodes_visited'])
    except KeyError: pass
    try: metric_time = get_metric_column(df2, ['Time (ms)','Runtime (ms)','time_ms','runtime_ms'])
    except KeyError: pass
    try: metric_cost = get_metric_column(df2, ['Path Length','Path Cost','path_length','path_cost'])
    except KeyError: pass

    for metric_col, title in [(metric_nodes, 'Nodes Visited'), (metric_time, 'Runtime (ms)')]:
        if metric_col is None: continue
        plt.figure(figsize=PLOT_CONFIG['figure_sizes']['violin'])
        palette_list = [PLOT_CONFIG['colors'].get(k) for k in PLOT_CONFIG['display'].keys() if k in df2['__algo_norm'].unique()]
        sns.violinplot(data=df2, x='__algo_display', y=metric_col, palette=palette_list)
        plt.title(f"Distribution of {title} by Algorithm")
        plt.xlabel('Algorithm'); plt.ylabel(title)
        plt.xticks(rotation=20)
        save_current_figure_unique(out_dir, f"violin_{sanitize_filename(metric_col)}")

    if metric_cost is not None:
        for algo in df2['__algo_norm'].unique():
            sub = df2[df2['__algo_norm'] == algo]
            if sub.empty: continue
            plt.figure(figsize=PLOT_CONFIG['figure_sizes']['hist'])
            sns.histplot(sub[metric_cost].dropna(), kde=True, bins=30)
            plt.title(f"{PLOT_CONFIG['display'].get(algo)} — {metric_cost} distribution")
            save_current_figure_unique(out_dir, f"{algo}_hist_{sanitize_filename(metric_cost)}")

def plot_tradeoff_time_vs_path(df: pd.DataFrame, output_dir: str):
    out_dir = os.path.join(output_dir, 'tradeoffs'); ensure_dir(out_dir)
    df2 = ensure_algorithm_column(df)
    try:
        metric_time = get_metric_column(df2, ['Time (ms)','Runtime (ms)','time_ms','runtime_ms'])
    except KeyError:
        metric_time = None
    try:
        metric_cost = get_metric_column(df2, ['Path Length','Path Cost','path_length','path_cost'])
    except KeyError:
        metric_cost = None

    if metric_time is None or metric_cost is None:
        print("Skipping tradeoff Time vs Path (missing columns)")
        return

    plt.figure(figsize=(10,7))
    df2['env_mode'] = df2['__algo_norm'].apply(lambda x: 'variable_cost' if 'variable_cost' in x else 'static')
    palette = {PLOT_CONFIG['display'].get(k): PLOT_CONFIG['colors'][k] for k in PLOT_CONFIG['display'].keys() if k in df2['__algo_norm'].unique()}
    sns.scatterplot(data=df2, x=metric_time, y=metric_cost, hue='__algo_display', style='env_mode', palette=palette, alpha=0.6, s=40)
    x = df2[metric_time].to_numpy(); y = df2[metric_cost].to_numpy()
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() >= 2:
        fit_type, fit_params, r2 = choose_best_fit(x, y)
        xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        
        if fit_type == 'exponential':
            a, b = fit_params
            ys = a * np.exp(b * xs)
            fit_label = f'{fit_type.upper()} (a={a:.3f}, b={b:.3f}, R²={r2:.3f})'
        elif fit_type == 'power':
            a, b = fit_params
            ys = a * np.power(xs, b)
            fit_label = f'{fit_type.upper()} (a={a:.3f}, b={b:.3f}, R²={r2:.3f})'
        else:
            slope, intercept = fit_params
            ys = slope * xs + intercept
            fit_label = f'LINEAR (m={slope:.3f}, b={intercept:.3f}, R²={r2:.3f})'
        
        plt.plot(xs, ys, color='black', linestyle='--', linewidth=2, label=fit_label)
        r, p = safe_pearson(x, y)
        info_text = f"{fit_type.upper()}: R²={r2:.3f}, Pearson r={r:.3f}, n={int(mask.sum())}"
        plt.gcf().text(0.02, 0.02, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel(metric_time); plt.ylabel(metric_cost)
    plt.title("Runtime vs Path Length (all algorithms)")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=9)
    save_current_figure_unique(out_dir, "tradeoff_time_vs_path_colored")

def compute_and_plot_percent_improvement_heatmap(df: pd.DataFrame, output_dir: str):
    out_dir = os.path.join(output_dir, 'heatmaps'); ensure_dir(out_dir)
    df2 = make_instance_id(df.copy()); df2 = ensure_algorithm_column(df2)
    try:
        metric_nodes = get_metric_column(df2, ['Nodes Visited','Nodes Expanded','nodes_visited'])
    except KeyError:
        print("Cannot compute percent improvement (no nodes column).")
        return

    def pair_improvement_df(df_all, baseline_key, test_key):
        sub = df_all[df_all['__algo_norm'].isin([baseline_key, test_key])]
        if sub.empty:
            return None
        wide = sub.pivot_table(index='instance_id', columns='__algo_norm', values=metric_nodes, aggfunc='first')
        layout_map = sub.drop_duplicates(subset=['instance_id'])[['instance_id'] + (['Layout Type'] if 'Layout Type' in sub.columns else [])]
        wide = wide.reset_index().merge(layout_map, on='instance_id', how='left')
        if baseline_key not in wide.columns or test_key not in wide.columns:
            return None
        wide['pct_improve'] = 100.0 * (wide[baseline_key] - wide[test_key]) / np.where(wide[baseline_key] == 0, np.nan, wide[baseline_key])
        return wide

    for baseline_key, test_key in PAIRWISE_PAIRS:
        wide = pair_improvement_df(df2, baseline_key, test_key)
        if wide is None or wide.shape[0] == 0:
            print(f"No paired instances for {baseline_key} vs {test_key}; skipping heatmap.")
            continue
        if 'Layout Type' in wide.columns:
            summary = wide.groupby('Layout Type')['pct_improve'].median().reset_index().rename(columns={'pct_improve':'median_pct_improve'})
            pivot = summary.pivot_table(index='Layout Type', values='median_pct_improve')
            csv_path = unique_csv_path(out_dir, f"median_pct_improve_{baseline_key}_vs_{test_key}")
            pivot.to_csv(csv_path)
            plt.figure(figsize=PLOT_CONFIG['figure_sizes']['heatmap'])
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap='Blues', cbar_kws={'label':'median % improvement (nodes)'})
            plt.title(f"Median % Improvement (nodes): {PLOT_CONFIG['display'][baseline_key]} vs {PLOT_CONFIG['display'][test_key]}")
            save_current_figure_unique(out_dir, f"heatmap_median_pct_{baseline_key}_vs_{test_key}")
            print(f"Saved heatmap for {baseline_key} vs {test_key} with {len(pivot)} layouts.")
        else:
            median_val = np.nanmedian(wide['pct_improve'])
            pd.DataFrame({'median_pct_improve':[median_val]}).to_csv(unique_csv_path(out_dir, f"median_pct_improve_overall_{baseline_key}_vs_{test_key}"), index=False)
            print(f"Overall median percent improvement {baseline_key} vs {test_key} = {median_val:.2f}%")

# --------------------------
# Summary & orchestration
# --------------------------
def compute_and_save_summary(df: pd.DataFrame, output_dir: str):
    out_dir = os.path.join(output_dir, 'summary'); ensure_dir(out_dir)
    df2 = ensure_algorithm_column(df)
    metric_candidates = [
        ['Nodes Visited','Nodes Expanded','nodes_visited'],
        ['Time (ms)','Runtime (ms)','time_ms','runtime_ms'],
        ['Path Length','Path Cost','path_length','path_cost']
    ]
    found = []
    for c in metric_candidates:
        try:
            found.append(get_metric_column(df2, c))
        except KeyError:
            found.append(None)
    rows = []
    for algo in df2['__algo_norm'].unique():
        sub = df2[df2['__algo_norm'] == algo]
        if sub.empty:
            continue
        r = {'algorithm': PLOT_CONFIG['display'].get(algo, algo), 'count': len(sub)}
        for m in found:
            if m is None:
                continue
            r[f'{m}_mean'] = float(sub[m].mean()); r[f'{m}_median'] = float(sub[m].median()); r[f'{m}_std'] = float(sub[m].std())
        rows.append(r)
    if rows:
        out_csv = unique_csv_path(out_dir, 'summary_by_algorithm')
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("Saved summary CSV:", out_csv)
    else:
        print("No metrics available for summary CSV.")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(current_dir, 'processed_data')
    if not os.path.isdir(processed_data_dir):
        raise FileNotFoundError("processed_data directory not found; place your CSV(s) there.")
    output_dir = create_output_directory(current_dir)

    print("RUN_ID:", RUN_ID)
    print("Loading processed CSVs from:", processed_data_dir)
    df = load_processed_data(processed_data_dir)
    print("Rows loaded:", len(df))
    print("Columns detected in CSV:", df.columns.tolist())
    print("First 3 rows preview:")
    print(df.head(3).to_string())

    df = compute_obstacle_density_from_grid_column(df)

    sns.set_style("whitegrid")

    try:
        df = ensure_algorithm_column(df)
    except KeyError as e:
        raise e

    try:
        nodes_metric_col = get_metric_column(df, ['Nodes Visited','Nodes Expanded','nodes_visited'])
    except KeyError:
        nodes_metric_col = None
        print("Warning: No 'Nodes Visited' column detected; some efficiency plots will be skipped.")

    compute_and_save_summary(df, output_dir)
    plot_pairwise_comparisons(df, output_dir)
    plot_per_algorithm_efficiency(df, output_dir)

    if nodes_metric_col:
        for a_key, b_key in PAIRWISE_PAIRS:
            if (a_key in df['__algo_norm'].values) and (b_key in df['__algo_norm'].values):
                print(f"Generating efficiency plots for pair: {a_key} vs {b_key}")
                plot_paired_scatter_and_stats(df, a_key, b_key, nodes_metric_col, output_dir)
            else:
                print(f"Pair {a_key} vs {b_key} not both present; skipping efficiency plots.")

    plot_overall_violin_and_hist(df, output_dir)
    plot_tradeoff_time_vs_path(df, output_dir)
    compute_and_plot_percent_improvement_heatmap(df, output_dir)

    print("Visualizations and summaries saved to:", output_dir)

if __name__ == '__main__':
    main()
