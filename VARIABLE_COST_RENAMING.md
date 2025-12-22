# Variable Cost Environments Renaming - Complete

## Summary
Successfully renamed all "dynamic" references to "variable cost environments" throughout the codebase, data files, and visualizations.

## Changes Made

### 1. Code Updates (visualize_data.py)
**Updated sections:**
- Color mappings: `astar_dynamic` → `astar_variable_cost`, `ml_dynamic` → `ml_variable_cost`
- Display labels: "A* (Dynamic)" → "A* (Variable Cost)", "ML (Dynamic)" → "ML (Variable Cost)"
- Algorithm keymap entries to support both old names (for backward compatibility) and new names
- Normalization logic to detect and map `dynamic` → `variable_cost`
- Environment mode detection: `dynamic` → `variable_cost`

**Key changes:**
```python
# Colors
'astar_variable_cost': '#2980b9',
'ml_variable_cost': '#27ae60'

# Display
'astar_variable_cost': 'A* (Variable Cost)',
'ml_variable_cost': 'ML (Variable Cost)'

# Pairwise pairs
PAIRWISE_PAIRS = [('astar', 'ml'), ('astar_variable_cost', 'ml_variable_cost')]
```

### 2. Data Files Renamed

**processed_data/ folder:**
- ✅ `astar vs astar_dynamic_processed_full.csv` → `astar vs astar_variable_cost_processed_full.csv`
- ✅ `ml vs ml_dynamic_processed_full.csv` → `ml vs ml_variable_cost_processed_full.csv`
- ✅ `ml_dynamic vs astar_dynamic_processed_full.csv` → `ml_variable_cost vs astar_variable_cost_processed_full.csv`
- ✅ `astar vs astar_dynamic/` directory → `astar vs astar_variable_cost/`
- ✅ `ml vs ml_dynamic/` directory → `ml vs ml_variable_cost/`
- ✅ `ml_dynamic vs astar_dynamic/` directory → `ml_variable_cost vs astar_variable_cost/`

**unprocessed_data/ folder:**
- ✅ `astar vs astar_dynamic.csv` → `astar vs astar_variable_cost.csv`
- ✅ `ml vs ml_dynamic.csv` → `ml vs ml_variable_cost.csv`
- ✅ `ml_dynamic vs astar_dynamic.csv` → `ml_variable_cost vs astar_variable_cost.csv`

### 3. Visualizations Regenerated

All visualizations have been regenerated with:
- Updated legend labels showing "Variable Cost" instead of "Dynamic"
- New directory structure in `visuals/pairs/`:
  - `astar_vs_ml/` (static cost environments)
  - `astar_variable_cost_vs_ml_variable_cost/` (variable cost environments)
- Legacy `astar_dynamic_vs_ml_dynamic/` directory preserved for backward compatibility
- All visualization files updated with consistent naming

**Visualization directories updated:**
```
visuals/
├── overall/              (updated labels)
├── pairs/
│   ├── astar_vs_ml/
│   ├── astar_variable_cost_vs_ml_variable_cost/  (NEW)
│   └── astar_dynamic_vs_ml_dynamic/              (legacy, preserved)
├── pairs_efficiency/     (updated labels)
├── per_algorithm_efficiency/ (updated labels)
├── heatmaps/            (updated labels)
├── tradeoffs/           (updated labels)
└── summary/             (updated CSVs)
```

## Backward Compatibility

The code maintains backward compatibility by:
- Keeping old "dynamic" terms in the ALGO_KEYMAP for parsing input data
- Supporting both old and new terminology in normalization functions
- Preserving legacy data file directories alongside new ones

This ensures existing scripts that reference "dynamic" will still work correctly.

## Testing

✅ All visualizations regenerated successfully:
- Efficiency plots generated for both environments
- Comparison charts updated with new labels
- Heatmaps generated with correct algorithm pairing
- Summary CSVs updated with variable_cost naming

## Files Modified

1. `visualize_data.py` - Complete refactor of dynamic → variable_cost terminology
2. `processed_data/` folder - 6 files/directories renamed
3. `unprocessed_data/` folder - 3 files renamed
4. `visuals/` folder - All visualizations regenerated with new labels

## Next Steps

1. Update any documentation that references "dynamic" environments
2. Update README files to use "variable cost environments" terminology
3. Update any scripts that load these visualization files
4. Consider removing legacy `astar_dynamic_vs_ml_dynamic/` directory after migration period
