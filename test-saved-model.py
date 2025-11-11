#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import math

GRID_SIZE = 20

def octile_distance(r1, c1, r2, c2):
    dx = abs(r1 - r2)
    dy = abs(c1 - c2)
    D = 1
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def normalize_distance(dist):
    max_dist = math.sqrt(2) * (GRID_SIZE - 1)
    return dist / max_dist

def test_model():
    print("Loading SavedModel from ml_heuristic_savedmodel_static...")
    model = tf.saved_model.load('ml_heuristic_savedmodel_static')
    infer = model.signatures['serving_default']
    
    print("Model loaded successfully!")
    
    # Test Grid Variations (50 random grids, same start/goal)
    print("\n=== Testing Grid Variations (50 grids) ===")
    start_row, start_col, goal_row, goal_col = 14, 3, 7, 13
    dist = octile_distance(start_row, start_col, goal_row, goal_col)
    sg_array = np.array([[start_row/19.0, start_col/19.0, goal_row/19.0, goal_col/19.0, normalize_distance(dist)]], dtype=np.float32)
    
    predictions = []
    for i in range(50):
        grid_arr = np.random.choice([0.0, 1.0], size=(1, 20, 20, 1), p=[0.8, 0.2]).astype(np.float32)
        wall_count = int(np.sum(grid_arr))
        
        result = infer(grid=tf.constant(grid_arr), start_goal=tf.constant(sg_array))
        pred = float(result['output_0'].numpy()[0, 0])
        predictions.append({'test': i+1, 'walls': wall_count, 'pred': pred})
    
    preds_vals = [p['pred'] for p in predictions]
    min_val = min(preds_vals)
    max_val = max(preds_vals)
    range_val = max_val - min_val
    
    print("Grid Variation results (first 10):")
    for p in predictions[:10]:
        print(f"  Test {p['test']:2d}: {p['walls']:3d} walls -> {p['pred']:.6f}")
    print(f"Min: {min_val:.6f}, Max: {max_val:.6f}, Range: {range_val:.6f}")
    if range_val > 0.1:
        print("✓ Grid sensitivity is EXCELLENT")
    
    # Test Start/Goal Variations (50 pairs, same grid)
    print("\n=== Testing Start/Goal Variations (50 pairs) ===")
    grid_arr = np.random.choice([0.0, 1.0], size=(1, 20, 20, 1), p=[0.8, 0.2]).astype(np.float32)
    
    sg_predictions = []
    for i in range(50):
        sr = np.random.randint(0, 20)
        sc = np.random.randint(0, 20)
        gr = np.random.randint(0, 20)
        gc = np.random.randint(0, 20)
        d = octile_distance(sr, sc, gr, gc)
        sg = np.array([[sr/19.0, sc/19.0, gr/19.0, gc/19.0, normalize_distance(d)]], dtype=np.float32)
        
        result = infer(grid=tf.constant(grid_arr), start_goal=tf.constant(sg))
        pred = float(result['output_0'].numpy()[0, 0])
        sg_predictions.append({'test': i+1, 'dist': d, 'pred': pred})
    
    sg_preds_vals = [p['pred'] for p in sg_predictions]
    sg_min = min(sg_preds_vals)
    sg_max = max(sg_preds_vals)
    sg_range = sg_max - sg_min
    
    print("Start/Goal Variation results (first 10):")
    for p in sg_predictions[:10]:
        print(f"  Test {p['test']:2d}: dist={p['dist']:6.2f} -> {p['pred']:.6f}")
    print(f"Min: {sg_min:.6f}, Max: {sg_max:.6f}, Range: {sg_range:.6f}")
    
    if sg_range > 0.05:
        print("✓✓✓ SUCCESS! Start/Goal sensitivity is STRONG (range > 0.05)")
        return True
    elif sg_range > 0.01:
        print("⚠ WARNING: Start/Goal sensitivity is WEAK (range 0.01-0.05)")
        return False
    else:
        print("✗ FAILURE: Start/Goal sensitivity is VERY WEAK (range < 0.01)")
        return False

if __name__ == '__main__':
    success = test_model()
    exit(0 if success else 1)
