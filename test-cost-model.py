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

def test_cost_predictions():
    """Test what the model predicts for cost and compute residuals."""
    print("Loading SavedModel...")
    model = tf.saved_model.load('ml_heuristic_savedmodel_static')
    infer = model.signatures['serving_default']
    
    # Create a fixed grid
    grid_arr = np.ones((1, 20, 20, 1), dtype=np.float32) * 0.0  # Empty grid
    
    print("\n=== Predictions by Distance (predicting COST now, not residual) ===")
    print("Distance | Predicted Cost | Octile | Residual (Cost-Octile)")
    print("-" * 65)
    
    max_cost = math.sqrt(2) * (GRID_SIZE - 1)
    distances = [1, 2, 3, 5, 10, 15, 20, 27]
    for target_dist in distances:
        # Find start/goal pair with approximately this distance
        for sr in range(20):
            for sc in range(20):
                for gr in range(20):
                    for gc in range(20):
                        d = octile_distance(sr, sc, gr, gc)
                        if abs(d - target_dist) < 0.5:
                            sg = np.array([[sr/19.0, sc/19.0, gr/19.0, gc/19.0, normalize_distance(d)]], dtype=np.float32)
                            result = infer(grid=tf.constant(grid_arr), start_goal=tf.constant(sg))
                            pred_normalized = float(result['output_0'].numpy()[0, 0])
                            pred_cost = pred_normalized * max_cost  # Denormalize
                            residual = pred_cost - d
                            
                            print(f"{d:8.1f} | {pred_cost:14.4f} | {d:6.2f} | {residual:10.4f}")
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

    # Test Start/Goal Variations (50 pairs, same grid)
    print("\n=== Testing Start/Goal Residual Variations (50 pairs) ===")
    print("(Higher variance = model better understands distance impact on residual)")
    
    sg_preds = []
    for i in range(50):
        sr = np.random.randint(0, 20)
        sc = np.random.randint(0, 20)
        gr = np.random.randint(0, 20)
        gc = np.random.randint(0, 20)
        d = octile_distance(sr, sc, gr, gc)
        sg = np.array([[sr/19.0, sc/19.0, gr/19.0, gc/19.0, normalize_distance(d)]], dtype=np.float32)
        
        result = infer(grid=tf.constant(grid_arr), start_goal=tf.constant(sg))
        pred_cost = float(result['output_0'].numpy()[0, 0]) * max_cost
        residual = pred_cost - d
        sg_preds.append({'dist': d, 'residual': residual})
    
    residuals = [p['residual'] for p in sg_preds]
    print(f"Residual Min: {min(residuals):.4f}, Max: {max(residuals):.4f}, Range: {max(residuals) - min(residuals):.4f}")
    print(f"Residual Mean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}")
    
    if max(residuals) - min(residuals) > 0.1:
        print("✓✓✓ EXCELLENT! Residuals vary significantly with distance")
    elif max(residuals) - min(residuals) > 0.05:
        print("✓✓ GOOD! Residuals show reasonable distance dependence")
    else:
        print("✗ POOR! Residuals still too constant")

if __name__ == '__main__':
    test_cost_predictions()
