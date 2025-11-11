#!/usr/bin/env python3
"""
Check the statistical distribution of model predictions across various test cases
to see if there's a pattern where certain grid types cause bad predictions.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import math
import random
from queue import PriorityQueue

GRID_SIZE = 20

def octile_distance(start, goal):
    dx = abs(start[0] - goal[0])
    dy = abs(start[1] - goal[1])
    D = 1
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def is_valid(r, c, grid):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and grid[r, c] == 0

def a_star(grid, start, goal):
    openset = PriorityQueue()
    openset.put((0, start))
    g_score = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while not openset.empty():
        _, current = openset.get()
        if current == goal:
            return g_score[current]
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if is_valid(neighbor[0], neighbor[1], grid):
                tentative_g_score = g_score[current] + (math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + octile_distance(neighbor, goal)
                    openset.put((f_score, neighbor))
    return None

# Load the SavedModel
print("Loading SavedModel...")
model = tf.saved_model.load("ml_heuristic_savedmodel_static")
concrete_func = model.signatures['serving_default']
max_cost = math.sqrt(2) * (GRID_SIZE - 1)

print("\nTesting model predictions on 50 random grids with 5 random start/goal pairs each")
print("=" * 100)

results = {'good': [], 'bad': [], 'terrible': []}

for test_num in range(50):
    grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.5, 0.5]).astype(np.float32)
    
    test_results = []
    for pair_num in range(5):
        start = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        goal = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        
        if grid[start] == 1 or grid[goal] == 1 or start == goal:
            continue
        
        true_cost = a_star(grid, start, goal)
        if true_cost is None:
            continue
        
        base_h = octile_distance(start, goal)
        true_residual = true_cost - base_h
        
        grid_input = tf.constant(grid.reshape(1, GRID_SIZE, GRID_SIZE, 1), dtype=tf.float32)
        normalized_dist = base_h / (math.sqrt(2) * (GRID_SIZE - 1))
        sg = [start[0] / (GRID_SIZE - 1), start[1] / (GRID_SIZE - 1), 
              goal[0] / (GRID_SIZE - 1), goal[1] / (GRID_SIZE - 1), 
              normalized_dist]
        sg_input = tf.constant([sg], dtype=tf.float32)
        
        result = concrete_func(grid=grid_input, start_goal=sg_input)
        pred_val = result['output_0'].numpy()[0][0]
        predicted_cost = pred_val * max_cost
        predicted_residual = predicted_cost - base_h
        
        error = abs(predicted_residual - true_residual)
        worse_than_octile = predicted_residual < 0  # Negative prediction makes A* worse
        much_worse = error > 5  # Prediction off by more than 5
        
        test_results.append({
            'true_residual': true_residual,
            'pred_residual': predicted_residual,
            'error': error,
            'worse_than_octile': worse_than_octile,
            'much_worse': much_worse
        })
    
    if len(test_results) == 0:
        continue
    
    # Categorize this grid based on prediction quality
    avg_error = np.mean([r['error'] for r in test_results])
    worse_count = sum(1 for r in test_results if r['worse_than_octile'])
    terrible_count = sum(1 for r in test_results if r['much_worse'])
    
    grid_stats = {
        'walls': np.sum(grid),
        'avg_error': avg_error,
        'worse_than_octile': worse_count,
        'terrible_predictions': terrible_count,
        'test_pairs': len(test_results)
    }
    
    if worse_count >= 3:
        results['terrible'].append(grid_stats)
        if len(results['terrible']) <= 3:
            print(f"Grid {test_num}: ⚠️ TERRIBLE - {worse_count}/5 predictions worse than octile, {terrible_count} terrible predictions")
            print(f"  Walls: {np.sum(grid)}, Avg Error: {avg_error:.2f}")
            for i, r in enumerate(test_results[:2]):
                print(f"    Pair {i}: True residual={r['true_residual']:.2f}, Predicted={r['pred_residual']:.2f}, Error={r['error']:.2f}")
    elif worse_count >= 1:
        results['bad'].append(grid_stats)
    else:
        results['good'].append(grid_stats)

print("\n" + "=" * 100)
print(f"Summary:")
print(f"  Good grids (0 worse): {len(results['good'])}")
print(f"  Bad grids (1-2 worse): {len(results['bad'])}")
print(f"  Terrible grids (3+ worse): {len(results['terrible'])}")

if len(results['terrible']) > 0:
    print("\nTerrible grid statistics:")
    avg_walls_terrible = np.mean([g['walls'] for g in results['terrible']])
    print(f"  Average walls: {avg_walls_terrible:.0f}/400")
    print(f"  Average error: {np.mean([g['avg_error'] for g in results['terrible']]):.2f}")
    print(f"\nMODEL ISSUE DETECTED: {len(results['terrible'])} grids cause model to output NEGATIVE residuals!")
else:
    print("\nModel appears to be working correctly on random grids")
