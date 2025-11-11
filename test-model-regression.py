#!/usr/bin/env python3
"""
Test the ML model predictions to diagnose regression.
Check if predictions changed unexpectedly between versions.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs

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
print("Loading SavedModel from ml_heuristic_savedmodel_static/...")
try:
    model = tf.saved_model.load("ml_heuristic_savedmodel_static")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Get the concrete function
try:
    concrete_func = model.signatures['serving_default']
    print(f"✓ Concrete function loaded")
except Exception as e:
    print(f"✗ Failed to get serving_default signature: {e}")
    exit(1)

print("\nTesting model on known cases from your browser tests:")
print("=" * 80)

# Test cases from your browser output
test_cases = [
    {"name": "Test 1", "start": (13, 1), "goal": (5, 15)},
    {"name": "Test 2", "start": (14, 4), "goal": (7, 15)},
    {"name": "Test 3", "start": (17, 3), "goal": (8, 15)},
]

# Generate random grids for each test
max_cost = math.sqrt(2) * (GRID_SIZE - 1)

for test_case in test_cases:
    start = test_case["start"]
    goal = test_case["goal"]
    name = test_case["name"]
    
    print(f"\n{name}: start={start}, goal={goal}")
    
    # Generate a random grid
    grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.5, 0.5])
    grid[start] = 0
    grid[goal] = 0
    
    # Get ground truth cost from A*
    true_cost = a_star(grid, start, goal)
    if true_cost is None:
        print(f"  A* path not found, skipping")
        continue
    
    base_h = octile_distance(start, goal)
    true_residual = true_cost - base_h
    
    # Prepare model inputs
    grid_input = tf.constant(grid.reshape(1, GRID_SIZE, GRID_SIZE, 1), dtype=tf.float32)
    normalized_dist = base_h / (math.sqrt(2) * (GRID_SIZE - 1))
    sg = [start[0] / (GRID_SIZE - 1), start[1] / (GRID_SIZE - 1), 
          goal[0] / (GRID_SIZE - 1), goal[1] / (GRID_SIZE - 1), 
          normalized_dist]
    sg_input = tf.constant([sg], dtype=tf.float32)
    
    print(f"  Grid: {np.sum(grid)} walls out of 400")
    print(f"  Input tensors: grid_input shape={grid_input.shape}, sg_input shape={sg_input.shape}")
    print(f"  Octile distance: {base_h:.3f}")
    print(f"  True cost (A*): {true_cost:.3f}")
    print(f"  True residual: {true_residual:.3f}")
    
    # Try different input name combinations
    success = False
    input_combos = [
        {"grid": grid_input, "start_goal": sg_input},
        {"start_goal": sg_input, "grid": grid_input},
    ]
    
    for inputs in input_combos:
        try:
            result = concrete_func(**inputs)
            # Result is a dict with the output tensor
            pred_val = result['output_0'].numpy()[0][0]
            predicted_cost = pred_val * max_cost
            predicted_residual = predicted_cost - base_h
            
            print(f"  Model prediction: normalized={pred_val:.4f}, cost={predicted_cost:.3f}, residual={predicted_residual:.3f}")
            print(f"  ✓ Success with inputs: {list(inputs.keys())}")
            success = True
            break
        except Exception as e:
            pass
    
    if not success:
        # Try direct call
        try:
            result = concrete_func(grid=grid_input, start_goal=sg_input)
            pred_val = result['output_0'].numpy()[0][0]
            predicted_cost = pred_val * max_cost
            predicted_residual = predicted_cost - base_h
            print(f"  Model prediction: normalized={pred_val:.4f}, cost={predicted_cost:.3f}, residual={predicted_residual:.3f}")
            print(f"  ✓ Success with direct call")
        except Exception as e:
            print(f"  ✗ Failed to get prediction: {e}")

print("\n" + "=" * 80)
print("Diagnosis: Testing actual model behavior and comparing with expected outputs")
