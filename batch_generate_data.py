#!/usr/bin/env python3
"""
Batch test data generator: A* (octile) vs A* (ML heuristic)
Generates comprehensive comparison data across grid types and obstacle densities
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import time
import math
import numpy as np
import tensorflow as tf
from queue import PriorityQueue
from datetime import datetime

GRID_SIZE = 20
TESTS_PER_CONFIG = 50
GRID_TYPES = ['random', 'maze', 'clustered', 'mixed']
DENSITIES = [0, 10, 20, 30, 40]

def octile(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)

def is_valid(r, c, grid):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and not grid[r][c]['isWall']

def a_star(grid, start, goal, heuristic_fn):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    nodes_expanded = 0

    while not open_set.empty():
        _, current = open_set.get()
        if current in closed:
            continue
        
        closed.add(current)
        nodes_expanded += 1
        
        if current == goal:
            path_cost = g_score[current]
            return nodes_expanded, path_cost

        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if not is_valid(nr, nc, grid):
                continue
            
            neighbor = (nr, nc)
            cost = math.sqrt(2) if dr != 0 and dc != 0 else 1
            tentative_g = g_score[current] + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_fn(neighbor, goal)
                open_set.put((f_score, neighbor))
        
        if nodes_expanded > 5000:
            return nodes_expanded, float('inf')
    
    return nodes_expanded, float('inf')

def generate_grid(grid_type, density):
    grid = [[{'isWall': False, 'cost': 1.0} for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    if grid_type == 'random':
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if np.random.rand() < density / 100:
                    grid[r][c]['isWall'] = True
    
    elif grid_type == 'maze':
        for r in range(0, GRID_SIZE, 2):
            for c in range(GRID_SIZE):
                grid[r][c]['isWall'] = True if np.random.rand() < density / 100 else False
        for c in range(0, GRID_SIZE, 2):
            for r in range(GRID_SIZE):
                grid[r][c]['isWall'] = True if np.random.rand() < (density / 100) * 0.5 else False
    
    elif grid_type == 'clustered':
        num_clusters = max(1, int(GRID_SIZE * density / 200))
        for _ in range(num_clusters):
            cr, cc = np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)
            for r in range(max(0, cr-2), min(GRID_SIZE, cr+3)):
                for c in range(max(0, cc-2), min(GRID_SIZE, cc+3)):
                    grid[r][c]['isWall'] = np.random.rand() < (density / 100)
    
    elif grid_type == 'mixed':
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r % 3 == 0 or c % 3 == 0) and np.random.rand() < (density / 100):
                    grid[r][c]['isWall'] = True
    
    return grid

def grid_to_tensor(grid):
    arr = np.zeros((20, 20, 1), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            arr[r, c, 0] = 1.0 if grid[r][c]['isWall'] else 0.0
    return arr

def predict_residual(model, grid, start, goal):
    try:
        grid_tensor = grid_to_tensor(grid)
        max_dist = math.sqrt(2) * (GRID_SIZE - 1)
        sg_dist = octile(start, goal) / max_dist if max_dist > 0 else 0
        sg_array = np.array([[start[0]/19, start[1]/19, goal[0]/19, goal[1]/19, sg_dist]], dtype=np.float32)
        
        result = model.signatures['serving_default'](
            grid=tf.constant(grid_tensor[np.newaxis, ...]),
            start_goal=tf.constant(sg_array)
        )
        return float(result['output'].numpy()[0, 0])
    except:
        return 0.0

def ml_heuristic(current, goal, grid, model):
    octile_val = octile(current, goal)
    if model is None:
        return octile_val
    
    residual = predict_residual(model, grid, current, goal)
    residual = max(0, min(residual, 5.0))
    return octile_val + residual * 0.7

print("Loading ML model...")
try:
    ml_model = tf.saved_model.load('ml_heuristic_savedmodel_static')
except:
    print("Warning: Could not load ML model, will use octile only")
    ml_model = None

results = []
total_tests = len(GRID_TYPES) * len(DENSITIES) * TESTS_PER_CONFIG
test_count = 0

for grid_type in GRID_TYPES:
    for density in DENSITIES:
        print(f"\nTesting {grid_type} @ {density}% density...")
        
        for test_num in range(TESTS_PER_CONFIG):
            grid = generate_grid(grid_type, density)
            
            start = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
            goal = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
            
            while grid[start[0]][start[1]]['isWall']:
                start = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
            while grid[goal[0]][goal[1]]['isWall']:
                goal = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
            
            def octile_heur(curr, g):
                return octile(curr, g)
            
            def ml_heur(curr, g):
                return ml_heuristic(curr, g, grid, ml_model)
            
            start_time = time.time()
            astar_nodes, astar_cost = a_star(grid, start, goal, octile_heur)
            astar_time = time.time() - start_time
            
            start_time = time.time()
            ml_nodes, ml_cost = a_star(grid, start, goal, ml_heur)
            ml_time = time.time() - start_time
            
            results.append({
                'grid_type': grid_type,
                'density': density,
                'start_row': start[0],
                'start_col': start[1],
                'goal_row': goal[0],
                'goal_col': goal[1],
                'euclidean_distance': math.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2),
                'octile_distance': octile(start, goal),
                'astar_nodes': astar_nodes,
                'astar_cost': astar_cost,
                'astar_time_ms': astar_time * 1000,
                'ml_nodes': ml_nodes,
                'ml_cost': ml_cost,
                'ml_time_ms': ml_time * 1000,
                'nodes_ratio': ml_nodes / astar_nodes if astar_nodes > 0 else 1.0,
            })
            
            test_count += 1
            if test_count % 10 == 0:
                print(f"  Completed {test_count}/{total_tests} tests...")

print("\nExporting to CSV...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f'batch_test_data_{timestamp}.csv'

with open(csv_file, 'w', newline='') as f:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"✓ Data exported to {csv_file}")
print(f"Total tests: {len(results)}")
print(f"Summary stats:")
print(f"  A* avg nodes: {np.mean([r['astar_nodes'] for r in results]):.1f}")
print(f"  ML avg nodes: {np.mean([r['ml_nodes'] for r in results]):.1f}")
print(f"  ML avg time: {np.mean([r['ml_time_ms'] for r in results]):.2f}ms")
