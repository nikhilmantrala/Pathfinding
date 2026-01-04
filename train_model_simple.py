#!/usr/bin/env python3
import numpy as np
import math
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import time

GRID_SIZE = 20
NUM_SAMPLES = 2000
OBSTACLE_PROB = 0.4
VARIABLE_COST_PROB = 0.3
MIN_RESIDUAL = 4.0
MAX_RESIDUAL = float('inf')

def generate_3channel_grid():
    obstacles = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[1-OBSTACLE_PROB, OBSTACLE_PROB])
    costs = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if obstacles[i, j] == 1:
                costs[i, j] = float('inf')
            elif np.random.random() < VARIABLE_COST_PROB:
                costs[i, j] = np.random.uniform(1.5, 3.0)
            else:
                costs[i, j] = 1.0
    
    distance_field = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if obstacles[i, j] == 0:
                min_dist = min(i, j, GRID_SIZE-1-i, GRID_SIZE-1-j)
                distance_field[i, j] = min_dist / GRID_SIZE
    
    return np.stack([obstacles, costs, distance_field], axis=2)

def is_valid(r, c, grid):
    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
        obstacle = grid[r, c, 0]
        cost = grid[r, c, 1]
        return obstacle == 0 and cost != float('inf')
    return False

def get_cost(grid, r, c):
    if is_valid(r, c, grid):
        return float(grid[r, c, 1])
    return float('inf')

def octile(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)

def a_star(grid, start, goal):
    openset = PriorityQueue()
    openset.put((0, start))
    g_score = {start: 0}
    
    while not openset.empty():
        _, current = openset.get()
        if current == goal:
            return g_score[current]
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = (current[0] + dr, current[1] + dc)
                if is_valid(neighbor[0], neighbor[1], grid):
                    base_cost = math.sqrt(2) if dr != 0 and dc != 0 else 1
                    tile_cost = get_cost(grid, neighbor[0], neighbor[1])
                    cost = base_cost * tile_cost
                    new_g = g_score[current] + cost
                    if neighbor not in g_score or new_g < g_score[neighbor]:
                        g_score[neighbor] = new_g
                        f = new_g + octile(neighbor, goal)
                        openset.put((f, neighbor))
    return None

print("Generating training data...")
grids, start_goals, residuals = [], [], []
hard_cases = []
max_dist = math.sqrt(2) * (GRID_SIZE - 1)

for i in range(NUM_SAMPLES):
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{NUM_SAMPLES}")
    
    grid = generate_3channel_grid()
    
    for _ in range(5):
        start = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        goal = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        
        if not is_valid(start[0], start[1], grid) or not is_valid(goal[0], goal[1], grid):
            continue
        
        actual_cost = a_star(grid, start, goal)
        if actual_cost is None:
            continue
        
        h_cost = octile(start, goal)
        residual = actual_cost - h_cost
        
        if residual < MIN_RESIDUAL or residual > MAX_RESIDUAL:
            continue
        
        grids.append(grid)
        start_goals.append([
            start[0] / (GRID_SIZE - 1),
            start[1] / (GRID_SIZE - 1),
            goal[0] / (GRID_SIZE - 1),
            goal[1] / (GRID_SIZE - 1)
        ])
        residuals.append([residual])
        
        if residual > 5.0:
            hard_cases.append((len(grids) - 1, residual))

grids = np.array(grids)
start_goals = np.array(start_goals)
residuals = np.array(residuals)

print(f"\nGenerated {len(grids)} valid samples")
print(f"Grid shape: {grids.shape} (3-channel input)")
print(f"Residual range: [{residuals.min():.2f}, {residuals.max():.2f}]")
print(f"Residual mean: {residuals.mean():.2f}, std: {residuals.std():.2f}")
print(f"Hard cases identified: {len(hard_cases)}")

hard_cases.sort(key=lambda x: x[1], reverse=True)
hard_indices = [idx for idx, _ in hard_cases[:min(len(hard_cases)//5, 100)]]

print(f"Including {len(hard_indices)} hard cases (residual > 5.0) in training")

grid_train, grid_test, sg_train, sg_test, res_train, res_test = train_test_split(
    grids, start_goals, residuals, test_size=0.2, random_state=42
)

print("\nBuilding model...")
grid_input = keras.Input(shape=(GRID_SIZE, GRID_SIZE, 3), name='grid')
coord_input = keras.Input(shape=(4,), name='coordinates')

x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(grid_input)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = keras.layers.GlobalAveragePooling2D()(x)

y = keras.layers.Dense(32, activation='relu')(coord_input)
y = keras.layers.Dense(16, activation='relu')(y)

combined = keras.layers.Concatenate()([x, y])
output = keras.layers.Dense(32, activation='relu')(combined)
output = keras.layers.Dense(1, activation='relu')(output)

model = keras.Model(inputs=[grid_input, coord_input], outputs=output)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("Training...")
model.fit(
    [grid_train, sg_train],
    res_train,
    validation_data=([grid_test, sg_test], res_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

print("\nExporting model...")
tf.saved_model.save(model, 'ml_heuristic_savedmodel_simple')

print("Converting to TensorFlow.js...")
import subprocess
subprocess.run([
    'tensorflowjs_converter',
    '--input_format', 'tf_saved_model',
    'ml_heuristic_savedmodel_simple',
    'web_model_simple'
], check=True)

print("Done! Model saved to ml_heuristic_savedmodel_simple/ and web_model_simple/")
