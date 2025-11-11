import numpy as np
import random
import math
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import time
layers = keras.layers
Input = keras.Input
Model = keras.Model

GRID_SIZE = 20 #if this is changed it needs to be changed in actual pathfinder as well, check if a 30x30 grid is better
OBSTACLE_PROB = 0.5 #increase to make more difficult training
NUM_SAMPLES = 2000  # increased for better generalization
MAX_ATTEMPTS = NUM_SAMPLES * 25 

def generate_random_grid():
    grid = np.random.choice([0,1], size=(GRID_SIZE, GRID_SIZE), p=[1-OBSTACLE_PROB, OBSTACLE_PROB])
    return grid

def generate_maze_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    #vertical walls
    for col in range(2, GRID_SIZE-2, 4):
        for row in range(GRID_SIZE):
            if random.random() < 0.85:
                grid[row, col] = 1

    for row in range(2, GRID_SIZE-2, 4):
        for col in range(GRID_SIZE):
            if random.random() < 0.85:
                grid[row, col] = 1
    # randomly clear some cells to ensure that most wall cells are connected
    for _ in range(GRID_SIZE * 2):
        grid[random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)] = 0
    return grid

def generate_obstacle_grid():
    # random areas with high density of obstacles
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    num_blocks = random.randint(3, 7)
    for _ in range(num_blocks):
        r, c = random.randint(0, GRID_SIZE-5), random.randint(0, GRID_SIZE-5)
        h, w = random.randint(2, 5), random.randint(2, 5)
        grid[r:r+h, c:c+w] = 1
    return grid

def generate_mixed_grid():
    #choose one of the grid types to train with
    choice = random.random()
    if choice < 0.4:
        return generate_maze_grid()
    elif choice < 0.7:
        return generate_obstacle_grid()
    else:
        return generate_random_grid()

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
    def heuristic(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        D = 1
        D2 = math.sqrt(2)
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
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


def flatten_grid(grid):
    return grid.flatten().tolist()

grid_samples, start_goal_samples, residual_targets, distances = [], [], [], []
print("generating training data with distance-aware sampling")
start_time = time.time()
attempts = 0

# Bucket samples by distance to ensure we get diverse distance ranges
distance_buckets = {i: [] for i in range(1, 27)}  # distances 1-26

while attempts < MAX_ATTEMPTS:
    grid = generate_mixed_grid()
    start = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    goal = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    attempts += 1
    if grid[start] == 1 or grid[goal] == 1 or start == goal:
        continue
    cost = a_star(grid, start, goal)
    if cost is not None:
        base_h = octile_distance(start, goal)
        dist = int(base_h)  # bucket by octile distance
        
        # Accept ALL costs to get full range
        if 1 <= dist <= 26:
            # Add 4 normalized features: start_r, start_c, goal_r, goal_c (no distance feature)
            distance_buckets[dist].append({
                'grid': grid.reshape((GRID_SIZE, GRID_SIZE, 1)),
                'sg': [start[0]/(GRID_SIZE-1), start[1]/(GRID_SIZE-1),
                       goal[0]/(GRID_SIZE-1), goal[1]/(GRID_SIZE-1)],  # 4 features only
                'cost': cost,  # Changed from residual to cost
                'distance': base_h
            })

# Balance samples across distance buckets, taking up to 100 from each
for dist in sorted(distance_buckets.keys()):
    samples_to_take = min(100, len(distance_buckets[dist]))
    for sample in distance_buckets[dist][:samples_to_take]:
        grid_samples.append(sample['grid'])
        start_goal_samples.append(sample['sg'])
        residual_targets.append(sample['cost'])  # Now storing cost, not residual
        distances.append(sample['distance'])
    if len(grid_samples) >= NUM_SAMPLES:
        break

if len(residual_targets) > NUM_SAMPLES:
    # Trim to exact NUM_SAMPLES
    grid_samples = grid_samples[:NUM_SAMPLES]
    start_goal_samples = start_goal_samples[:NUM_SAMPLES]
    residual_targets = residual_targets[:NUM_SAMPLES]
    distances = distances[:NUM_SAMPLES]

print(f"Generated {len(residual_targets)} samples across {len([d for d in distances if d > 0])} distance values")
print(f"Distance range: {min(distances):.1f} - {max(distances):.1f}")
print(f"Attempts: {attempts}")
if len(residual_targets) < NUM_SAMPLES:
    print(f"WARNING: Only {len(residual_targets)} samples generated after {attempts} attempts. Consider lowering OBSTACLE_PROB or increasing MAX_ATTEMPTS.")
end_time = time.time()
print(f"Data generation completed in {end_time-start_time:.2f} seconds.")

grid_samples = np.array(grid_samples, dtype=np.float32)
start_goal_samples = np.array(start_goal_samples, dtype=np.float32)
residual_targets = np.array(residual_targets, dtype=np.float32)

# Normalize cost targets to [0, 1] range for better training
# Cost ranges from ~1 to ~27, so normalize by max expected cost
max_cost = math.sqrt(2) * (GRID_SIZE - 1)  # ~26.87
residual_targets = residual_targets / max_cost

plt.hist(residual_targets, bins=50)
plt.title('Cost Distribution (normalized by max possible cost)')
plt.xlabel('Normalized Cost')
plt.ylabel('Count')
plt.savefig('residual_distribution.png')
plt.close()
print("Distribution plot saved to residual_distribution.png")

print(f"Target cost stats: min={residual_targets.min():.3f}, max={residual_targets.max():.3f}, mean={residual_targets.mean():.3f}, std={residual_targets.std():.3f}")
print("Sample grid shape:", grid_samples[0].shape, "Sample start/goal:", start_goal_samples[0])


train_grids, val_grids, train_start_goals, val_start_goals, train_targets, val_targets = train_test_split(
    grid_samples, start_goal_samples, residual_targets, test_size=0.1, random_state=42)

def build_model():
    grid_input = keras.Input(shape=(GRID_SIZE, GRID_SIZE, 1), name="grid")
    start_goal_input = keras.Input(shape=(4,), name="start_goal")  # 4 features: start_r, start_c, goal_r, goal_c
    
    # Grid processing branch - extract spatial features from walls
    grid_branch = layers.Conv2D(16, (3,3), activation='relu', padding='same')(grid_input)
    grid_branch = layers.Conv2D(8, (3,3), activation='relu', padding='same')(grid_branch)
    grid_branch = layers.Flatten()(grid_branch)
    grid_branch = layers.Dense(128, activation='relu')(grid_branch)
    
    # Start/goal processing branch - emphasize distance information
    sg_branch = layers.Dense(32, activation='relu')(start_goal_input)
    sg_branch = layers.Dense(32, activation='relu')(sg_branch)
    sg_branch = layers.Dense(16, activation='relu')(sg_branch)
    
    # Combine both branches
    x = layers.Concatenate()([grid_branch, sg_branch])
    x = layers.Dense(96, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(48, activation='relu')(x)
    # Add slight positive bias to output layer to counteract underestimation tendency
    output = layers.Dense(1, activation='linear', bias_initializer=keras.initializers.Constant(0.1))(x)
    
    model = keras.Model(inputs=[grid_input, start_goal_input], outputs=output)
    return model

model = build_model()

# Simple approach: use MSE but with initial bias to help with calibration
model.compile(optimizer='adam', loss='mse')

cb = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
]
model.fit([train_grids, train_start_goals], train_targets, epochs=100, batch_size=32, validation_data=([val_grids, val_start_goals], val_targets), callbacks=cb, verbose=2)


print("\nSkipping hard case mining for faster deployment...")
# Hard mining commented out to speed up training
# The initial training with 1000 samples should be sufficient for basic testing


model.export("ml_heuristic_savedmodel_static")
print("Model exported as TensorFlow SavedModel (ml_heuristic_savedmodel/)")

# converting to tensorflow:
# docker run --rm -v "${PWD}:/workspace" tfjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ml_heuristic_savedmodel_static web_model_static


