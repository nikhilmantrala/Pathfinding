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
NUM_SAMPLES = 5000  # Lower for faster testing
MAX_ATTEMPTS = NUM_SAMPLES * 20  # Prevent infinite loop if too many unsolvable grids

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

grid_samples, start_goal_samples, residual_targets = [], [], []
print("generating training data")
start_time = time.time()
attempts = 0
while len(residual_targets) < NUM_SAMPLES and attempts < MAX_ATTEMPTS:
    grid = generate_mixed_grid()
    start = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    goal = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    attempts += 1
    if grid[start] == 1 or grid[goal] == 1 or start == goal:
        continue
    cost = a_star(grid, start, goal)
    if cost is not None:
        base_h = octile_distance(start, goal)
        residual = cost - base_h
        if residual < 4.0:
            continue
        sg_features = [
            start[0]/(GRID_SIZE-1), start[1]/(GRID_SIZE-1),
            goal[0]/(GRID_SIZE-1), goal[1]/(GRID_SIZE-1)
        ]
        grid_samples.append(grid.reshape((GRID_SIZE, GRID_SIZE, 1)))
        start_goal_samples.append(sg_features)
        residual_targets.append(residual)
    if len(residual_targets) % 500 == 0 and len(residual_targets) > 0:
        print(f"Progress: {len(residual_targets)}/{NUM_SAMPLES} samples generated after {attempts} attempts...")
if len(residual_targets) < NUM_SAMPLES:
    print(f"WARNING: Only {len(residual_targets)} samples generated after {attempts} attempts. Consider lowering OBSTACLE_PROB or increasing MAX_ATTEMPTS.")
end_time = time.time()
print(f"Data generation completed in {end_time-start_time:.2f} seconds.")

grid_samples = np.array(grid_samples, dtype=np.float32)
start_goal_samples = np.array(start_goal_samples, dtype=np.float32)
residual_targets = np.array(residual_targets, dtype=np.float32)
print(f"Generated {len(residual_targets)} samples.")

max_cost = math.sqrt(2) * (GRID_SIZE - 1)
residual_targets = residual_targets / max_cost

plt.hist(residual_targets, bins=50)
plt.title('Normalized Residual Distribution (cost - octile) / max_cost')
plt.xlabel('Normalized Residual')
plt.ylabel('Count')
plt.show()

print(f"Target residual stats: min={residual_targets.min():.3f}, max={residual_targets.max():.3f}, mean={residual_targets.mean():.3f}, std={residual_targets.std():.3f}")
print("Sample grid shape:", grid_samples[0].shape, "Sample start/goal:", start_goal_samples[0])


train_grids, val_grids, train_start_goals, val_start_goals, train_targets, val_targets = train_test_split(
    grid_samples, start_goal_samples, residual_targets, test_size=0.1, random_state=42)
def build_model():
    grid_input = keras.Input(shape=(GRID_SIZE, GRID_SIZE, 1), name="grid")
    start_goal_input = keras.Input(shape=(4,), name="start_goal")
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(grid_input)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, start_goal_input])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=[grid_input, start_goal_input], outputs=output)
    return model

model = build_model()
model.compile(optimizer='adam', loss='mse')

cb = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5)
]
model.fit([train_grids, train_start_goals], train_targets, epochs=60, batch_size=64, validation_data=([val_grids, val_start_goals], val_targets), callbacks=cb, verbose=2)


print("\nmining hard cases on residual underestimates")
hard_grid_samples, hard_start_goal_samples, hard_residual_targets = [], [], []
for _ in range(20000):
    grid = generate_mixed_grid()
    start = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    goal = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    if grid[start] == 1 or grid[goal] == 1 or start == goal:
        continue
    cost = a_star(grid, start, goal)
    if cost is not None:
        base_h = octile_distance(start, goal)
        residual = cost - base_h
        if residual < 4.0:
            continue
        sg_features = [
            start[0]/(GRID_SIZE-1), start[1]/(GRID_SIZE-1),
            goal[0]/(GRID_SIZE-1), goal[1]/(GRID_SIZE-1)
        ]
        pred = model.predict([grid.reshape((1, GRID_SIZE, GRID_SIZE, 1)), np.array([sg_features])], verbose=0)[0][0] * max_cost
        if pred < (residual * 0.7):
            hard_grid_samples.append(grid.reshape((GRID_SIZE, GRID_SIZE, 1)))
            hard_start_goal_samples.append(sg_features)
            hard_residual_targets.append(residual / max_cost)

print(f"Found {len(hard_grid_samples)} hard mined samples.")
if hard_grid_samples:
    aug_grid_samples = np.concatenate([grid_samples, np.array(hard_grid_samples)])
    aug_start_goal_samples = np.concatenate([start_goal_samples, np.array(hard_start_goal_samples)])
    aug_residual_targets = np.concatenate([residual_targets, np.array(hard_residual_targets)])
    train_grids2, val_grids2, train_start_goals2, val_start_goals2, train_targets2, val_targets2 = train_test_split(
        aug_grid_samples, aug_start_goal_samples, aug_residual_targets, test_size=0.1, random_state=42)
    print("Retraining data...")
    history2 = model.fit([train_grids2, train_start_goals2], train_targets2, epochs=40, batch_size=64, validation_data=([val_grids2, val_start_goals2], val_targets2), callbacks=cb, verbose=2)
    preds2 = model.predict([val_grids2, val_start_goals2])
    print(" MSE after hard mining:", mean_squared_error(val_targets2, preds2))
    # checking if overfitting of data is occuring
    if hasattr(history2, 'history'):
        print("Final training loss:", history2.history['loss'][-1])
        print("Final validation loss:", history2.history['val_loss'][-1])
else:
    print("No hard cases found for augmentation.")


model.export("ml_heuristic_savedmodel_static")
print("Model exported as TensorFlow SavedModel (ml_heuristic_savedmodel/)")

# converting to tensorflow:
# docker run --rm -v "${PWD}:/workspace" tfjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ml_heuristic_savedmodel_static web_model_static


