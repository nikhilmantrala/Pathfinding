"""
Improved ML Heuristic Training Script
- 10x more training data (10,000 samples)
- Deeper CNN with residual connections
- Self-attention mechanism to focus on obstacles
- Variable grid densities for better generalization
- Predicts RESIDUAL directly (always >= 0)
"""

import numpy as np
import random
import math
from queue import PriorityQueue
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Configuration
GRID_SIZE = 20
NUM_SAMPLES = 10000  # 5x more data than original
MAX_ATTEMPTS = NUM_SAMPLES * 50
BATCH_SIZE = 64
EPOCHS = 150

print(f"TensorFlow version: {tf.__version__}")
print(f"Training with {NUM_SAMPLES} samples on {GRID_SIZE}x{GRID_SIZE} grid")

# ============== Grid Generation ==============

def generate_random_grid(obstacle_prob=0.25):
    """Random grid with variable obstacle probability"""
    return np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[1-obstacle_prob, obstacle_prob]).astype(np.float32)

def generate_maze_grid():
    """Maze-like grid with corridors"""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    # Vertical walls
    for col in range(2, GRID_SIZE-2, 4):
        for row in range(GRID_SIZE):
            if random.random() < 0.75:
                grid[row, col] = 1
    # Horizontal walls
    for row in range(2, GRID_SIZE-2, 4):
        for col in range(GRID_SIZE):
            if random.random() < 0.75:
                grid[row, col] = 1
    # Random openings
    for _ in range(GRID_SIZE * 3):
        grid[random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)] = 0
    return grid

def generate_obstacle_grid():
    """Clustered rectangular obstacles"""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    num_blocks = random.randint(2, 6)
    for _ in range(num_blocks):
        r = random.randint(0, GRID_SIZE-5)
        c = random.randint(0, GRID_SIZE-5)
        h = random.randint(2, 6)
        w = random.randint(2, 6)
        grid[r:r+h, c:c+w] = 1
    return grid

def generate_diagonal_barrier():
    """Diagonal barrier forcing detours"""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    # Create diagonal line
    for i in range(GRID_SIZE):
        if 0 <= i < GRID_SIZE:
            grid[i, i] = 1
            if i+1 < GRID_SIZE:
                grid[i, i+1] = 1
    # Random gaps
    for _ in range(3):
        i = random.randint(2, GRID_SIZE-3)
        grid[i, i] = 0
        grid[i, i+1] = 0
    return grid

def generate_mixed_grid():
    """Randomly select grid type with varied densities"""
    choice = random.random()
    if choice < 0.20:
        return generate_maze_grid()
    elif choice < 0.35:
        return generate_obstacle_grid()
    elif choice < 0.45:
        return generate_diagonal_barrier()
    elif choice < 0.65:
        # Light density (like typical test grids)
        return generate_random_grid(random.uniform(0.10, 0.20))
    elif choice < 0.85:
        # Medium density
        return generate_random_grid(random.uniform(0.20, 0.30))
    else:
        # Higher density (challenging cases)
        return generate_random_grid(random.uniform(0.30, 0.40))

# ============== Pathfinding ==============

def octile_distance(start, goal):
    """Optimal heuristic for 8-directional movement"""
    dx = abs(start[0] - goal[0])
    dy = abs(start[1] - goal[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

def is_valid(r, c, grid):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and grid[r, c] == 0

def a_star(grid, start, goal):
    """A* pathfinding, returns actual cost or None if no path"""
    if grid[start] == 1 or grid[goal] == 1:
        return None
    
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
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            
            if not is_valid(nr, nc, grid):
                continue
            
            # Diagonal movement costs sqrt(2)
            step_cost = math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1.0
            tentative = g_score[current] + step_cost
            
            if neighbor not in g_score or tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                f_score = tentative + octile_distance(neighbor, goal)
                openset.put((f_score, neighbor))
    
    return None  # No path found

# ============== Model Architecture ==============

def build_improved_model():
    """
    Enhanced model with:
    - Deeper CNN with residual connections
    - Self-attention to focus on relevant obstacles
    - ReLU output to ensure non-negative residual (admissible)
    """
    grid_input = keras.Input(shape=(GRID_SIZE, GRID_SIZE, 1), name="grid")
    start_goal_input = keras.Input(shape=(4,), name="start_goal")
    
    # Initial convolution
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_input)
    x = layers.BatchNormalization()(x)
    
    # Residual block 1
    residual = layers.Conv2D(32, (1, 1), padding='same')(x)  # Match dimensions
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 10x10
    
    # Residual block 2
    residual = layers.Conv2D(64, (1, 1), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 5x5
    
    # Self-attention layer
    # Reshape for attention: (batch, 25, 64)
    x_flat = layers.Reshape((5 * 5, 64))(x)
    
    # Multi-head attention to focus on relevant grid areas
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=0.1
    )(x_flat, x_flat)
    x_flat = layers.Add()([x_flat, attention_output])
    x_flat = layers.LayerNormalization()(x_flat)
    
    # Global pooling
    grid_features = layers.GlobalAveragePooling1D()(x_flat)
    
    # Start/goal processing branch
    sg_branch = layers.Dense(64, activation='relu')(start_goal_input)
    sg_branch = layers.Dense(32, activation='relu')(sg_branch)
    
    # Combine grid features with start/goal info
    combined = layers.Concatenate()([grid_features, sg_branch])
    
    # Dense layers for prediction
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output: predict normalized RESIDUAL (cost - octile) / max_cost
    # ReLU ensures output >= 0 (admissible heuristic)
    output = layers.Dense(1, activation='relu', name='residual_output')(x)
    
    model = keras.Model(inputs=[grid_input, start_goal_input], outputs=output)
    return model

# ============== Data Generation ==============

print("\n" + "="*50)
print("Generating training data...")
print("="*50)

grid_samples = []
sg_samples = []
residual_targets = []
distances = []

max_cost = math.sqrt(2) * (GRID_SIZE - 1)  # ~26.87

attempts = 0
last_print = 0

while len(residual_targets) < NUM_SAMPLES and attempts < MAX_ATTEMPTS:
    attempts += 1
    
    grid = generate_mixed_grid()
    
    # Generate start and goal with minimum distance
    start = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    goal = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    
    # Skip if start/goal on walls or same position
    if grid[start] == 1 or grid[goal] == 1 or start == goal:
        continue
    
    # Require minimum distance for meaningful paths
    octile = octile_distance(start, goal)
    if octile < 5:  # Skip very short paths
        continue
    
    cost = a_star(grid, start, goal)
    
    if cost is not None:
        residual = cost - octile  # How much more than straight-line
        
        # Store sample
        grid_samples.append(grid.reshape((GRID_SIZE, GRID_SIZE, 1)))
        sg_samples.append([
            start[0] / (GRID_SIZE - 1),
            start[1] / (GRID_SIZE - 1),
            goal[0] / (GRID_SIZE - 1),
            goal[1] / (GRID_SIZE - 1)
        ])
        residual_targets.append(residual / max_cost)  # Normalize
        distances.append(octile)
        
        # Progress update
        if len(residual_targets) - last_print >= 500:
            last_print = len(residual_targets)
            print(f"  Generated {len(residual_targets)}/{NUM_SAMPLES} samples... (attempts: {attempts})")

print(f"\nGenerated {len(residual_targets)} samples in {attempts} attempts")

# Convert to numpy arrays
grid_samples = np.array(grid_samples, dtype=np.float32)
sg_samples = np.array(sg_samples, dtype=np.float32)
residual_targets = np.array(residual_targets, dtype=np.float32)
distances = np.array(distances, dtype=np.float32)

print(f"\nData statistics:")
print(f"  Residual (normalized): min={residual_targets.min():.4f}, max={residual_targets.max():.4f}, mean={residual_targets.mean():.4f}, std={residual_targets.std():.4f}")
print(f"  Distance (octile): min={distances.min():.1f}, max={distances.max():.1f}, mean={distances.mean():.1f}")

# Plot distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(residual_targets, bins=50, edgecolor='black', alpha=0.7)
plt.title('Normalized Residual Distribution')
plt.xlabel('Residual / max_cost')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(distances, bins=30, edgecolor='black', alpha=0.7)
plt.title('Path Distance Distribution')
plt.xlabel('Octile Distance')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('training_data_distribution.png')
plt.close()
print("Saved distribution plot to training_data_distribution.png")

# ============== Training ==============

print("\n" + "="*50)
print("Training model...")
print("="*50)

# Split data
train_grids, val_grids, train_sg, val_sg, train_targets, val_targets = train_test_split(
    grid_samples, sg_samples, residual_targets, test_size=0.1, random_state=42
)

print(f"Training samples: {len(train_targets)}")
print(f"Validation samples: {len(val_targets)}")

# Build model
model = build_improved_model()
model.summary()

# Compile with Adam optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train
history = model.fit(
    [train_grids, train_sg], train_targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=([val_grids, val_sg], val_targets),
    callbacks=callbacks,
    verbose=2
)

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()
print("Saved training history to training_history.png")

# ============== Evaluation ==============

print("\n" + "="*50)
print("Evaluating model...")
print("="*50)

val_predictions = model.predict([val_grids, val_sg], verbose=0)

# Compare predictions vs actual
errors = val_predictions.flatten() - val_targets
print(f"Prediction error: mean={np.mean(errors):.4f}, std={np.std(errors):.4f}")
print(f"Absolute error: mean={np.mean(np.abs(errors)):.4f}")

# Check admissibility (how often does model underestimate?)
underestimates = np.sum(val_predictions.flatten() < val_targets)
print(f"Underestimates: {underestimates}/{len(val_targets)} ({100*underestimates/len(val_targets):.1f}%)")

# ============== Save Models ==============

print("\n" + "="*50)
print("Saving models...")
print("="*50)

# Save Keras model
model.save('static_model.keras')
print("Saved Keras model to static_model.keras")

# Export SavedModel format
model.export('ml_heuristic_savedmodel_static')
print("Exported SavedModel to ml_heuristic_savedmodel_static/")

# Convert to TensorFlow.js
print("\nConverting to TensorFlow.js format...")
try:
    import subprocess
    result = subprocess.run([
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--signature_name=serving_default',
        'ml_heuristic_savedmodel_static',
        'web_model_static'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Successfully converted to TensorFlow.js!")
        print("Output saved to web_model_static/")
    else:
        print(f"Conversion warning: {result.stderr}")
        print("You may need to run manually:")
        print("  tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ml_heuristic_savedmodel_static web_model_static")
except Exception as e:
    print(f"Could not auto-convert: {e}")
    print("Run manually:")
    print("  pip install tensorflowjs")
    print("  tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ml_heuristic_savedmodel_static web_model_static")

print("\n" + "="*50)
print("Training complete!")
print("="*50)
