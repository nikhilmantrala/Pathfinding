#!/usr/bin/env python3
"""
Convert improved admissible Keras model to SavedModel format.
Handles custom loss functions properly.
"""
import os
import sys
import shutil

try:
    import tensorflow as tf
    import numpy as np
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow not installed in this Python environment")
    sys.exit(1)

# Define custom loss function for loading
def admissible_heuristic_loss(y_true, y_pred):
    """
    Custom loss that penalizes overestimation heavily.
    h must be admissible (never overestimate true cost).
    """
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    overestimation = tf.maximum(0.0, y_pred - y_true)
    overestimation_penalty = 10.0 * overestimation
    return mse + overestimation_penalty

def keras_to_savedmodel(keras_path, output_path):
    """Convert Keras h5/keras model to SavedModel format"""
    print(f"\nLoading Keras model: {keras_path}")
    
    try:
        # Load with custom objects
        model = tf.keras.models.load_model(
            keras_path,
            custom_objects={'admissible_heuristic_loss': admissible_heuristic_loss}
        )
        print(f"✓ Model loaded successfully")
        print(f"  Input shapes: {[inp.shape for inp in model.inputs]}")
        print(f"  Output shapes: {[out.shape for out in model.outputs]}")
        print(f"  Total params: {model.count_params():,}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    print(f"\nConverting to SavedModel: {output_path}")
    try:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        model.export(output_path)
        print(f"✓ SavedModel exported successfully")
        
        # Verify SavedModel
        print(f"\nVerifying SavedModel...")
        loaded = tf.saved_model.load(output_path)
        print(f"✓ SavedModel verified and can be loaded")
        
        return True
    except Exception as e:
        print(f"✗ Failed to export SavedModel: {e}")
        return False

if __name__ == "__main__":
    keras_file = "improved_admissible_static_best.keras"
    savedmodel_dir = "temp_savedmodel_admissible"
    
    if not os.path.exists(keras_file):
        print(f"✗ Keras file not found: {keras_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("KERAS TO SAVEDMODEL CONVERTER")
    print("=" * 60)
    
    if keras_to_savedmodel(keras_file, savedmodel_dir):
        print(f"\n{'=' * 60}")
        print(f"✓ CONVERSION SUCCESSFUL")
        print(f"✓ Next step: Docker TFJS conversion")
        print(f"  Command: docker run --rm -v \"${{PWD}}:/workspace\" tfjs_converter \\")
        print(f"    --input_format=tf_saved_model \\")
        print(f"    --output_format=tfjs_graph_model \\")
        print(f"    /workspace/{savedmodel_dir} \\")
        print(f"    /workspace/web_model_improved_static")
        print(f"=" * 60)
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(f"✗ CONVERSION FAILED")
        print(f"=" * 60)
        sys.exit(1)
