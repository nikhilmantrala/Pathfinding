#!/usr/bin/env python3
"""
Convert improved admissible Keras model to SavedModel format.
This is step 1 before Docker conversion to TFJS.
"""
import os
import sys
import shutil

try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow not installed in this Python environment")
    sys.exit(1)

def keras_to_savedmodel(keras_path, output_path):
    """Convert Keras h5/keras model to SavedModel format"""
    print(f"\nLoading Keras model: {keras_path}")
    
    try:
        model = tf.keras.models.load_model(keras_path)
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
    
    print("="*60)
    print("KERAS TO SAVEDMODEL CONVERTER")
    print("="*60)
    
    success = keras_to_savedmodel(keras_file, savedmodel_dir)
    
    if success:
        print(f"\n✓ Ready for Docker conversion!")
        print(f"  SavedModel location: {savedmodel_dir}/")
        print(f"  Next step: Run Docker converter")
    else:
        print(f"\n✗ Conversion failed")
        sys.exit(1)
