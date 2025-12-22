#!/usr/bin/env python3
"""
Convert Keras model to TFJS format for browser use.
This script:
1. Loads the improved_static_5feat.keras model
2. Converts it to SavedModel format
3. Exports as TFJS graph model
"""
import os
import sys
import shutil
from pathlib import Path

try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow not installed. Installing...")
    os.system(f"{sys.executable} -m pip install tensorflow")
    import tensorflow as tf

def convert_keras_to_tfjs(keras_path, output_dir):
    """Convert Keras model to TFJS format"""
    print(f"\n{'='*60}")
    print(f"Converting: {keras_path}")
    print(f"Output Dir: {output_dir}")
    print(f"{'='*60}\n")
    
    # Step 1: Load Keras model
    print("Step 1: Loading Keras model...")
    try:
        model = tf.keras.models.load_model(keras_path)
        print(f"✓ Model loaded successfully")
        print(f"  - Input shapes: {[inp.shape for inp in model.inputs]}")
        print(f"  - Output shapes: {[out.shape for out in model.outputs]}")
        print(f"  - Total params: {model.count_params():,}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Step 2: Convert to SavedModel format (intermediate step)
    print("\nStep 2: Converting to SavedModel format...")
    saved_model_path = os.path.join(output_dir, "temp_savedmodel")
    try:
        # Clear if exists
        if os.path.exists(saved_model_path):
            shutil.rmtree(saved_model_path)
        
        # Save as SavedModel
        model.export(saved_model_path)
        print(f"✓ SavedModel exported to: {saved_model_path}")
    except Exception as e:
        print(f"✗ Failed to export SavedModel: {e}")
        return False
    
    # Step 3: Convert SavedModel to TFJS
    print("\nStep 3: Converting SavedModel to TFJS format...")
    try:
        import subprocess
        
        # Try using tensorflowjs converter CLI
        cmd = [
            sys.executable,
            "-m", "tensorflowjs.converters.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            saved_model_path,
            output_dir
        ]
        
        print(f"  Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError("Conversion failed")
        
        print(result.stdout)
        print(f"✓ TFJS conversion completed")
    except Exception as e:
        print(f"✗ Failed to convert to TFJS: {e}")
        print(f"  Trying alternative method...")
        return False
    
    # Step 4: Verify output
    print("\nStep 4: Verifying output...")
    model_json = os.path.join(output_dir, "model.json")
    if os.path.exists(model_json):
        print(f"✓ model.json exists: {os.path.getsize(model_json)} bytes")
        
        # Check for weight files
        weight_files = [f for f in os.listdir(output_dir) if f.endswith('.bin')]
        print(f"✓ Weight files found: {len(weight_files)}")
        for wf in weight_files[:3]:  # Show first 3
            size_mb = os.path.getsize(os.path.join(output_dir, wf)) / 1024 / 1024
            print(f"  - {wf}: {size_mb:.2f} MB")
        
        print(f"\n✓ Conversion successful!")
        return True
    else:
        print(f"✗ model.json not found in {output_dir}")
        return False

def main():
    """Main conversion workflow"""
    print("KERAS TO TFJS CONVERSION TOOL")
    print("="*60)
    
    # Paths
    keras_model = "Nikhil_ML/improved_static_5feat.keras"
    output_dir = "web_model_improved_static"
    
    # Check inputs
    if not os.path.exists(keras_model):
        print(f"✗ Keras model not found: {keras_model}")
        return False
    
    print(f"\n✓ Keras model found: {keras_model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")
    
    # Convert
    success = convert_keras_to_tfjs(keras_model, output_dir)
    
    # Cleanup temp
    temp_dir = os.path.join(output_dir, "temp_savedmodel")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up temporary files")
    
    if success:
        print(f"\n{'='*60}")
        print(f"✓ CONVERSION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Your new TFJS model is ready in: {output_dir}/")
        print(f"2. To use it in your app:")
        print(f"   - Replace 'web_model_static' with '{output_dir}' in ml-heuristic.js")
        print(f"   - Or copy {output_dir}/* to web_model_static/")
        print(f"3. Test it by opening test-model-simple.html in your browser")
        return True
    else:
        print(f"\n✗ Conversion failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
