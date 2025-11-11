#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

model = tf.saved_model.load("ml_heuristic_savedmodel_static")
concrete_func = model.signatures['serving_default']

print("Concrete function inputs:")
for name, spec in concrete_func.structured_input_signature[1].items():
    print(f"  {name}: {spec}")

print("\nConcrete function outputs:")
print(f"  Output type: {type(concrete_func.structured_outputs)}")
result = concrete_func(grid=tf.constant(0.0, shape=[1,20,20,1]), start_goal=tf.constant(0.0, shape=[1,5]))
print(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
print(f"  Result: {result}")
