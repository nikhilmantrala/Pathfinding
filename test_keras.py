from tensorflow import keras
model = keras.models.load_model("ml_heuristic_model.h5")
model.summary()