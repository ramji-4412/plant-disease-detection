import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import json

try:
    with open('temp_model/config.json', 'r') as f:
        config = json.load(f)
    m = tf.keras.models.model_from_config(config)
    m.load_weights('temp_model/model.weights.h5')
    print("Loaded successfully, type:", type(m))
    print("Has predict:", hasattr(m, 'predict'))
except Exception as e:
    print("Error:", e)