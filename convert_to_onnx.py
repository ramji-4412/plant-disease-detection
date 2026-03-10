import tensorflow as tf
import tf2onnx
import onnxruntime as ort

# Load the Keras model
model = tf.keras.models.load_model('plant_disease_model.keras')

# Convert to ONNX directly from Keras
output_path = "model.onnx"
try:
    tf2onnx.convert.from_keras(model, output_path=output_path)
    print("Model converted to ONNX successfully!")
except Exception as e:
    print(f"Conversion failed: {e}")

# Test the ONNX model
try:
    session = ort.InferenceSession(output_path)
    print("ONNX model loaded successfully!")
except Exception as e:
    print(f"ONNX load failed: {e}")