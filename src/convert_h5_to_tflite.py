import tensorflow as tf
import numpy as np

# Load the Keras model
model = tf.keras.models.load_model("sin_predictor.h5")

# Define a representative dataset function for quantization
def representative_dataset():
    for _ in range(100):
        data = np.linspace(0, 2*np.pi, 100)
        sin_vals = np.sin(data)
        for i in range(len(sin_vals) - 7):
            sample = sin_vals[i:i+7]
            yield [np.expand_dims(sample.astype(np.float32), axis=0)]

# Converter setup
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save it
with open("sin_predictor_int8.tflite", "wb") as f:
    f.write(tflite_model)
