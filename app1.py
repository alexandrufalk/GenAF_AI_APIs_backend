from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model_dir/efficientnetb0_fine_tuned_101_classes_mixed_precision')  # Update path accordingly
class_names = ['class1', 'class2', ..., 'class101']  # Add actual class names here

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image
        image = Image.open(io.BytesIO(file.read())).resize((224, 224))  # EfficientNetB0 expects 224x224 input size
        image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions[0])]

        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
