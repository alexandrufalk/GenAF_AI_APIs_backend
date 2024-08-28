from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the pre-trained Keras model
model = tf.keras.models.load_model(r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\food_vision.keras')

# Define a route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    # Convert the image file to a PIL image
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the image to match the input format of the model
    image = image.resize((224, 224))  # resize to match model input shape
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # add batch dimension

    # Perform the prediction
    prediction = model.predict(image)

    # Convert the prediction to a list and return as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
