from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import io
from flask_cors import CORS
from food_list import FOOD_LIST



app = Flask(__name__)
CORS(app)

# Load the pre-trained Keras model
model = tf.keras.models.load_model(r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\food_vision.keras')

#Load object detection model
detector = hub.load("https://kaggle.com/models/tensorflow/efficientdet/frameworks/TensorFlow2/variations/d0/versions/1")

#load mobilenet_v2 model

detector_mnet=hub.load("https://kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1")

def preprocess_image(image):
    # Open and preprocess the image
    image = Image.open(image).convert('RGB')
    image = np.array(image)  # Convert to NumPy array
    image = tf.convert_to_tensor(image, dtype=tf.uint8)  # Convert to tf.uint8
    image = tf.image.resize(image, (512, 512))  # Resize to (512, 512)
    image = image[tf.newaxis, :]  # Add batch dimension
    return image


# Define a route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    image = image.resize((224, 224))  # Resize to match model input shape
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform the prediction
    prediction = model.predict(image)
    squeeze_result = np.squeeze(prediction)

    # Find the label with the highest prediction score
    max_index = squeeze_result.argmax(axis=0)
    label_pred = FOOD_LIST[max_index]

    print("Predicted label:", label_pred)

    # Convert the prediction to JSON and return
    return jsonify({'prediction': label_pred})

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image
        image_tensor = preprocess_image(file)
        
        # Ensure the tensor is of type tf.uint8 and has the correct shape
        if image_tensor.dtype != tf.uint8:
            image_tensor = tf.cast(image_tensor, dtype=tf.uint8)
        
        # Apply the model
        detector_output = detector(image_tensor)

        # Extract and convert model outputs to lists
        num_detections = int(detector_output["num_detections"].numpy())
        detection_boxes = detector_output["detection_boxes"].numpy().tolist()
        detection_classes = detector_output["detection_classes"].numpy().tolist()
        detection_scores = detector_output["detection_scores"].numpy().tolist()

        # Print detection boxes for debugging
        print("Detection Boxes:")
        for box in detection_boxes:
            print(box)
        
        print("Classes:", detection_classes)
        
        # Optionally include more detailed output
        raw_detection_boxes = detector_output["raw_detection_boxes"].numpy().tolist()
        raw_detection_scores = detector_output["raw_detection_scores"].numpy().tolist()
        detection_anchor_indices = detector_output["detection_anchor_indices"].numpy().tolist()
        detection_multiclass_scores = detector_output["detection_multiclass_scores"].numpy().tolist()

        result = {
            'num_detections': num_detections,
            'detection_boxes': detection_boxes,
            'detection_classes': detection_classes,
            'detection_scores': detection_scores,
            # 'raw_detection_boxes': raw_detection_boxes,
            # 'raw_detection_scores': raw_detection_scores,
            # 'detection_anchor_indices': detection_anchor_indices,
            # 'detection_multiclass_scores': detection_multiclass_scores
        }

        
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/detect_mnet', methods=['POST'])
def detect_mnet():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image
        image_tensor = preprocess_image(file)
        
        # Ensure the tensor is of type tf.uint8 and has the correct shape
        if image_tensor.dtype != tf.float32:
            image_tensor = tf.cast(image_tensor, dtype=tf.float32)

        # Scale the values from [0, 255] to [0.0, 1.0]
        image_tensor /= 255.0    
        
        # Apply the model
        detector_output = detector_mnet(image_tensor)

        # Extract and convert model outputs to lists
        num_detections = int(detector_output["detection_class_labels"].numpy())
        detection_boxes = detector_output["detection_boxes"].numpy().tolist()
        detection_classes = detector_output["detection_class_name"].numpy().tolist()
        detection_scores = detector_output["detection_scores"].numpy().tolist()

        # Print detection boxes for debugging
        print("Detection Boxes:")
        for box in detection_boxes:
            print(box)
        
        print("Classes:", detection_classes)
        

        result = {
            'num_detections': num_detections,
            'detection_boxes': detection_boxes,
            'detection_classes': detection_classes,
            'detection_scores': detection_scores,
            # 'raw_detection_boxes': raw_detection_boxes,
            # 'raw_detection_scores': raw_detection_scores,
            # 'detection_anchor_indices': detection_anchor_indices,
            # 'detection_multiclass_scores': detection_multiclass_scores
        }

        
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
