from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import PIL
from PIL import Image
import numpy as np
import io
import base64
from flask_cors import CORS
from food_list import FOOD_LIST

import keras
from keras.utils import img_to_array


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

app = Flask(__name__)
CORS(app)

# Load the pre-trained Keras model
model = tf.keras.models.load_model(r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\food_vision.keras')

def depth_to_space(x, block_size):
    return tf.nn.depth_to_space(x, block_size=block_size)

model_resolution=tf.keras.models.load_model(r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\image_high_resolution_reconstruction.keras',custom_objects={'depth_to_space': depth_to_space})
model_resolution2=tf.keras.models.load_model(r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\image_high_resolution_reconstruction2.keras',custom_objects={'depth_to_space': depth_to_space})
srgan_generator_model_resolution=tf.keras.models.load_model(r'D:\Projects\GenAF_AI_APIs\genaf_ai_apis_backend\models\SRGAN_generator.keras')
#Load object detection model
detector = hub.load("https://kaggle.com/models/tensorflow/efficientdet/tensorflow2/lite2-detection/1")

#load mobilenet_v2 model

#detector_mnet=hub.load("https://kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1")
detector_mnet= hub.load("https://kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1")



print("Signatures",detector_mnet.signatures)
signature_mnet = detector_mnet.signatures['default']
print("signature_mnet:",signature_mnet)

def preprocess_image(image):
    # Open and preprocess the image
    image = Image.open(image).convert('RGB')
    image = np.array(image)  # Convert to NumPy array
    image = tf.convert_to_tensor(image, dtype=tf.uint8)  # Convert to tf.uint8
    image = tf.image.resize(image, (512, 512))  # Resize to (512, 512)
    image = image[tf.newaxis, :]  # Add batch dimension
    return image

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )

def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img

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

# Define a route for image high resolution
@app.route('/resolution', methods=['POST'])
def resolution():
    # Ensure an image file is in the request
    upscale_factor = 3
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
   

    # Preprocess the image
    lowres_input = get_lowres_image(image, upscale_factor)
    

    # Perform the prediction
    prediction = upscale_image(model_resolution, lowres_input)

    # Convert the prediction to a format that can be returned (base64)
    buffered = io.BytesIO()
    prediction.save(buffered, format="JPEG")  # Or PNG depending on your needs
    prediction_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    

    # Return the base64-encoded image as a JSON response
    return jsonify({'prediction': prediction_base64})

@app.route('/resolution2', methods=['POST'])
def resolution2():
    # Ensure an image file is in the request
    upscale_factor = 3
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
   

    # Preprocess the image
    lowres_input = get_lowres_image(image, upscale_factor)
    

    # Perform the prediction
    prediction = upscale_image(model_resolution2, lowres_input)

    # Convert the prediction to a format that can be returned (base64)
    buffered = io.BytesIO()
    prediction.save(buffered, format="JPEG")  # Or PNG depending on your needs
    prediction_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    

    # Return the base64-encoded image as a JSON response
    return jsonify({'prediction': prediction_base64})

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
        print("detector_output", detector_output)

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
        print("Try detect_mnet")

        file=Image.open(file)
        file=file.convert("RGB")
        
        # Preprocess image to match model input
        image_resized=file.resize((512,512))
        image_array=np.array(image_resized)/255.0
        image_tensor=np.expand_dims(image_array,axis=0)

        #print("Image-tensor typr", image_tensor.dtype)  
        
        # Ensure the tensor is of type tf.uint8 and has the correct shape
        if image_tensor.dtype != tf.float32:
            image_tensor = tf.cast(image_tensor, dtype=tf.float32)
            #print("Image-tensor cast", image_tensor)  

        

        

        #print("Image-tensor", image_tensor)  
        
        # Apply the model
        detector_output = signature_mnet(image_tensor)
        print("detector_output", detector_output)

         # Extract detection outputs
        detection_boxes = detector_output["detection_boxes"].numpy().tolist()
        detection_class_labels = detector_output["detection_class_labels"].numpy().tolist()
        detection_class_names = detector_output["detection_class_names"].numpy().tolist()
        print("detection_class_names",detection_class_names)
        detection_class_entities = detector_output["detection_class_entities"].numpy().tolist()
        detection_scores = detector_output["detection_scores"].numpy().tolist()

        # Convert byte strings to normal strings if needed
        detection_class_names = [cls.decode('utf-8') if isinstance(cls, bytes) else cls for cls in detection_class_names]
        detection_class_entities = [ent.decode('utf-8') if isinstance(ent, bytes) else ent for ent in detection_class_entities]

        

        # Filter the results to only include detections with scores > 0.51
        filtered_boxes = []
        filtered_class_names = []
        filtered_scores = []
        filtered_labels = []
        filtered_entities=[]

        print("lenght od detection_score",len(detection_scores))

        for i in range(len(detection_scores)):
            
            if detection_scores[i] > 0.3:  # Check if the score is greater than 0.51
                
                filtered_boxes.append(detection_boxes[i])
                
                filtered_class_names.append(detection_class_names[i])
                filtered_scores.append(detection_scores[i])
                filtered_labels.append(detection_class_labels[i])
                filtered_entities.append(detection_class_entities[i])

        # Count the number of filtered detections
        num_detections = len(filtered_class_names)

        # Create the result dictionary
        result = {
            'num_detections': num_detections,
            'detection_boxes': filtered_boxes,
            'detection_classes': filtered_class_names,
            'detection_scores': filtered_scores,
            'detection_labels': filtered_labels,
            'detection_entries': filtered_entities
        }

        print("result:",result)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    
    


if __name__ == '__main__':
    app.run(debug=True)
