from flask import Flask, request, render_template, jsonify
import os
import json
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/disease_model.h5'
TREATMENTS_PATH = 'data/treatments.json'

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load treatments database
try:
    with open(TREATMENTS_PATH, 'r') as f:
        treatments_db = json.load(f)
except Exception as e:
    print(f"Error loading treatments: {e}")
    treatments_db = {}

# Class names - MUST match your model's output classes
CLASS_NAMES = [
    "Rice Blast",
    "Brown Spot", 
    "Bacterial Blight",
    "Tungro Virus",
    "Sheath Blight",
    "False Smut",
    "Healthy",
    "Leaf Scald"
]

# Load model if exists
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print("⚠️ Model not found. Using mock prediction.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Prediction logic
    if model is not None:
        try:
            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_index]
            
            # Debug info
            confidence = float(predictions[0][predicted_index])
            print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_class = "Unknown Disease"
    else:
        # Mock prediction for demo (will be replaced with real model)
        predicted_class = "Rice Blast"

    # Get treatment recommendation
    treatment = treatments_db.get(predicted_class, "Consult your local agricultural officer.")

    return jsonify({
        'disease': predicted_class,
        'treatment': treatment,
        'image_path': f'/static/uploads/{file.filename}'
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)