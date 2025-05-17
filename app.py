import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model
model = load_model('model.h5')

# Label mapping
labels = ['Cyst', 'Stone', 'Tumor', 'Normal']

@app.route('/')
def index():
    return 'Kidney CT Scan Classifier is Running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        img = Image.open(request.files['image']).convert('RGB')
        img = img.resize((352, 233))  # Resize to match model expectation
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, -1)  # Flatten to (1, 246016)

        prediction = model.predict(img_array)
        predicted_class = labels[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
