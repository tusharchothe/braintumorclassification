from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import io
import base64

app = Flask(__name__)

# Load model and label encoder
model = tf.keras.models.load_model('fused_cnn_model.keras')
label_encoder = LabelEncoder()
label_encoder.fit(['Healthy', 'Tumor'])

def predict_image(image):
    img = image.resize((224, 224))
    arr = img_to_array(img) / 255.0
    fused = np.expand_dims((arr + arr) / 2.0, axis=0).astype(np.float32)
    prob = model.predict(fused, verbose=0)[0][0]
    label = label_encoder.inverse_transform([int(prob > 0.5)])[0]
    confidence = prob if label == "Tumor" else (1 - prob)
    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        image = Image.open(file.stream)
        label, confidence = predict_image(image)
        
        return jsonify({
            'prediction': label,
            'confidence': f"{confidence:.1%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)