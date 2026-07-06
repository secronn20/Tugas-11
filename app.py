import os
import time
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
HISTORY_FILE = 'history.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit (keeps within Werkzeug's 10MB read buffer)

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable for lazy loading
model = None
MODEL_PATH = os.path.join('model', 'road_crack_vgg16.h5')

def get_model():
    """Lazy load the TensorFlow model to prevent startup crashes if not trained yet."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading trained VGG16 model from {MODEL_PATH}...")
            # Load model (can be slow on first load)
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"WARNING: Model file not found at {MODEL_PATH}. Prediction features will be unavailable.")
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_history():
    """Load prediction history from the JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading history file: {e}")
            return []
    return []

def save_history(history_data):
    """Save prediction history to the JSON file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_data, f, indent=4)
    except Exception as e:
        print(f"Error saving history file: {e}")

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({
        'success': False,
        'message': 'Ukuran file terlalu besar. Maksimum ukuran file adalah 5 MB.'
    }), 413

@app.route('/')
def index():
    # Check if model exists to display status to user
    model_loaded = os.path.exists(MODEL_PATH)
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if model is trained
    trained_model = get_model()
    if trained_model is None:
        return jsonify({
            'success': False,
            'message': 'Model VGG16 belum selesai dilatih atau tidak ditemukan. Selesaikan pelatihan terlebih dahulu!'
        }), 500

    # 2. Check if file is in request
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'Tidak ada file gambar yang dikirim.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Nama file kosong.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Format file tidak didukung. Unggah gambar JPG, JPEG, atau PNG.'}), 400

    try:
        # Save image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Start timer for performance analysis
        start_time = time.time()

        # Preprocess image
        # Target size for VGG16 is 224x224
        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale to match training data preprocessing

        # Predict
        prediction = trained_model.predict(img_array)
        prob = float(prediction[0][0])
        
        processing_time = round(time.time() - start_time, 4)

        # Classification rules:
        # Class 0: Negative (Tidak Retak)
        # Class 1: Positive (Retak)
        if prob >= 0.5:
            result = "Retak"
            confidence = prob * 100
        else:
            result = "Tidak Retak"
            confidence = (1.0 - prob) * 100

        # Save to history
        record = {
            'id': timestamp + "_" + str(np.random.randint(100, 999)),
            'filename': unique_filename,
            'filepath': filepath.replace('\\', '/'),  # normalize for url
            'result': result,
            'confidence': round(confidence, 2),
            'processing_time': processing_time,
            'timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        }

        history = load_history()
        history.insert(0, record)  # Add to top
        save_history(history)

        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'processing_time': f"{processing_time} detik",
            'filepath': url_for('static', filename=f'uploads/{unique_filename}'),
            'timestamp': record['timestamp']
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'success': False, 'message': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500

@app.route('/history')
def history():
    records = load_history()
    return render_template('history.html', records=records)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    # Clear records
    save_history([])
    
    # Optional: clean upload directory except placeholder files if any
    for f in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
