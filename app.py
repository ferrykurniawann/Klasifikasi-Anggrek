import os
import json
import datetime
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import keras
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

app = Flask(__name__)
app.secret_key = 'rahasia_anggrek_2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
HISTORY_FILE = 'history.json'

CLASS_NAMES = ['Cattleya', 'Dendrobium', 'Grammatophyllum', 'Phalaenopsis', 'Vanda']

model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'MODEL_TERBAIK_Bilateral.keras')
        model = keras.models.load_model(model_path)

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    return idx, CLASS_NAMES[idx], confidence, {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r') as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Format file tidak didukung. Gunakan: png, jpg, jpeg, bmp, tiff')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                load_model()
                idx, label, confidence, all_probs = predict_image(filepath)
                save_history({
                    'filename': filename,
                    'label': label,
                    'confidence': round(confidence, 4),
                    'all_probs': {k: round(v, 4) for k, v in all_probs.items()},
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                return render_template('result.html',
                                     filename=filename,
                                     label=label,
                                     confidence=confidence,
                                     all_probs=all_probs)
            except Exception as e:
                flash(f'Error saat prediksi: {str(e)}')
                return redirect(request.url)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history():
    data = load_history()
    return render_template('history.html', history=data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
