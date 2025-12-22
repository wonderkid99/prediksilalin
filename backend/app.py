import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keras.models import load_model

# --- KONFIGURASI PATH ---
# Mendapatkan lokasi file app.py saat ini
base_dir = os.path.dirname(os.path.abspath(__file__))

# Menentukan lokasi folder frontend (naik satu level dari backend, lalu masuk frontend)
template_dir = os.path.join(base_dir, '../frontend')
static_dir = os.path.join(base_dir, '../frontend')

# Inisialisasi Flask dengan folder kustom
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# --- LOAD MODEL ---
# Path ke models/model_lalin.h5
model_path = os.path.join(base_dir, 'models', 'model_lalin.h5')

# Cek apakah model ada sebelum di-load
if os.path.exists(model_path):
    print(f"Loading model dari: {model_path}")
    model = load_model(model_path)
else:
    print(f"ERROR: Model tidak ditemukan di {model_path}")
    # Kita tidak exit, agar server tetap jalan meski error (untuk debugging)
    model = None

# --- ROUTE ---

# 1. Halaman Utama
@app.route('/')
def home():
    # Ini akan mencari index.html di folder ../frontend
    return render_template('index.html')

# 2. API Prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum dimuat'}), 500

    try:
        data = request.json
        
        # Ambil input dan pastikan tipe datanya float
        input_data = np.array([[
            float(data['speed']), 
            float(data['rain']), 
            int(data['holiday'])
        ]])
        
        # Scaling Manual (Sesuai kode Anda)
        # [Max Speed, Max Rain, Max Holiday]
        input_data_scaled = input_data / [80.0, 50.0, 1.0]
        
        # Reshape untuk LSTM (Duplikasi jadi 12 timestep)
        final_input = np.repeat(input_data_scaled[np.newaxis, :, :], 12, axis=1)

        # Prediksi
        prediction = model.predict(final_input)
        
        # Denormalisasi
        hasil_kmh = prediction[0][0] * 80.0
        
        return jsonify({'prediksi': round(float(hasil_kmh), 2)})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)