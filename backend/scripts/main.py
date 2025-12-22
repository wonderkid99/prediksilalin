import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Import Library Keras (Gunakan style ini agar kompatibel dengan Keras 3.x)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ==========================================
# KONFIGURASI
# ==========================================
FILENAME = 'traffic_simulation_data.csv'
LOOK_BACK = 12       # Melihat 3 jam ke belakang (12 x 15 menit)
EPOCHS = 10          # Jumlah putaran latihan
BATCH_SIZE = 32

# ==========================================
# 1. LOAD DATA DARI CSV
# ==========================================
print(f"--- [1] Membaca File {FILENAME} ---")

# Cek apakah file ada
if not os.path.exists(FILENAME):
    print(f"ERROR: File {FILENAME} tidak ditemukan!")
    print("Silakan jalankan 'data_generator.py' terlebih dahulu.")
    exit()

df = pd.read_csv(FILENAME)

# Tampilkan info data untuk memastikan terbaca dengan benar
print(f"Data Loaded: {len(df)} baris.")
print(df.head())

# ==========================================
# 2. PREPROCESSING
# ==========================================
print("\n--- [2] Preprocessing Data ---")

# Pilih Fitur yang sesuai dengan nama kolom di CSV generator Anda:
# Target: avg_speed_kmh
# Fitur Pendukung: precipitation_mm, is_holiday
features = df[['avg_speed_kmh', 'precipitation_mm', 'is_holiday']].values

# Normalisasi Data (Scaling ke 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

def create_dataset(data, look_back=12):
    X, Y = [], []
    for i in range(len(data) - look_back):
        # Ambil 12 baris ke belakang (Semua fitur)
        X.append(data[i:(i + look_back), :])
        # Ambil 1 data ke depan (Hanya kolom ke-0 yaitu avg_speed_kmh)
        Y.append(data[i + look_back, 0]) 
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_features, LOOK_BACK)

# Split Training (80%) & Testing (20%)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Shape Training Input: {X_train.shape}")
print(f"Shape Testing Input: {X_test.shape}")

# ==========================================
# 3. BUILD & TRAIN MODEL
# ==========================================
print("\n--- [3] Membangun & Melatih Model LSTM ---")
model = Sequential()
# Input Shape: (TimeSteps, Features) -> (12, 3)
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1)) # Output prediksi 1 nilai (Speed)

model.compile(optimizer='adam', loss='mean_squared_error')

# Mulai Training
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)

# ==========================================
# 4. PREDIKSI & VISUALISASI
# ==========================================
print("\n--- [4] Evaluasi & Visualisasi ---")
test_predict = model.predict(X_test)

# Denormalisasi (Kembalikan ke satuan km/jam asli)
def inverse_transform_speed(pred_array, scaler):
    # Kita butuh array dummy dengan 3 kolom karena scaler dilatih dengan 3 kolom
    dummy = np.zeros((len(pred_array), 3)) 
    dummy[:, 0] = pred_array.flatten() # Masukkan prediksi ke kolom pertama (avg_speed_kmh)
    return scaler.inverse_transform(dummy)[:, 0]

y_test_inv = inverse_transform_speed(y_test, scaler)
test_predict_inv = inverse_transform_speed(test_predict, scaler)

# Plotting Grafik
plt.figure(figsize=(12, 6))
subset = 300 # Ambil 300 data terakhir agar grafik terlihat jelas
plt.plot(y_test_inv[:subset], label='Kecepatan Asli (Real)', color='blue', alpha=0.6)
plt.plot(test_predict_inv[:subset], label='Prediksi AI (Forecast)', color='red', linestyle='--')

plt.title('Prediksi Kecepatan Lalu Lintas (Input CSV)')
plt.xlabel('Waktu (Interval 15 Menit)')
plt.ylabel('Kecepatan (km/jam)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Simpan hasil
output_img = 'hasil_prediksi_csv.png'
plt.savefig(output_img)
print(f"Selesai! Grafik disimpan sebagai '{output_img}'")
plt.show()

# Menyimpan model ke file
model.save('model_lalin.h5') 