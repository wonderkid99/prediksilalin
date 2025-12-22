import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_traffic_data(start_date, days, road_name="Jalan Jenderal Sudirman"):
    # 1. Setup Range Waktu (Interval 15 menit)
    end_date = start_date + timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='15T')

    data = []

    for timestamp in date_range:
        hour = timestamp.hour
        weekday = timestamp.weekday() # 0=Senin, 6=Minggu
        is_weekend = weekday >= 5

        # --- LOGIKA SIMULASI ---

        # A. Tentukan Base Speed & Flow berdasarkan Jam (Pola Jakarta)
        # Default: Malam sepi (Lancar), Siang ramai (Padat)
        if 0 <= hour < 5: # Dini hari
            base_speed = 55
            traffic_flow = np.random.randint(50, 200)
        elif 6 <= hour < 10: # Pagi (Berangkat Kerja)
            base_speed = 20 if not is_weekend else 45
            traffic_flow = np.random.randint(1500, 2500) if not is_weekend else np.random.randint(500, 800)
        elif 16 <= hour < 20: # Sore (Pulang Kerja)
            base_speed = 15 if not is_weekend else 40
            traffic_flow = np.random.randint(1800, 3000) if not is_weekend else np.random.randint(600, 900)
        else: # Siang/Malam biasa
            base_speed = 35
            traffic_flow = np.random.randint(800, 1200)

        # B. Faktor Cuaca (Random Hujan 10% dari waktu)
        # 0 = Cerah, 1 = Hujan Ringan, 2 = Hujan Lebat
        weather_prob = np.random.rand()
        if weather_prob > 0.95:
            weather = 2 # Hujan Lebat
            rain_mm = np.random.randint(10, 50)
        elif weather_prob > 0.85:
            weather = 1 # Hujan Ringan
            rain_mm = np.random.randint(1, 10)
        else:
            weather = 0 # Cerah
            rain_mm = 0

        # C. Kalkulasi Kecepatan Akhir (Dipengaruhi Flow & Cuaca)
        # Tambahkan noise random (+- 5 km/h) biar terlihat natural
        noise = np.random.normal(0, 3)

        final_speed = base_speed + noise

        # Jika Hujan, kecepatan turun
        if weather == 1: final_speed -= 5
        if weather == 2: final_speed -= 15

        # Batasan Logis (Speed gak boleh minus atau > 80 di dlm kota)
        final_speed = max(5, min(final_speed, 80))

        # Simpan baris data
        data.append([
            timestamp,
            road_name,
            int(traffic_flow),
            round(final_speed, 2),
            rain_mm,
            1 if is_weekend else 0, # Is Holiday/Weekend
            1 if weather > 0 else 0 # Is Raining
        ])

    # Buat DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'road_name', 'flow_vehicle_per_hour',
        'avg_speed_kmh', 'precipitation_mm', 'is_holiday', 'is_raining'
    ])

    return df

# Generate data untuk 30 hari (1 Bulan)
df_simulasi = generate_traffic_data(datetime(2024, 1, 1), 30)

# Tampilkan sampel data
print("Sampel 10 Baris Pertama (Pagi Hari Senin):")
print(df_simulasi[(df_simulasi['timestamp'].dt.hour >= 6) & (df_simulasi['timestamp'].dt.day == 1)].head(10))

# Simpan ke CSV (Virtual)
df_simulasi.to_csv('traffic_simulation_data.csv', index=False)