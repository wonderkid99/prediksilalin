import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'dataseries.csv'
df = pd.read_csv(file_path)

# Data Cleaning
# 1. Convert 'kecepatan' from string "22,58" to float 22.58
# Check if it's string first to avoid errors if pandas already parsed it as float
if df['kecepatan'].dtype == 'O':
    df['kecepatan'] = df['kecepatan'].str.replace(',', '.').astype(float)

# 2. Convert 'periode_data' (e.g., 202402) to datetime
df['date'] = pd.to_datetime(df['periode_data'].astype(str), format='%Y%m')

# Display basic info and first few rows to understand the structure
print("Info Dataset:")
print(df.info())
print("\nSample Data:")
print(df.head())

# Analysis 1: Top 5 Jalan Termacet (Lowest Speed) on average
avg_speed_per_road = df.groupby('ruas_jalan')['kecepatan'].mean().sort_values()
top_5_congested = avg_speed_per_road.head(5)

# Analysis 2: Trend over time for top 3 roads
top_3_roads = avg_speed_per_road.index[:3]
df_top_3 = df[df['ruas_jalan'].isin(top_3_roads)]

# Plotting
plt.figure(figsize=(12, 5))

# Plot 1: Trend Time Series
plt.subplot(1, 2, 1)
sns.lineplot(data=df_top_3, x='date', y='kecepatan', hue='ruas_jalan', marker='o')
plt.title('Trend Kecepatan 3 Jalan Termacet (Time Series)')
plt.xticks(rotation=45)
plt.ylabel('Kecepatan Rata-rata (km/h)')

# Plot 2: Speed Distribution per Region
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='wilayah', y='kecepatan')
plt.title('Distribusi Kecepatan per Wilayah')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('traffic_analysis.png')

print("\nTop 5 Jalan dengan Kecepatan Rata-rata Terendah:")
print(top_5_congested)