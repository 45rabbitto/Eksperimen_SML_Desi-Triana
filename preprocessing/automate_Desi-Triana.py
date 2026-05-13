import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def main():
    # Load dataset
    df = pd.read_csv('data_raw/diamonds.csv')
    
    # Cek dan hapus missing values
    print("Missing values:", df.isnull().sum().sum())
    df = df.dropna()
    
    # Hapus duplikat
    print("Duplikat dihapus:", df.duplicated().sum())
    df = df.drop_duplicates()
    
    # Hapus hanya outlier ekstrem (persentil 0.5 dan 99.5)
    lower_bound = df['price'].quantile(0.005)
    upper_bound = df['price'].quantile(0.995)
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    
    # Transformasi log pada target untuk mengurangi skewness
    df['price_log'] = np.log1p(df['price'])
    
    # Encoding ordinal untuk cut (Fair ke Ideal: 0-4)
    cut_order = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    df['cut_encoded'] = df['cut'].map(cut_order)
    
    # Encoding ordinal untuk color (J ke D: 0-6)
    color_order = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    df['color_encoded'] = df['color'].map(color_order)
    
    # Encoding ordinal untuk clarity (I1 ke IF: 0-7)
    clarity_order = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 
                      'VVS2': 5, 'VVS1': 6, 'IF': 7}
    df['clarity_encoded'] = df['clarity'].map(clarity_order)
    
    # Feature engineering: volume berlian
    df['volume'] = df['x'] * df['y'] * df['z']
    
    # Feature engineering: harga per karat
    df['price_per_carat'] = df['price'] / df['carat']
    
    # Feature engineering: interaksi cut dan clarity
    df['cut_clarity_score'] = df['cut_encoded'] * df['clarity_encoded']
    
    # Standarisasi fitur numerik (kecuali target)
    features_to_scale = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'price_per_carat']
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Hapus kolom asli yang sudah tidak diperlukan
    df = df.drop(['cut', 'color', 'clarity'], axis=1)
    
    # Tampilkan hasil preprocessing
    print("\nHasil preprocessing (5 baris pertama):")
    print(df.head())
    
    print("\nInfo dataset setelah preprocessing:")
    print(df.info())
    
    # Simpan dataset yang sudah dipreprocessing
    os.makedirs('preprocessing/data_preprocessed', exist_ok=True)
    df.to_csv('preprocessing/data_preprocessed/diamonds_clean.csv', index=False)
    
    print("\nDataset bersih telah disimpan sebagai 'preprocessing/data_preprocessed/diamonds_clean.csv'")
    print(f"Jumlah baris: {len(df)}")
    print(f"Jumlah kolom: {len(df.columns)}")
    print("\nNama-nama kolom dalam dataset bersih:")
    for col in df.columns:
        print(f"  - {col}")

if __name__ == "__main__":
    main()
