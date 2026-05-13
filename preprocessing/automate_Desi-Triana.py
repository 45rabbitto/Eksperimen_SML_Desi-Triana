
Script otomatis untuk preprocessing dataset diamonds
Digunakan untuk GitHub Actions workflow

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def create_directories():
    os.makedirs('preprocessing/data_preprocessed', exist_ok=True)
    print("Direktori siap")

def load_data():
    file_path = 'data_raw/diamonds.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")
    return df

def preprocess_data(df):
    """Melakukan preprocessing data untuk regresi"""
    
    # Hapus missing values
    initial_rows = len(df)
    df = df.dropna()
    print(f"  - Missing values dihapus: {initial_rows - len(df)} baris")
    
    # Hapus duplikat
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"  - Duplikat dihapus: {initial_rows - len(df)} baris")
    
    # Hapus outlier ekstrem (persentil 0.5 dan 99.5)
    lower_bound = df['price'].quantile(0.005)
    upper_bound = df['price'].quantile(0.995)
    initial_rows = len(df)
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    print(f"  - Outlier ekstrem dihapus: {initial_rows - len(df)} baris (1% data)")
    
    # Transformasi log pada target
    df['price_log'] = np.log1p(df['price'])
    print("  - Log transformasi pada price berhasil")
    
    # Encoding ordinal untuk cut
    cut_order = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    df['cut_encoded'] = df['cut'].map(cut_order)
    
    # Encoding ordinal untuk color
    color_order = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    df['color_encoded'] = df['color'].map(color_order)
    
    # Encoding ordinal untuk clarity
    clarity_order = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 
                      'VVS2': 5, 'VVS1': 6, 'IF': 7}
    df['clarity_encoded'] = df['clarity'].map(clarity_order)
    print("  - Encoding kategorikal selesai")
    
    # Feature engineering
    df['volume'] = df['x'] * df['y'] * df['z']
    df['price_per_carat'] = df['price'] / df['carat']
    df['cut_clarity_score'] = df['cut_encoded'] * df['clarity_encoded']
    print("  - Feature engineering selesai (volume, price_per_carat, cut_clarity_score)")
    
    # Standarisasi fitur numerik
    features_to_scale = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'price_per_carat']
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    print("  - Standarisasi fitur numerik selesai")
    
    # Hapus kolom kategorikal asli
    df = df.drop(['cut', 'color', 'clarity'], axis=1)
    
    print("Preprocessing selesai!")
    return df

def save_data(df):
    output_path = 'preprocessing/data_preprocessed/diamonds_clean.csv'
    df.to_csv(output_path, index=False)
    print(f"\n Dataset bersih disimpan di: {output_path}")
    print(f"   - Jumlah baris: {len(df)}")
    print(f"   - Jumlah kolom: {len(df.columns)}")
    print(f"   - Ukuran file: {os.path.getsize(output_path) / 1024:.2f} KB")
    

    print("\nKolom dalam dataset bersih:")
    for col in df.columns:
        print(f"   - {col}")

def main():
 
    print("AUTOMATED DIAMONDS DATA PREPROCESSING")
    
    try:
        create_directories()
        df_raw = load_data()
        df_clean = preprocess_data(df_raw)
        save_data(df_clean)

        print("WORKFLOW PREPROCESSING BERHASIL!")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
