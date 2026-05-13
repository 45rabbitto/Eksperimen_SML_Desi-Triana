# -*- coding: utf-8 -*-

Automatic preprocessing script for diamonds dataset
Used for GitHub Actions workflow

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

def create_directories():
    """Create directories if not exist"""
    os.makedirs('preprocessing/data_preprocessed', exist_ok=True)
    print("[OK] Directory ready")

def download_sample_data():
    """Create sample data if file not found (for testing)"""
    print("[WARNING] diamonds.csv dataset not found!")
    print("[INFO] Creating sample data for testing...")
    
    # Sample data diamonds 
    np.random.seed(42)
    n_samples = 50
    
    sample_data = {
        'carat': np.random.uniform(0.2, 5.0, n_samples),
        'cut': np.random.choice(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], n_samples),
        'color': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], n_samples),
        'clarity': np.random.choice(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], n_samples),
        'depth': np.random.uniform(43, 79, n_samples),
        'table': np.random.uniform(43, 95, n_samples),
        'price': np.random.uniform(326, 18823, n_samples),
        'x': np.random.uniform(3, 10, n_samples),
        'y': np.random.uniform(3, 10, n_samples),
        'z': np.random.uniform(2, 6, n_samples),
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data to data_raw folder
    os.makedirs('data_raw', exist_ok=True)
    df.to_csv('data_raw/diamonds.csv', index=False)
    print("[OK] Sample dataset created: data_raw/diamonds.csv")
    return df

def load_data():
    """Load raw dataset"""
    file_path = 'data_raw/diamonds.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("[ERROR] Dataset not found at: {}".format(file_path))
        print("[INFO] Searching in alternative locations...")
        
        # Search in alternative locations
        alt_paths = [
            'diamonds.csv',
            '../data_raw/diamonds.csv',
            'preprocessing/data_raw/diamonds.csv'
        ]
        
        found = False
        for alt in alt_paths:
            if os.path.exists(alt):
                file_path = alt
                found = True
                print("[OK] Dataset found at: {}".format(file_path))
                break
        
        if not found:
            print("[INFO] Dataset not found, creating sample data...")
            df = download_sample_data()
            return df
    
    df = pd.read_csv(file_path)
    print("[OK] Dataset loaded: {} rows, {} columns".format(len(df), len(df.columns)))
    print("    Columns: {}".format(list(df.columns)))
    return df

def preprocess_data(df):
    """Preprocess data for regression"""
    
    print("\n[INFO] Starting preprocessing...")
    
    # Remove missing values
    initial_rows = len(df)
    df = df.dropna()
    print("  - Missing values removed: {} rows".format(initial_rows - len(df)))
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print("  - Duplicates removed: {} rows".format(initial_rows - len(df)))
    
    # Remove extreme outliers (percentile 0.5 and 99.5) 
    if len(df) > 10:
        lower_bound = df['price'].quantile(0.005)
        upper_bound = df['price'].quantile(0.995)
        initial_rows = len(df)
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        print("  - Extreme outliers removed: {} rows".format(initial_rows - len(df)))
    else:
        print("  - Skip outlier handling (data too small)")
    
    # Log transformation on target
    df['price_log'] = np.log1p(df['price'])
    print("  - Log transformation on price completed")
    
    # Ordinal encoding for cut
    cut_order = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    df['cut_encoded'] = df['cut'].map(cut_order)
    
    # Ordinal encoding for color
    color_order = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    df['color_encoded'] = df['color'].map(color_order)
    
    # Ordinal encoding for clarity
    clarity_order = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 
                      'VVS2': 5, 'VVS1': 6, 'IF': 7}
    df['clarity_encoded'] = df['clarity'].map(clarity_order)
    print("  - Categorical encoding completed")
    
    # Feature engineering
    df['volume'] = df['x'] * df['y'] * df['z']
    df['price_per_carat'] = df['price'] / df['carat']
    df['cut_clarity_score'] = df['cut_encoded'] * df['clarity_encoded']
    print("  - Feature engineering completed")
    
    # Standardization of numerical features
    features_to_scale = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'price_per_carat']
    # Only standardize existing columns
    features_to_scale_existing = [f for f in features_to_scale if f in df.columns]
    
    if len(features_to_scale_existing) > 0:
        scaler = StandardScaler()
        df[features_to_scale_existing] = scaler.fit_transform(df[features_to_scale_existing])
        print("  - Numerical feature standardization completed: {}".format(features_to_scale_existing))
    
    # Drop original categorical columns (if exist)
    cols_to_drop = ['cut', 'color', 'clarity']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_cols_to_drop:
        df = df.drop(existing_cols_to_drop, axis=1)
    
    print("[OK] Preprocessing completed!")
    return df

def save_data(df):
    """Save preprocessed dataset"""
    output_path = 'preprocessing/data_preprocessed/diamonds_clean.csv'
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print("\n[OK] Clean dataset saved to: {}".format(output_path))
    print("    - Number of rows: {}".format(len(df)))
    print("    - Number of columns: {}".format(len(df.columns)))
    print("    - File size: {:.2f} KB".format(os.path.getsize(output_path) / 1024))
    
    # Display column info
    print("\n[INFO] Columns in clean dataset:")
    for col in df.columns:
        dtype = df[col].dtype
        print("    - {} ({})".format(col, dtype))
    
    return output_path

def main():
    """Main function to run preprocessing"""
    print("AUTOMATED DIAMONDS DATA PREPROCESSING")

    try:
        create_directories()
        df_raw = load_data()
        df_clean = preprocess_data(df_raw)
        output_path = save_data(df_clean)
        
        # Verify file was created successfully
        if os.path.exists(output_path):
            print("\n" + "="*50)
            print("[SUCCESS] WORKFLOW PREPROCESSING COMPLETED!")
            print("="*50)
            sys.exit(0)
        else:
            print("\n[ERROR] Output file not found!")
            sys.exit(1)
        
    except Exception as e:
        print("\n[ERROR] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
