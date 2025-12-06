import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# --- CONFIGURATION ---
INPUT_FILE = "FINAL_DATASET_READY_FOR_TRAINING.parquet"
OUTPUT_FILE = "FINAL_DATASET_PCA.parquet"

def get_pca_score(df, columns, name):
    """
    Extracts the First Principal Component (PC1) from a group of columns.
    This represents the 'Dominant Trend' shared by all those features.
    """
    # 1. Select & Clean Data
    cols_present = [c for c in columns if c in df.columns]
    
    if not cols_present:
        print(f"Skipping {name}: No columns found.")
        return pd.Series(0, index=df.index)
        
    X = df[cols_present].values
    
    # 2. Impute & Scale
    X = SimpleImputer(strategy='mean').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    
    # 3. Run PCA
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).flatten()
    
    # 4. Sign Correction 
    # PCA direction is arbitrary. It might say "-5" means "High Anger".
    # We take the average (mean) of all input features to be safer.
    # If 'face_angry' is broken (all zeros), 'iemo_ang' will still save us.
    input_sample = np.mean(X, axis=1) 
    
    correlation = np.corrcoef(input_sample, pc1)[0, 1]
    
    if correlation < 0:
        pc1 = pc1 * -1
        
    print(f"Created '{name}' from {len(cols_present)} features.")
    print(f"Explained Variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
    
    return pc1

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        return

    print("Loading Dataset...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Aggregate to Minute
    df['minute_time'] = pd.to_datetime(df['real_time']).dt.floor('min')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Close' not in numeric_cols: numeric_cols.append('Close')
    
    df_agg = df.groupby(['video_id', 'minute_time'])[numeric_cols].mean().reset_index()
    
    print(f"Processing {len(df_agg)} minute-intervals...")

    # 2. DEFINE GROUPS FOR PCA
    
    # A. NEGATIVITY (Stress/Anger)
    cols_neg = ['face_angry', 'iemo_ang', 'fin_neg', 'iemo_fru']
    df_agg['Unified_Negative'] = get_pca_score(df_agg, cols_neg, "Unified_Negative")

    # B. POSITIVITY (Confidence/Happy)
    cols_pos = ['face_happy', 'iemo_hap', 'openness', 'fin_pos']
    df_agg['Unified_Positive'] = get_pca_score(df_agg, cols_pos, "Unified_Positive")

    # C. UNCERTAINTY (Fear/Closed/Tilt)
    cols_unc = ['face_fear', 'iemo_fea', 'head_tilt_deg']
    df_agg['Unified_Uncertainty'] = get_pca_score(df_agg, cols_unc, "Unified_Uncertainty")

    # D. NEUTRALITY (Stability)
    cols_neu = ['face_neutral', 'iemo_neu', 'fin_neu']
    df_agg['Unified_Neutral'] = get_pca_score(df_agg, cols_neu, "Unified_Neutral")

    # 3. Clean Up
    final_cols = [
        'video_id', 'minute_time', 'Close', 
        'Unified_Negative', 'Unified_Positive', 'Unified_Uncertainty', 'Unified_Neutral'
    ]
    
    df_final = df_agg[final_cols]
    
    # 4. Save
    df_final.to_parquet(OUTPUT_FILE)
    print(f"Created PCA Unified Dataset: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()