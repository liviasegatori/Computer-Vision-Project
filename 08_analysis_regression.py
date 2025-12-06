import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# We point this to the NEW PCA file you just created
FILE_PATH = "FINAL_DATASET_PCA.parquet"

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found. Run create_unified_pca.py first.")
        return

    print("Loading PCA Unified Dataset")
    df = pd.read_parquet(FILE_PATH)
    
    # 1. Aggregate to Minute
    df['minute_time'] = pd.to_datetime(df['minute_time'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Close' not in numeric_cols: numeric_cols.append('Close')
    df_agg = df.groupby(['video_id', 'minute_time'])[numeric_cols].mean().reset_index()
    
    # Base Features (The PCA Scores)
    base_features = ['Unified_Negative', 'Unified_Positive', 'Unified_Uncertainty', 'Unified_Neutral']
    
    print("\nStarting Iterative OLS Regression")
    
    best_r2 = -1
    best_model_summary = ""
    best_lag = 0
    
    # Loop through Lags to find the "Sweet Spot"
    for lag in range(1, 16): # Test 1 to 15 minutes reaction time
        # Create Target (Future Return)
        future_close = df_agg.groupby('video_id')['Close'].shift(-lag)
        # We multiply by 100 to make coefficients readable (percentages)
        df_agg['Target'] = (future_close - df_agg['Close']) / df_agg['Close'] * 100
        
        # Prepare Data
        df_model = df_agg.dropna(subset=['Target'] + base_features).copy()
        
        # --- FEATURE ENGINEERING (Non-Linearity) ---
        # Maybe the market reacts exponentially to high values?
        for f in base_features:
            df_model[f"{f}_Sq"] = df_model[f] ** 2
            
        # Interaction: Does Negativity matter more if Uncertainty is high?
        df_model['Neg_x_Uncertainty'] = df_model['Unified_Negative'] * df_model['Unified_Uncertainty']
        
        # Full Feature Set to Test
        features_extended = base_features + [
            'Unified_Negative_Sq', 'Neg_x_Uncertainty'
        ]
        
        # Regression
        X = df_model[features_extended]
        X = sm.add_constant(X)
        Y = df_model['Target']
        
        try:
            model = sm.OLS(Y, X).fit()
            
            # We look for the model with the highest Explanatory Power (R-squared)
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_model_summary = model.summary()
                best_lag = lag
                
        except: pass

    print(f"\nBest model at: {best_lag} minutes")
    print(best_model_summary)
    
    with open("regression_pca_results.txt", "w") as f:
        f.write(str(best_model_summary))
    print("\nSaved results to 'regression_pca_results.txt'")

if __name__ == "__main__":
    main()