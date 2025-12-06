"""
Market Direction Prediction (Classification)
--------------------------------------------
This script runs a Grid Search to train a classifier (Random Forest, SVM, MLP)
to predict the DIRECTION of the market (Up/Down) based on behavioral features.

It optimizes for two environmental hyperparameters:
1. Lag: Reaction time of the market.
2. Threshold: The minimum volatility required to label a move as 'significant'.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

FILE_PATH = "FINAL_DATASET_PCA.parquet"

# 1. SEARCH SPACE FOR DATA (The "Environment")
LAGS = [1, 2, 3, 5] 
THRESHOLDS = [0.0001, 0.0002] 

# 2. Hyperparameters - the search space
# We define a grid for each model
PARAM_GRIDS = {
    "RF": {
        'n_estimators': [50, 100, 300, 500],       # Number of trees
        'max_depth': [3, 5, 8, 10, None],          # How deep each tree can go
        'min_samples_split': [2, 5, 10],           # Prevent overfitting
        'class_weight': ['balanced', None],        # Handle rare market jumps
        'random_state': [42]
    },
    "SVM": {
        'C': [0.1],                    # Regularization strength
        'kernel': ['rbf'],       # Type of decision boundary
        'gamma': ['scale'],                # Kernel coefficient
        'probability': [True]
    },
    "MLP": {
        'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64)], 
        'alpha': [0.0001, 0.01],              # Regularization
        'learning_rate_init': [0.001, 0.01],       # Speed of learning
        'max_iter': [1000],
        'random_state': [42]
    }
}

def get_data_for_config(df_orig, lag, threshold):
    df = df_orig.copy()
    df['minute_time'] = pd.to_datetime(df['minute_time'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    future_close = df.groupby('video_id')['Close'].shift(-lag)
    current_close = df['Close']
    ret = (future_close - current_close) / current_close
    
    conditions = [(ret > threshold), (ret < -threshold)]
    choices = [1, 0]
    
    df['Target'] = np.select(conditions, choices, default=np.nan)
    df_clean = df.dropna(subset=['Target'])
    
    features = ['Unified_Negative', 'Unified_Positive', 'Unified_Uncertainty', 'Unified_Neutral']
    features = [f for f in features if f in df_clean.columns]
    
    return df_clean[features], df_clean['Target'], df_clean['video_id']

def main():
    print("Starting Grid Search")
    
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found.")
        return
        
    df = pd.read_parquet(FILE_PATH)
    
    best_score = 0
    best_config_str = ""
    
    # Calculate total combinations 
    print("Testing combinations of Lags, Thresholds, and Model Parameters.")
    
    # Loop Lags & Thresholds
    for lag in LAGS:
        for thresh in THRESHOLDS:
            X, y, groups = get_data_for_config(df, lag, thresh)
            
            # Skip if too few samples
            if len(X) < 100: continue
            
            # Loop Models
            for model_name, grid in PARAM_GRIDS.items():
                # ParameterGrid creates a list of all combinations from the dictionary
                for params in ParameterGrid(grid):
                    
                    # Instantiate Model
                    if model_name == "SVM": model = SVC(**params)
                    elif model_name == "MLP": model = MLPClassifier(**params)
                    elif model_name == "RF": model = RandomForestClassifier(**params)
                    
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler()), 
                        ('model', model)
                    ])
                    
                    try:
                        scores = cross_val_score(pipeline, X, y, groups=groups, cv=GroupKFold(5), scoring='roc_auc')
                        avg_auc = scores.mean()
                        
                        # Create a short string for the params to keep output readable
                        # e.g. "RF | Est:50 D:3"
                        param_desc = " ".join([f"{k}:{v}" for k,v in params.items() if k != 'random_state' and k != 'probability' and k != 'max_iter'])
                        
                        if avg_auc > 0.54: # Only print interesting results to reduce noise
                            print(f"Better than random guess at: Lag {lag}m | Thresh {thresh} | {model_name} | {param_desc} -> AUC: {avg_auc:.4f}")
                        
                        if avg_auc > best_score:
                            best_score = avg_auc
                            best_config_str = f"Lag {lag}m | Thresh {thresh} | {model_name} | {param_desc}"
                    except: pass

    print("The best model:")
    print(f"   {best_config_str}")
    print(f"   AUC: {best_score:.4f}")

if __name__ == "__main__":
    main()