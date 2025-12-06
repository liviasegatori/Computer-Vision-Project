import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- CONFIGURATION ---
FILE_PATH = "FINAL_DATASET_READY_FOR_TRAINING.parquet"
OUTPUT_DIR = "plots_minute_resolution"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

def get_aggregated_dataset(df):
    print("   -> Aggregating full dataset to 1-minute intervals...")
    df['real_time'] = pd.to_datetime(df['real_time'])
    df['minute_time'] = df['real_time'].dt.floor('min')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_agg = df.groupby(['video_id', 'minute_time'])[numeric_cols].mean().reset_index()
    return df_agg

def plot_minute_correlation_heatmap(df_agg, save_folder):
    print("\nGenerating Minute-Level Correlation Matrix...")
    cols = [
        'iemo_ang', 'iemo_hap', 'iemo_fea', 
        'face_angry', 'face_happy', 'face_fear', 
        'openness', 'head_tilt_deg', 
        'fin_neg', 'fin_pos', 
        'Close'
    ]
    cols = [c for c in cols if c in df_agg.columns]
    corr = df_agg[cols].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, square=True)
    plt.title("Correlation Matrix (1-Minute Resolution)", fontweight='bold', fontsize=16)
    plt.tight_layout()
    filename = f"{save_folder}/global_correlation_matrix_1min.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   -> Saved: {filename}")

def plot_minute_timeline(video_id, df_agg_vid, save_folder):
    # Calculate Price Index
    start_price = df_agg_vid['Close'].iloc[0]
    if start_price == 0: start_price = 1.0
    df_agg_vid = df_agg_vid.copy()
    df_agg_vid['Price_Index'] = df_agg_vid['Close'] / start_price
    
    # Create relative time axis
    df_agg_vid['Time_Min'] = (df_agg_vid['minute_time'] - df_agg_vid['minute_time'].iloc[0]).dt.total_seconds() / 60

    # Create 5 Subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
    x_col = 'Time_Min'
    
    # --- ROW 1: AUDIO ---
    ax1 = axes[0]
    if 'iemo_ang' in df_agg_vid.columns:
        ax1.plot(df_agg_vid[x_col], df_agg_vid['iemo_ang'], label='Anger (Voice)', color='#D62728', linewidth=2.5, marker='o', markersize=4)
        ax1.plot(df_agg_vid[x_col], df_agg_vid['iemo_hap'], label='Confidence (Voice)', color='#2CA02C', linewidth=2.5, marker='o', markersize=4)
    ax1.set_ylabel("Avg Prob")
    ax1.set_title(f"1. Audio Tone (Minute Avg) - {video_id[:8]}", loc='left', fontweight='bold')
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # --- ROW 2: FACE (FIXED: Happy instead of Fear) ---
    ax2 = axes[1]
    if 'face_angry' in df_agg_vid.columns:
        ax2.plot(df_agg_vid[x_col], df_agg_vid['face_angry'], label='Anger (Face)', color='#FF7F0E', linewidth=2.5) # Orange
        # CHANGED HERE: Back to Happiness
        ax2.plot(df_agg_vid[x_col], df_agg_vid['face_happy'], label='Happiness (Face)', color='#17BECF', linewidth=2.5) # Cyan
    ax2.set_ylabel("Avg Prob")
    ax2.set_title("2. Facial Expressions (Minute Avg)", loc='left', fontweight='bold')
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # --- ROW 3: BODY ---
    ax3 = axes[2]
    if 'openness' in df_agg_vid.columns:
        ax3.plot(df_agg_vid[x_col], df_agg_vid['openness'], label='Body Openness', color='#17BECF', linewidth=2.5)
        ax3.plot(df_agg_vid[x_col], df_agg_vid['head_tilt_deg'] / 20, label='Head Tilt (Scaled)', color='gray', linestyle='--', linewidth=2)
    ax3.set_ylabel("Score")
    ax3.set_title("3. Body Language (Minute Avg)", loc='left', fontweight='bold')
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # --- ROW 4: TEXT ---
    ax4 = axes[3]
    if 'fin_neg' in df_agg_vid.columns:
        ax4.fill_between(df_agg_vid[x_col], df_agg_vid['fin_neg'], color='red', alpha=0.2, label='Negative Words')
        ax4.plot(df_agg_vid[x_col], df_agg_vid['fin_pos'], label='Positive Words', color='green', linewidth=2.5)
    ax4.set_ylabel("Avg Prob")
    ax4.set_title("4. Text Sentiment (Minute Avg)", loc='left', fontweight='bold')
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    # --- ROW 5: MARKET ---
    ax5 = axes[4]
    ax5.plot(df_agg_vid[x_col], df_agg_vid['Price_Index'], color='#1F77B4', linewidth=3, marker='o', markersize=4, label='S&P 500')
    ax5.set_ylabel("Index (Start=1.0)")
    ax5.set_xlabel("Time (Minutes from Start)")
    ax5.set_title("5. Market Reaction (Real 1-Min Data)", loc='left', fontweight='bold')
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_folder}/minute_timeline_{video_id[:15]}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   -> Saved plot: {filename}")

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading Dataset...")
    df = pd.read_parquet(FILE_PATH)
    
    df_agg = get_aggregated_dataset(df)
    plot_minute_correlation_heatmap(df_agg, OUTPUT_DIR)

    unique_videos = df_agg['video_id'].unique()
    print(f"\nGenerating Minute-Level Plots for {len(unique_videos)} videos...")
    
    for vid in unique_videos:
        subset = df_agg[df_agg['video_id'] == vid]
        if len(subset) > 10: 
            plot_minute_timeline(vid, subset, OUTPUT_DIR)

    print(f"\nDONE. Check the folder '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()