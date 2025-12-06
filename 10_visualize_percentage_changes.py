import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

filepath = "FINAL_DATASET_READY_FOR_TRAINING.parquet"
out_path = "plots_percentage_change"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

def aggregate_to_minute(df):
    # Aggregates 2s data to 1m data for smoother trends
    df['real_time'] = pd.to_datetime(df['real_time'])
    df['minute_time'] = df['real_time'].dt.floor('min')
    
    # Force numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_agg = df.groupby(['video_id', 'minute_time'])[numeric_cols].mean().reset_index()
    return df_agg

def calculate_pct_change_from_start(df, cols):
    # Calculates the % change of a column relative to its starting value
    for col in cols:
        if col not in df.columns: continue
        
        start_val = df[col].iloc[0]
        
        # Handle Zero Start (to avoid infinity)
        if start_val == 0: start_val = 0.01 
            
        new_col_name = f"{col}_PctChange"
        df[new_col_name] = ((df[col] - start_val) / start_val) * 100
        
    return df

def plot_pct_correlation_heatmap(df, save_folder):
    print("\nGenerating Percentage Change Correlation Matrix")
    
    # Define the specific columns we want to correlate
    target_cols = [
        'iemo_ang_PctChange', 'iemo_hap_PctChange',       # Audio
        'face_angry_PctChange', 'face_happy_PctChange',   # Face
        'openness_PctChange', 'head_tilt_deg_PctChange',  # Body
        'fin_neg_PctChange', 'fin_pos_PctChange',         # Text
        'Close_PctChange'                                 # Market
    ]
    
    # Filter for columns that actually exist in the dataframe
    cols_to_plot = [c for c in target_cols if c in df.columns]
    
    if not cols_to_plot:
        print("No percentage change columns found for heatmap.")
        return

    # Calculate Correlation
    corr = df[cols_to_plot].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, square=True)
    
    plt.title("Correlation of % Changes (Normalized Trends)", fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    filename = f"{save_folder}/global_correlation_matrix_pct_change.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_pct_timeline_from_processed(video_id, df_agg, save_folder):
    # Create Time Axis (Minutes) if not exists
    if 'Time_Min' not in df_agg.columns:
        df_agg['Time_Min'] = (df_agg['minute_time'] - df_agg['minute_time'].iloc[0]).dt.total_seconds() / 60
    
    x_col = 'Time_Min'

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    
    # Audio
    ax1 = axes[0]
    if 'iemo_ang_PctChange' in df_agg.columns:
        ax1.plot(df_agg[x_col], df_agg['iemo_ang_PctChange'], label='Anger Δ%', color='#D62728', linewidth=2)
        ax1.plot(df_agg[x_col], df_agg['iemo_hap_PctChange'], label='Confidence Δ%', color='#2CA02C', linewidth=2)
    ax1.set_ylabel("% Change")
    ax1.set_title(f"1. Audio Tone (% Change) - {video_id[:8]}", loc='left', fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.grid(True, alpha=0.3)
    
    # Face
    ax2 = axes[1]
    if 'face_angry_PctChange' in df_agg.columns:
        ax2.plot(df_agg[x_col], df_agg['face_angry_PctChange'], label='Anger Δ%', color='#FF7F0E', linewidth=2)
        ax2.plot(df_agg[x_col], df_agg['face_happy_PctChange'], label='Happiness Δ%', color='#17BECF', linewidth=2)
    ax2.set_ylabel("% Change")
    ax2.set_title("2. Facial Expressions (% Change)", loc='left', fontweight='bold')
    ax2.legend(loc="upper left")
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.grid(True, alpha=0.3)

    # Body
    ax3 = axes[2]
    if 'openness_PctChange' in df_agg.columns:
        ax3.plot(df_agg[x_col], df_agg['openness_PctChange'], label='Openness Δ%', color='#17BECF', linewidth=2)
        ax3.plot(df_agg[x_col], df_agg['head_tilt_deg_PctChange'], label='Head Tilt Δ%', color='gray', linestyle='--', linewidth=2)
    ax3.set_ylabel("% Change")
    ax3.set_title("3. Body Language (% Change)", loc='left', fontweight='bold')
    ax3.legend(loc="upper left")
    ax3.axhline(0, color='black', linewidth=1, linestyle='--')
    ax3.grid(True, alpha=0.3)

    # Text
    ax4 = axes[3]
    if 'fin_neg_PctChange' in df_agg.columns:
        ax4.plot(df_agg[x_col], df_agg['fin_neg_PctChange'], label='Negative Δ%', color='red', linewidth=2)
        ax4.plot(df_agg[x_col], df_agg['fin_pos_PctChange'], label='Positive Δ%', color='green', linewidth=2)
    ax4.set_ylabel("% Change")
    ax4.set_title("4. Text Sentiment (% Change)", loc='left', fontweight='bold')
    ax4.legend(loc="upper left")
    ax4.axhline(0, color='black', linewidth=1, linestyle='--')
    ax4.grid(True, alpha=0.3)

    # Market
    ax5 = axes[4]
    if 'Close_PctChange' in df_agg.columns:
        ax5.plot(df_agg[x_col], df_agg['Close_PctChange'], color='#1F77B4', linewidth=3, marker='o', markersize=4, label='S&P 500 Δ%')
    ax5.set_ylabel("% Change")
    ax5.set_xlabel("Time (Minutes from Start)")
    ax5.set_title("5. Market Reaction (% Change)", loc='left', fontweight='bold')
    ax5.legend(loc="upper left")
    ax5.axhline(0, color='black', linewidth=1, linestyle='--')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_folder}/pct_change_{video_id[:15]}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved timeline: {filename}")

def main():
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    os.makedirs(out_path, exist_ok=True)
    print("Loading Dataset")
    df = pd.read_parquet(filepath)
    
    # Force Close to numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    # Pre-process all videos 
    all_video_dfs = []
    unique_videos = df['video_id'].unique()
    
    print("Calculating Percentage Changes for videos")
    
    features_to_convert = [
        'iemo_ang', 'iemo_hap',       
        'face_angry', 'face_happy',   
        'openness', 'head_tilt_deg',  
        'fin_neg', 'fin_pos',         
        'Close'                       
    ]

    for vid in unique_videos:
        subset = df[df['video_id'] == vid].copy()
        if len(subset) < 10: continue
        
        # 1. Aggregate
        df_agg = aggregate_to_minute(subset)
        
        # 2. Calculate % Change
        df_agg = calculate_pct_change_from_start(df_agg, features_to_convert)
        
        all_video_dfs.append(df_agg)
        
    if not all_video_dfs:
        print("No valid data found.")
        return
        
    # Combine for Global Analysis
    df_final = pd.concat(all_video_dfs, ignore_index=True)
    
    # Plotting correlation matrix
    plot_pct_correlation_heatmap(df_final, out_path)
    
    # Plotting timelines
    print(f"\nGenerating Timelines for {len(all_video_dfs)} videos")
    for vid_df in all_video_dfs:
        vid_id = vid_df['video_id'].iloc[0]
        plot_pct_timeline_from_processed(vid_id, vid_df, out_path)

    print(f"\nOutput saved to '{out_path}'")

if __name__ == "__main__":
    main()