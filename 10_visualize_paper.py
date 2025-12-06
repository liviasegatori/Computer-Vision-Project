import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- CONFIGURATION ---
FILE_PATH = "FINAL_DATASET_READY_FOR_TRAINING.parquet"
OUTPUT_DIR = "plots_paper_full"

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def calculate_market_returns(df):
    # Force numeric and calculate index
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    if df.empty:
        df['Price_Index'] = np.nan
        return df

    start_price = df['Close'].iloc[0]
    if start_price == 0: df['Price_Index'] = 1.0
    else: df['Price_Index'] = df['Close'] / start_price
    return df

def plot_master_timeline(video_id, df_vid, save_folder):
    window = 20 # Slightly smoother for readability
    
    df_vid = df_vid.sort_values("t_start")
    df_vid = calculate_market_returns(df_vid)
    
    if df_vid.empty: return

    # Create 5 Subplots (Rows) sharing the same Time X-Axis
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
    
    # --- ROW 1: AUDIO (Voice) ---
    ax1 = axes[0]
    if 'iemo_ang' in df_vid.columns:
        ax1.plot(df_vid['t_start'], df_vid['iemo_ang'].rolling(window).mean(), 
                 label='Anger (Audio)', color='#D62728', linewidth=2)
        ax1.plot(df_vid['t_start'], df_vid['iemo_hap'].rolling(window).mean(), 
                 label='Confidence/Happy (Audio)', color='#2CA02C', linewidth=2)
    ax1.set_ylabel("Prob")
    ax1.set_title(f"1. Audio Tone (Voice)", loc='left', fontweight='bold')
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # --- ROW 2: FACE (Expression) ---
    ax2 = axes[1]
    if 'face_angry' in df_vid.columns:
        ax2.plot(df_vid['t_start'], df_vid['face_angry'].rolling(window).mean(), 
                 label='Anger (Face)', color='#FF7F0E', linewidth=2)
        ax2.plot(df_vid['t_start'], df_vid['face_fear'].rolling(window).mean(), 
                 label='Fear (Face)', color='#9467BD', linewidth=2)
    ax2.set_ylabel("Prob")
    ax2.set_title("2. Facial Micro-Expressions", loc='left', fontweight='bold')
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # --- ROW 3: BODY (Posture) ---
    ax3 = axes[2]
    if 'openness' in df_vid.columns:
        # Openness (0 to 1 usually)
        ax3.plot(df_vid['t_start'], df_vid['openness'].rolling(window).mean(), 
                 label='Body Openness', color='#17BECF', linewidth=2)
        # Head Tilt (can be degrees, so we plot on twin axis if needed, but let's keep simple)
        # Normalize tilt roughly to 0-1 for visual comparison if needed, or just plot raw
        ax3.plot(df_vid['t_start'], df_vid['head_tilt_deg'].rolling(window).mean() / 20, 
                 label='Head Tilt (Scaled)', color='gray', linestyle='--', linewidth=1.5)
    ax3.set_ylabel("Score")
    ax3.set_title("3. Body Language & Posture", loc='left', fontweight='bold')
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # --- ROW 4: TEXT (Sentiment) ---
    ax4 = axes[3]
    if 'fin_neg' in df_vid.columns:
        ax4.fill_between(df_vid['t_start'], df_vid['fin_neg'].rolling(window).mean(), 
                         color='red', alpha=0.3, label='Negative Words')
        ax4.plot(df_vid['t_start'], df_vid['fin_pos'].rolling(window).mean(), 
                 label='Positive Words', color='green', linewidth=2)
    ax4.set_ylabel("Prob")
    ax4.set_title("4. Text Sentiment (What he said)", loc='left', fontweight='bold')
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    # --- ROW 5: MARKET (Outcome) ---
    ax5 = axes[4]
    ax5.plot(df_vid['t_start'], df_vid['Price_Index'], 
             color='#1F77B4', linewidth=3, label='S&P 500')
    ax5.set_ylabel("Index (Start=1.0)")
    ax5.set_xlabel("Time (seconds)")
    ax5.set_title("5. Market Reaction", loc='left', fontweight='bold')
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_folder}/multimodal_timeline_{video_id[:15]}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   -> Saved: {filename}")

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating Full 5-Row Plots...")
    df = pd.read_parquet(FILE_PATH)
    
    unique_videos = df['video_id'].unique()
    for vid in unique_videos:
        subset = df[df['video_id'] == vid].copy()
        if len(subset) > 60:
            plot_master_timeline(vid, subset, OUTPUT_DIR)

    print(f"\nDone. Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()