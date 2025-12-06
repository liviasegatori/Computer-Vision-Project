import pandas as pd
import glob
import os
import re

# --- CONFIGURATION ---
EMOTION_FEATURES_DIR = "features_with_emotions"
FACE_FEATURES_DIR = "features"
MARKET_DIR = "market_data"
TIMESTAMPS_CSV = "real_start_times.csv"
OUTPUT_FILE = "FINAL_DATASET_READY_FOR_TRAINING.parquet"

def clean_id_for_matching(raw_id):
    """ Extracts 8-digit date from ID. """
    match = re.search(r"(\d{8})", str(raw_id))
    if match:
        return match.group(1)
    return str(raw_id).strip()

def load_and_combine_market_data():
    """
    Loads all 'df_*.csv' files from market_data/, combines them,
    and standardizes the Timezone to match video timestamps.
    """
    files = glob.glob(os.path.join(MARKET_DIR, "df_*.csv"))
    if not files:
        print("No 'df_*.csv' files found in market_data/")
        return None
    
    print(f"   -> Loading {len(files)} market files...")
    dfs = []
    for f in files:
        try:
            # Read CSV
            df = pd.read_csv(f)
            # Parse Timestamp (Handling the offset automatically)
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            dfs.append(df)
        except Exception as e:
            print(f"   Error reading {f}: {e}")
    
    if not dfs: return None
    
    # Combine all chunks
    big_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by time
    big_df = big_df.sort_values('ts')
     
    # 1. Convert to NY Timezone
    big_df['ts'] = big_df['ts'].dt.tz_convert('America/New_York')
    # 2. Remove timezone info 
    big_df['ts'] = big_df['ts'].dt.tz_localize(None)
    
    # Set Index
    big_df = big_df.set_index('ts')
    
    print(f"   Combined Market Data: {len(big_df)} rows")
    print(f"   Time Range: {big_df.index.min()} to {big_df.index.max()}")
    
    return big_df

def main():
    print("STARTING FINAL MERGE")

    # 1. LOAD FEATURES
    print("Loading Main Features...")
    emo_files = glob.glob(os.path.join(EMOTION_FEATURES_DIR, "*_with_emotions.parquet"))
    if not emo_files:
        print("No features found")
        return
        
    df_main = pd.concat([pd.read_parquet(f) for f in emo_files], ignore_index=True)
    df_main['match_key'] = df_main['video_id'].apply(clean_id_for_matching)
    print(f"   Loaded {len(df_main)} rows.")

    # 2. LOAD FACES
    print("Loading Faces...")
    face_files = glob.glob(os.path.join(FACE_FEATURES_DIR, "faces_*.parquet"))
    if face_files:
        df_faces = pd.concat([pd.read_parquet(f) for f in face_files], ignore_index=True)
        df_main['t_round'] = df_main['t_start'].round(3)
        df_faces['t_round'] = df_faces['t_start'].round(3)
        
        print("   Merging Faces...")
        df_main = pd.merge(
            df_main,
            df_faces.drop(columns=['t_start', 't_end']),
            on=['video_id', 't_round'],
            how='left'
        )
    else:
        print("No face files found.")

    # 3. LOAD MARKET DATA (Global Load)
    print("Loading & Processing Market Data...")
    market_df = load_and_combine_market_data()
    if market_df is None: return

    # 4. ALIGNMENT LOOP
    print("Aligning Video with Market...")
    try:
        # Handle CSV delimiter
        with open(TIMESTAMPS_CSV, 'r') as f: header = f.readline()
        sep = ';' if ';' in header else ','
        time_map = pd.read_csv(TIMESTAMPS_CSV, sep=sep)
    except:
        print(f"Could not read {TIMESTAMPS_CSV}")
        return

    final_dfs = []
    
    for _, row in time_map.iterrows():
        csv_id_raw = row['video_id']
        match_date = clean_id_for_matching(csv_id_raw)
        
        subset = df_main[df_main['match_key'] == match_date].copy()
        
        if subset.empty:
            print(f"Skipping {match_date} (No features found)")
            continue

        try:
            start_dt = pd.to_datetime(row['start_time_et'])
            # Create Real Time column 
            subset['real_time'] = start_dt + pd.to_timedelta(subset['t_end'], unit='s')
        except:
            print(f"Invalid date in CSV for {csv_id_raw}")
            continue

        # Merge with Market Data
        subset = subset.sort_values('real_time')
        
        # Merge closest candle (backward look)
        aligned = pd.merge_asof(
            subset,
            market_df[['Close', 'Open', 'Volume']], 
            left_on='real_time',
            right_index=True,
            direction='backward',
            tolerance=pd.Timedelta("10m") 
        )
        
        valid = aligned['Close'].notna().sum()
        if valid > 0:
            final_dfs.append(aligned)
            print(f"   Merged {match_date}: {valid} rows with price")
        else:
            print(f"   {match_date}: No market overlap found.")
            print(f"    Video Time: {subset['real_time'].iloc[0]}")

    # 5. SAVE
    if final_dfs:
        full_dataset = pd.concat(final_dfs, ignore_index=True)
        # Clean columns
        drops = ['match_key', 't_round', 't_start_round']
        full_dataset.drop(columns=[c for c in drops if c in full_dataset.columns], inplace=True)
        
        full_dataset.to_parquet(OUTPUT_FILE)
        print(f"Final Dataset: {OUTPUT_FILE}")
        print(f"   Rows: {len(full_dataset)} | Columns: {len(full_dataset.columns)}")
    else:
        print("Failed")

if __name__ == "__main__":
    main()