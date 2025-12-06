"""
Audio Inference & Feature Augmentation
--------------------------------------
This script applies the fine-tuned Wav2Vec2 model (from script 04) to the 
financial dataset. It reads the raw features, locates the original audio, 
and appends 8 distinct emotion probability columns (Anger, Happiness, etc.).
"""

import pandas as pd
import torch
import torchaudio
import glob
import os
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from tqdm import tqdm

# --- CONFIGURATION ---
# Use the current working directory as the project root
PROJECT_ROOT = os.getcwd()

# Looks for audio files
# It checks the root AND the 'output' subfolder
SEARCH_DIRS = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "output")
]

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODEL_PATH = os.path.join(PROJECT_ROOT, "wav2vec2-iemocap-local")
RESULT_DIR = os.path.join(PROJECT_ROOT, "features_with_emotions")

# IEMOCAP Label Map
ID2LABEL = {
    0: 'ang', 1: 'dis', 2: 'exc', 3: 'fea', 4: 'fru', 
    5: 'hap', 6: 'neu', 7: 'oth', 8: 'sad', 9: 'sur'
}

def find_audio_path_smart(video_id):
    # Take the first 8 characters (the date) as the Short ID
    short_id = str(video_id)[:8] 
    
    for root in SEARCH_DIRS:
        if not os.path.exists(root): continue
        
        # 1. Search for folders containing the date in the name
        candidates = glob.glob(os.path.join(root, f"*{short_id}*"))
        
        for folder in candidates:
            if os.path.isdir(folder):
                audio_files = glob.glob(os.path.join(folder, "**", "audio.wav"), recursive=True)
                if audio_files:
                    return audio_files[0] # Return the first valid audio found

    return None

def process_file(parquet_path, model, feature_extractor, device):
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {os.path.basename(parquet_path)} with {len(df)} rows.")
    
    emotion_cols = {label: [None]*len(df) for label in ID2LABEL.values()}
    unique_videos = df['video_id'].unique()
    
    for vid in unique_videos:
        vid_indices = df[df['video_id'] == vid].index
        
        audio_path = find_audio_path_smart(vid)
            
        try:
            # Load Audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 16k if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Convert Stereo to Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze()

            # Batch processing per video
            for idx in tqdm(vid_indices, desc=f"   Scoring {vid[:10]}..."):
                t_start = df.loc[idx, 't_start']
                t_end = df.loc[idx, 't_end']
                
                s_start = int(t_start * 16000)
                s_end = int(t_end * 16000)
                
                chunk = waveform[s_start:s_end]
                
                # Skip if chunk is empty or too short
                if chunk.numel() < 100: continue

                inputs = feature_extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                for i, score in enumerate(probs):
                    emotion_cols[ID2LABEL[i]][idx] = float(score)

        except Exception as e:
            print(f"Error processing audio for {vid}: {e}")

    # Add columns to DataFrame
    for label, values in emotion_cols.items():
        df[f"iemo_{label}"] = values
        
    return df

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Load Model
    print(f"Loading model from {MODEL_PATH} ...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    except Exception as e:
        print(f"Could not load model. Check path.\n{e}")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # --- FILTERING ---
    all_files = glob.glob(os.path.join(FEATURES_DIR, "*.parquet"))
    
    files_to_process = []
    print("\n--- Selecting Files ---")
    for f in all_files:
        fname = os.path.basename(f)
        
        # 1. MUST start with "features_"
        if not fname.startswith("features_"):
            continue
            
        # 2. MUST NOT contain "faces_" (redundant but safe)
        if "faces_" in fname:
            continue
            
        # 3. MUST NOT be already processed ("_with_emotions")
        if "_with_emotions" in fname:
            continue
            
        files_to_process.append(f)
        print(f"Selected: {fname}")

    if not files_to_process:
        print("No valid 'features_*.parquet' files found to process.")
        return
    
    print(f"\nStarting processing of {len(files_to_process)} files...\n")
    
    for f in files_to_process:
        print(f"--- Processing {os.path.basename(f)} ---")
        df_new = process_file(f, model, feature_extractor, device)
        
        out_name = os.path.basename(f).replace(".parquet", "_with_emotions.parquet")
        out_path = os.path.join(RESULT_DIR, out_name)
        df_new.to_parquet(out_path)
        print(f"Saved: {out_name}\n")

if __name__ == "__main__":
    main()