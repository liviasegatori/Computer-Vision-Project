import os, json, argparse, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# SMART PATH RESOLUTION
# =========================
def find_file_smart(base_dir: Path, video_id: str, filename: str) -> Path:
    """
    Searches for a file (like audio.wav) recursively if direct path fails.
    """
    # 1. Try Direct Path
    direct_path = base_dir / video_id / filename
    if direct_path.exists():
        return direct_path
    
    # 2. Try looking in the specific Output Folder for this date
    short_id = str(video_id)[:8]
    
    # Check current directory and subfolders
    search_roots = [base_dir, base_dir / f"output_{short_id}"]
    
    for root in search_roots:
        if not root.exists(): continue
        
        # Look for the file recursively inside this root
        candidates = list(root.glob(f"**/{filename}"))
        
        # Filter candidates that match the video_id (to avoid grabbing wrong audio)
        for c in candidates:
            if video_id in str(c) or short_id in str(c):
                return c
                
    return Path("") 

# =========================
# Models (lazy singletons)
# =========================
_W2V, _W2V_PROC = None, None
_FINBERT, _FIN_TOK = None, None

def get_w2v():
    global _W2V, _W2V_PROC
    if _W2V is None:
        _W2V_PROC = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _W2V = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()
    return _W2V, _W2V_PROC

def get_finbert():
    global _FINBERT, _FIN_TOK
    if _FINBERT is None:
        name = "ProsusAI/finbert"
        _FIN_TOK = AutoTokenizer.from_pretrained(name)
        _FINBERT = AutoModelForSequenceClassification.from_pretrained(name).eval()
    return _FINBERT, _FIN_TOK

# =========================
# Feature helpers
# =========================
def load_audio_slice(media_path: Path, t0: float, t1: float, sr=16000):
    if not media_path or not media_path.exists() or media_path.is_dir():
        return None, None
    dur = max(0.0, float(t1) - float(t0))
    try:
        y, sr = librosa.load(str(media_path), sr=sr, offset=float(t0), duration=dur, mono=True)
        return y, sr
    except Exception as e:
        # print(f"Error loading audio: {e}")
        return None, None

def wav2vec2_embed(y, sr):
    if y is None or len(y) == 0:
        return None
    model, proc = get_w2v()
    with torch.no_grad():
        inputs = proc(y, sampling_rate=sr, return_tensors="pt")
        out = model(inputs.input_values)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return emb

def finbert_scores(text):
    if not isinstance(text, str) or not text.strip():
        return dict(fin_pos=np.nan, fin_neu=np.nan, fin_neg=np.nan)
    model, tok = get_finbert()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return dict(fin_neg=float(probs[0]), fin_neu=float(probs[1]), fin_pos=float(probs[2]))

def read_pose_feats(kp_path: Path):
    if not kp_path or not kp_path.exists():
        return dict(openness=np.nan, torso_lean_deg=np.nan, head_tilt_deg=np.nan, shoulder_width=np.nan)
    try:
        with open(kp_path, "r") as f:
            j = json.load(f)
        L = j.get("pose_landmarks")
        if not L or len(L) < 25:
            return dict(openness=np.nan, torso_lean_deg=np.nan, head_tilt_deg=np.nan, shoulder_width=np.nan)
        def v(idx):
            a = L[idx]; return np.array([a["x"], a["y"], a["z"]], dtype=float)
        Ls, Rs, Lh, Rh = v(11), v(12), v(23), v(24)
        wristL, wristR = v(15), v(16)
        nose = v(0)
        out = dict(openness=np.nan, torso_lean_deg=np.nan, head_tilt_deg=np.nan, shoulder_width=np.nan)
        shoulder_vec = Rs - Ls
        out["shoulder_width"] = float(np.linalg.norm(shoulder_vec))
        mid_s = 0.5*(Ls+Rs); mid_h = 0.5*(Lh+Rh)
        torso = mid_s - mid_h
        vertical = np.array([0, -1, 0], dtype=float)
        cosang = np.clip(np.dot(torso/np.linalg.norm(torso), vertical), -1, 1)
        out["torso_lean_deg"] = float(np.degrees(np.arccos(cosang)))
        wrist_dist = np.linalg.norm(wristR - wristL)
        out["openness"] = float(wrist_dist / (out["shoulder_width"] + 1e-6))
        head_vec = nose - mid_s
        cosang_h = np.clip(np.dot(head_vec/np.linalg.norm(head_vec), vertical), -1, 1)
        out["head_tilt_deg"] = float(np.degrees(np.arccos(cosang_h)))
        return out
    except Exception:
        return dict(openness=np.nan, torso_lean_deg=np.nan, head_tilt_deg=np.nan, shoulder_width=np.nan)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", default="features.parquet")
    ap.add_argument("--audio_sr", type=int, default=16000)
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    
    if not Path(args.manifest).exists():
        print(f"Manifest not found: {args.manifest}")
        return

    df = pd.read_csv(args.manifest)
    total_rows = len(df)
    
    unique_videos = df['video_id'].unique()
    audio_map = {}
    print(f"Scanning for audio files for {len(unique_videos)} videos...")
    for vid in unique_videos:
        found_path = find_file_smart(base_dir, vid, "audio.wav")
        if found_path.exists():
            audio_map[vid] = found_path
            print(f"Found Audio for {vid}: {found_path}")
        else:
            print(f"Could NOT find audio.wav for {vid}")
            audio_map[vid] = Path("")

    rows = []
    start_time = time.time()

    with tqdm(total=total_rows, desc="Extracting features", ncols=100, dynamic_ncols=True) as pbar:
        for i, r in df.iterrows():
            vid = str(r.get("video_id", "")).strip()
            t0, t1 = float(r["t_start"]), float(r["t_end"])

            # Resolve paths
            audio_path = audio_map.get(vid, Path(""))
            
            # Keypoints
            kp_rel = str(r.get("keypoints_path", ""))
            if audio_path.exists() and "audio.wav" in str(audio_path):
                 keyp_path = audio_path.parent / "keypoints" / Path(kp_rel).name
            else:
                 keyp_path = base_dir / kp_rel

            a_emb = None
            if audio_path.exists():
                y, sr = load_audio_slice(audio_path, t0, t1, sr=args.audio_sr)
                if y is not None and y.size > 0:
                    a_emb = wav2vec2_embed(y, sr)

            s = finbert_scores(r.get("text_segment", ""))
            bl = read_pose_feats(keyp_path)

            row = {
                "video_id": vid, "t_start": t0, "t_end": t1,
                "fin_pos": s["fin_pos"], "fin_neu": s["fin_neu"], "fin_neg": s["fin_neg"],
                "openness": bl["openness"], "torso_lean_deg": bl["torso_lean_deg"],
                "head_tilt_deg": bl["head_tilt_deg"], "shoulder_width": bl["shoulder_width"],
                "audio_source": str(audio_path)
            }
            if a_emb is not None:
                for j, v in enumerate(a_emb.astype(np.float32)):
                    row[f"w2v_{j:03d}"] = float(v)
            rows.append(row)

            pbar.update(1)

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Done. Saved features to {args.out}")

if __name__ == "__main__":
    main()