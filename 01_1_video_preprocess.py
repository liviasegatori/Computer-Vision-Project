import os
import json
import math
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import pandas as pd
from tqdm import tqdm

# Whisper for timestamped ASR
import whisper
import ffmpeg

# MediaPipe for body keypoints
import mediapipe as mp

# -------------------------
# Helpers
# -------------------------

def slugify(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )

def run_cmd(cmd: List[str]) -> None:
    """Run a shell command and raise on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nOutput:\n{proc.stdout}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def video_duration_seconds(video_path: Path) -> float:
    """Get duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(proc.stdout.strip())
    except:
        return 0.0

# -------------------------
# Stage 1: Frames & Audio
# -------------------------

def extract_frames_ffmpeg(video_path: Path, frames_dir: Path, fps: float):
    ensure_dir(frames_dir)
    # e.g., 0.5 fps => 1 frame every 2 seconds
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(frames_dir / "frame_%06d.jpg")
    ]
    run_cmd(cmd)

def extract_audio_ffmpeg(video_path: Path, audio_path: Path):
    # ~230 MB for a 2h06m file (vs ~1.38 GB before)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",              # no video
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        "-c:a", "pcm_s16le",# WAV PCM
        str(audio_path)     # .../audio.wav
    ]
    run_cmd(cmd)


# -------------------------
# Stage 2: Whisper ASR
# -------------------------

def transcribe_whisper(audio_path: Path, out_json: Path, whisper_model: str = "large", device: str = None):
    model = whisper.load_model(whisper_model, device=device) if device else whisper.load_model(whisper_model)
    result = model.transcribe(str(audio_path), verbose=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# -------------------------
# Stage 3: Pose / Keypoints (MediaPipe)
# -------------------------

def extract_pose_on_frames(frames_dir: Path, keypoints_dir: Path):
    ensure_dir(keypoints_dir)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)
    image_files = sorted(frames_dir.glob("frame_*.jpg"))
    for img_path in tqdm(image_files, desc=f"Pose on {frames_dir.parent.name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(img_rgb)
        data = {"pose_landmarks": None}
        if res.pose_landmarks:
            data["pose_landmarks"] = [
                {
                    "x": float(l.x),
                    "y": float(l.y),
                    "z": float(l.z),
                    "visibility": float(l.visibility)
                } for l in res.pose_landmarks.landmark
            ]
        out_path = keypoints_dir / f"{img_path.stem}.json"
        with open(out_path, "w") as f:
            json.dump(data, f)
    mp_pose.close()

# -------------------------
# Stage 4: Build Manifest
# -------------------------

def load_whisper_segments(transcript_json: Path) -> List[Dict[str, Any]]:
    if not transcript_json.exists():
        return []
    with open(transcript_json, "r", encoding="utf-8") as f:
        j = json.load(f)
    # Whisper stores segments with start/end seconds
    return j.get("segments", [])

def nearest_frame_for_time(frames_dir: Path, fps_sampling: float, t: float) -> Path:
    """
    Given a time t (sec), find the frame index that matches nearest sampling step.
    With fps=0.5, frames are at 0s, 2s, 4s...
    """
    step = 1.0 / fps_sampling  # 2 sec at 0.5 fps
    idx = int(round(t / step)) + 1  # 1-based indexing in FFmpeg output
    return frames_dir / f"frame_{idx:06d}.jpg"

def text_for_interval(segments: List[Dict[str, Any]], t0: float, t1: float) -> str:
    """Concat all Whisper texts whose (start,end) overlaps [t0,t1]."""
    parts = []
    for s in segments:
        s0, s1 = float(s["start"]), float(s["end"])
        if not (s1 < t0 or s0 > t1):  # overlap
            parts.append(s.get("text", "").strip())
    return " ".join(parts).strip()

def build_manifest_for_video(
    video_dir: Path,
    fps_sampling: float,
    segment_sec: float,
    manifest_rows: List[Dict[str, Any]]
):
    frames_dir = video_dir / "frames"
    audio_path = video_dir / "audio.wav"
    transcript_json = video_dir / "transcript.json"
    keypoints_dir = video_dir / "keypoints"

    duration = video_duration_seconds(video_dir / "video.mp4")
    if duration <= 0:
        print(f"[WARN] Could not read duration for {video_dir.name}, skipping manifest build.")
        return

    segments = load_whisper_segments(transcript_json)
    n_segments = math.ceil(duration / segment_sec)

    for i in range(n_segments):
        t0 = i * segment_sec
        t1 = min((i + 1) * segment_sec, duration)

        frame_path = nearest_frame_for_time(frames_dir, fps_sampling, (t0 + t1) / 2.0)
        frame_exists = frame_path.exists()
        keypoints_path = (video_dir / "keypoints" / f"{frame_path.stem}.json") if frame_exists else None
        keypoints_exists = keypoints_path and keypoints_path.exists()

        row = {
            "video_id": video_dir.name,
            "t_start": round(t0, 3),
            "t_end": round(t1, 3),
            "frame_path": str(frame_path) if frame_exists else "",
            "audio_path": str(audio_path),
            "text_segment": text_for_interval(segments, t0, t1),
            "keypoints_path": str(keypoints_path) if keypoints_exists else "",
        }
        manifest_rows.append(row)

# -------------------------
# Orchestration
# -------------------------

def process_video(
    video_path: Path,
    out_root: Path,
    fps_sampling: float,
    whisper_model: str,
    device: str
):
    vid_slug = slugify(video_path.stem)
    video_dir = out_root / vid_slug
    ensure_dir(video_dir)

    # Copy or link original video
    dst_video = video_dir / "video.mp4"
    if not dst_video.exists():
        # Hard link if possible, else copy
        try:
            os.link(video_path, dst_video)
        except Exception:
            import shutil
            shutil.copy2(video_path, dst_video)

    frames_dir = video_dir / "frames"
    audio_path = video_dir / "audio.wav"
    transcript_json = video_dir / "transcript.json"
    keypoints_dir = video_dir / "keypoints"

    # 1) Frames
    if not frames_dir.exists() or not any(frames_dir.glob("*.jpg")):
        print(f"[Frames] {video_path.name}")
        extract_frames_ffmpeg(dst_video, frames_dir, fps_sampling)
    else:
        print(f"[Frames] Skipped (already exists) for {video_path.name}")

    # 2) Audio
    if not audio_path.exists():
        print(f"[Audio] {video_path.name}")
        extract_audio_ffmpeg(dst_video, audio_path)
    else:
        print(f"[Audio] Skipped (already exists) for {video_path.name}")

    # 3) Whisper
    if not transcript_json.exists():
        print(f"[Whisper] {video_path.name}")
        transcribe_whisper(audio_path, transcript_json, whisper_model=whisper_model, device=device)
    else:
        print(f"[Whisper] Skipped (already exists) for {video_path.name}")

    # 4) Pose Keypoints
    if not keypoints_dir.exists() or not any(keypoints_dir.glob("*.json")):
        print(f"[Pose] {video_path.name}")
        extract_pose_on_frames(frames_dir, keypoints_dir)
    else:
        print(f"[Pose] Skipped (already exists) for {video_path.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with raw videos (mp4, mov, mkv...).")
    ap.add_argument("--output_dir", type=str, required=True, help="Where to write processed data.")
    ap.add_argument("--fps", type=float, default=0.5, help="Sampling FPS for frames, e.g., 0.5 = 1 frame every 2s.")
    ap.add_argument("--segment_sec", type=float, default=2.0, help="Manifest time grid (seconds).")
    ap.add_argument("--whisper_model", type=str, default="large", help="Whisper model size (tiny/base/small/medium/large).")
    ap.add_argument("--device", type=str, default=None, help="Whisper device, e.g., 'cuda' if you have a GPU.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_root  = Path(args.output_dir)
    ensure_dir(out_root)

    video_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".mp4", ".mov", ".mkv", ".m4v"]])
    if not video_files:
        print(f"No videos found in {input_dir}.")
        return

    # Process each video
    for vp in video_files:
        print(f"\n=== Processing: {vp.name} ===")
        process_video(
            video_path=vp,
            out_root=out_root,
            fps_sampling=args.fps,
            whisper_model=args.whisper_model,
            device=args.device
        )

    # Build a single manifest over all videos
    manifest_rows: List[Dict[str, Any]] = []
    for video_dir in sorted(out_root.iterdir()):
        if not video_dir.is_dir():
            continue
        if not (video_dir / "video.mp4").exists():
            continue
        build_manifest_for_video(
            video_dir=video_dir,
            fps_sampling=args.fps,
            segment_sec=args.segment_sec,
            manifest_rows=manifest_rows
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_root / "manifest.csv", index=False)
    print(f"\nDone. Manifest saved to: {out_root / 'manifest.csv'}")
    print(f"Processed videos: {len([d for d in out_root.iterdir() if d.is_dir()])}")


###### Path configuration

from pathlib import Path

# ⬇️ EDIT THESE ⬇️
INPUT_DIR = Path("input_20240105")   # Folder with .mp4/.mov/.mkv/.m4v
OUTPUT_DIR = Path("output_20240105")          # Output folder

# Sampling & segmentation (defaults match the original CLI)
FPS = 0.5             # 0.5 => 1 frame every 2 seconds
SEGMENT_SEC = 2.0     # manifest time grid (seconds)
WHISPER_MODEL = "large"  # tiny/base/small/medium/large
DEVICE = None         # Use "cuda" for GPU if available, otherwise None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
print("DEVICE:", DEVICE)


from typing import Dict, Any, List

# Gather video files
video_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".mp4", ".mov", ".mkv", ".m4v"]])
if not video_files:
    print(f"No videos found in {INPUT_DIR}.")
else:
    for vp in video_files:
        print(f"\n=== Processing: {vp.name} ===")
        process_video(
            video_path=vp,
            out_root=OUTPUT_DIR,
            fps_sampling=FPS,
            whisper_model=WHISPER_MODEL,
            device=DEVICE
        )

    # Build manifest across all processed videos
    manifest_rows: List[Dict[str, Any]] = []
    for video_dir in sorted(OUTPUT_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        if not (video_dir / "video.mp4").exists():
            continue
        build_manifest_for_video(
            video_dir=video_dir,
            fps_sampling=FPS,
            segment_sec=SEGMENT_SEC,
            manifest_rows=manifest_rows
        )

    import pandas as pd
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nDone. Manifest saved to: {manifest_path}")
    print(f"Processed videos: {len([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])}")
