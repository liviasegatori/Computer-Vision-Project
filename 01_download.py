"""
Data Acquisition Module
-----------------------
This script automates the retrieval of high-fidelity video data from YouTube
using 'yt-dlp'. It ensures that the Federal Reserve press conferences are
downloaded in the best available audio/video format for analysis.

Requirement: FFmpeg must be installed on the system to merge audio/video streams.
"""

# pip install yt-dlp

from pathlib import Path
from yt_dlp import YoutubeDL

# ================= CONFIGURATION =================
# PASTE URL(S) BELOW inside the quotes. 
URLS_TO_DOWNLOAD = [
    "https://www.youtube.com/watch?v=iyFc9ytZ_Rk",
]

OUT_DIR = Path("downloads")
# =================================================

def download_with_ytdlp(urls: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(upload_date)s_%(title).80s.%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "ignoreerrors": True,
        "continuedl": True,
        "retries": 3,
        "quiet": False,
    }

    failures = []
    
    with YoutubeDL(ydl_opts) as ydl:
        for i, url in enumerate(urls, 1):
            # clean up whitespace just in case
            url = url.strip()
            if not url: continue

            print(f"\n[{i}/{len(urls)}] Processing: {url}")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"Failed: {url} :: {e}")
                failures.append((url, str(e)))

# ---- run it ----
if __name__ == "__main__":
    download_with_ytdlp(URLS_TO_DOWNLOAD, OUT_DIR)