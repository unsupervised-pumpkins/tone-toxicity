import json
from pathlib import Path
import subprocess

#  https://github.com/yt-dlp/yt-dlp
#  https://github.com/yt-dlp/yt-dlp#output-template


DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "segments_text"
AUDIO_RAW_DIR = DATA_DIR / "audio_raw"
AUDIO_RAW_DIR.mkdir(parents=True, exist_ok=True)

JSONL_FILES = [
    TEXT_DIR / "ben_shapiro.jsonl",
    TEXT_DIR / "joe_rogan.jsonl",
    TEXT_DIR / "jon_stewart.jsonl",
]

def collect_video_ids(jsonl_files):
    video_ids = set()
    for path in jsonl_files:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                video_ids.add(rec["video_id"])
    return sorted(video_ids)

def download_audio_for_video(video_id):

    out_template = str(AUDIO_RAW_DIR / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-o", out_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]

    print("[AUDIO] Downloading", video_id)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] yt-dlp failed for {video_id}: {e}")

def main():
    video_ids = collect_video_ids(JSONL_FILES)
    print(f"Found {len(video_ids)} unique video_ids")

    for vid in video_ids:
        existing = list(AUDIO_RAW_DIR.glob(f"{vid}.*"))
        if existing:
            print(f"[SKIP] Already have audio for {vid}: {existing[0].name}")
            continue

        download_audio_for_video(vid)

if __name__ == "__main__":
    main()
