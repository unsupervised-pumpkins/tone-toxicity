import json
from pathlib import Path
from tqdm import tqdm

AUDIO_ROOT = Path("data") / "audio_segments"
SEGMENTS_DIR = Path("data") / "segments_text"

INPUT_FILES = [
    SEGMENTS_DIR / "ben_shapiro_labeled.jsonl",
    SEGMENTS_DIR / "joe_rogan_labeled.jsonl",
    SEGMENTS_DIR / "jon_stewart_labeled.jsonl",
]

def make_audio_path(record) -> Path:
    speaker = record["speaker"]
    video_id = record["video_id"]
    start_ms = int(round(record["start"] * 1000))
    end_ms = int(round(record["end"]* 1000))
    file_name = f"{speaker}_{video_id}_{start_ms:09d}_{end_ms:09d}.wav"
    return AUDIO_ROOT /speaker /file_name
def process_file(in_path: Path, out_path: Path):
    print(f"input path == {in_path} output path == {out_path}")
    kept = 0
    missing = 0

    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in tqdm(f_in):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "toxicity" not in record:
                continue
            audio_path = make_audio_path(record)
            if not audio_path.exists():
                missing += 1
                continue
            record["audio_path"] = str(audio_path)
            f_out.write(json.dumps(record) + "\n")
            kept += 1
    # print(f"Saved in {in_path.name}: kept == {kept} segments, missing == {missing} audio files")

def main():
    for in_path in INPUT_FILES:
        out_path = in_path.with_name(in_path.stem + "_audio.jsonl")
        process_file(in_path, out_path)

if __name__ == "__main__":
    main()
