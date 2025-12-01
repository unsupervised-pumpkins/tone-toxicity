import json
import csv

FILES = [
    "data/segments_text/ben_shapiro.jsonl",
    "data/segments_text/joe_rogan.jsonl",
    "data/segments_text/jon_stewart.jsonl",
]

OUT_PATH = "data/segments_for_labeling.csv"

fieldnames = ["segment_id", "speaker", "video_id", "start", "end", "text", "toxicity_true"]

with open(OUT_PATH, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()

    for path in FILES:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                segment_id = f"{obj['speaker']}_{obj['video_id']}_{idx:04d}"
                writer.writerow({
                    "segment_id": segment_id,
                    "speaker": obj["speaker"],
                    "video_id": obj["video_id"],
                    "start": obj["start"],
                    "end": obj["end"],
                    "text": obj["text"],
                    "toxicity_true": "",
                })
