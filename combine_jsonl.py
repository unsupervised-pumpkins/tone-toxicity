import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path("data") / "segments_text"
INPUT_FILES = [
    DATA_DIR / "ben_shapiro_labeled.jsonl",
    DATA_DIR / "joe_rogan_labeled.jsonl",
    DATA_DIR / "jon_stewart_labeled.jsonl",
]

OUT_DIR = DATA_DIR / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1


def load_records():
    records = []
    for path in INPUT_FILES:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "toxicity" not in rec or "text" not in rec:
                    continue
                records.append(rec)
    return records


def write_jsonl(path: Path, records):
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    records = load_records()
    print(f"Loaded {len(records)} labeled segments")

    random.shuffle(records)

    n = len(records)
    n_train = int(TRAIN_FRACTION * n)
    n_val = int(VAL_FRACTION * n)
    n_test = n - n_train - n_val

    train_recs = records[:n_train]
    val_recs = records[n_train:n_train + n_val]
    test_recs = records[n_train + n_val:]

    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    write_jsonl(OUT_DIR / "train.jsonl", train_recs)
    write_jsonl(OUT_DIR / "val.jsonl",   val_recs)
    write_jsonl(OUT_DIR / "test.jsonl",  test_recs)


if __name__ == "__main__":
    main()
