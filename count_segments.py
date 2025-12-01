# Note:
# Run "python count_segments.py data/segments_text/*" in your console.
# Expected Output:
# data/segments_text/ben_shapiro.jsonl: 1106 segments
# data/segments_text/joe_rogan.jsonl: 2006 segments
# data/segments_text/jon_stewart.jsonl: 769 segments
# TOTAL: 3881 segments

def count_segments(path):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def main():
    files = [
        "data/segments_text/ben_shapiro.jsonl",
        "data/segments_text/joe_rogan.jsonl",
        "data/segments_text/jon_stewart.jsonl",
    ]

    total = 0
    for p in files:
        n = count_segments(p)
        total += n
        print(f"{p}: {n} segments")
    print(f"TOTAL: {total} segments")

if __name__ == "__main__":
    main()