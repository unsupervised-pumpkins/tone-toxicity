# Please run make_label_csv.py first to generate segments_for_labelling.csv file.

# Usage in CLI:
# python sample_segments.py -- per_speaker 100

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_speaker", type=int, default=100)
    parser.add_argument("--input", default="data/segments_for_labeling.csv")
    parser.add_argument("--output", default="data/segments_for_labeling_sampled.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    def sample_group(g):
        n = min(len(g), args.per_speaker)
        return g.sample(n=n, random_state=args.seed)

    sampled = df.groupby("speaker", group_keys=False).apply(sample_group)
    sampled.to_csv(args.output, index=False)
    print(f"Saved {len(sampled)} rows to {args.output}")

if __name__ == "__main__":
    main()