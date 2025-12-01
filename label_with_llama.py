import json
import textwrap
from pathlib import Path

from tqdm import tqdm
from transformers import pipeline

import sys

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)


def llama_chat(user_content: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "user", "content": user_content},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    assistant_msg = outputs[0]["generated_text"][-1]["content"]
    return assistant_msg.strip()

def build_label_prompt(segment_text: str) -> str:
    prompt = f"""
    You are an expert annotator of rhetorical tone and toxic language.

    Given the following transcript segment, identify whether the language is toxic, insulting, demeaning, or harsh in tone.

    Return a SINGLE JSON object with the following fields:
    - "overall_toxicity": float between 0 and 1
    - "toxicity_level": one of ["none", "mild", "moderate", "severe"]
    - "toxic_spans": list of exact substrings from the input that are toxic or harsh
    - "has_sarcasm": boolean
    - "has_profanity": boolean
    - "notes": short free-text justification

    Use the full [0, 1] range. Even mildly rude or dismissive language should have
    an "overall_toxicity" of at least 0.1. Completely neutral, factual segments can be 0.0.

    Respond with ONLY the JSON object, no extra text.

    Transcript:
    \"\"\"{segment_text}\"\"\"
    """
    return textwrap.dedent(prompt).strip()




def parse_tox_score(raw: str) -> float:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(raw[start:end+1])

    if "toxicity" not in obj:
        raise ValueError(f"raise error -- no toxicirt {obj}")

    return float(obj["toxicity"])


def parse_label_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            return json.loads(candidate)
        raise

def label_segment(segment_text: str) -> float:
    prompt = build_label_prompt(segment_text)
    raw = llama_chat(prompt, max_new_tokens=256)
    obj = parse_label_json(raw)

    if "overall_toxicity" not in obj:
        raise ValueError(f"raise error -- no toxicirt")

    return float(obj["overall_toxicity"])



DATA_DIR = Path("data") / "segments_text"
INPUT_FILES = [
    DATA_DIR / "ben_shapiro.jsonl",
    DATA_DIR / "joe_rogan.jsonl",
    DATA_DIR / "jon_stewart.jsonl",
]


def label_file(input_path: Path, output_path: Path, max_examples=None):
    print(f"[INFO] Labelling {input_path} -> {output_path}")

    # Count total lines so tqdm can show "XX/total"
    with input_path.open() as f:
        total = sum(1 for _ in f)

    with input_path.open() as f_in, output_path.open("w") as f_out:
        for i, line in enumerate(tqdm(f_in, total=total, desc=input_path.name)):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if "toxicity" in rec:
                f_out.write(json.dumps(rec) + "\n")
                continue

            text = rec["text"]

            try:
                tox_score = label_segment(text)
            except Exception as e:
                print(f"[WARN] LLM failed on example {i}: {e}")
                continue

            rec["toxicity"] = tox_score
            f_out.write(json.dumps(rec) + "\n")

            if max_examples is not None and (i + 1) >= max_examples:
                break



def main():
    # If user passes a filename, only process that file
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])

        # If they passed a bare name like "ben_shapiro.jsonl",
        # look for it under DATA_DIR.
        if not arg.is_absolute() and not arg.exists():
            in_path = DATA_DIR / arg.name
        else:
            in_path = arg

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        out_path = in_path.with_name(in_path.stem + "_labeled.jsonl")
        label_file(in_path, out_path)
    else:
        # Fallback: original behavior – run all three files
        for in_path in INPUT_FILES:
            out_path = in_path.with_name(in_path.stem + "_labeled.jsonl")
            label_file(in_path, out_path)



if __name__ == "__main__":
    main()