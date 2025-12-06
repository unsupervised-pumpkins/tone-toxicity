import json
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

MODEL_NAME = "facebook/wav2vec2-base"
NUM_CLASSES = 5 # not tunable
SAMPLE_RATE = 16_000 # not tunable
MAX_DURATION = 15.00
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5


BIN_MIDPOINTS = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

def logits_to_continuous(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    mids = BIN_MIDPOINTS.to(probs.device)
    return (probs * mids).sum(dim=-1)


def plot_curves(history, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history["val_mae"], label="val MAE")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_val_mae_curve.png")
    plt.close()


def bin_toxicity(score: float) -> int:

    if score < 0.2:
        return 0
    elif score < 0.4:
        return 1
    elif score < 0.6:
        return 2
    elif score < 0.8:
        return 3
    else:
        return 4

class ToxicityAudioDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        feature_extractor,
        sample_rate: int = SAMPLE_RATE,
        max_duration: float = MAX_DURATION,
        audio_key: str = "audio_path",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.audio_key = audio_key

        self.records = []
        with self.jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "toxicity" not in rec or audio_key not in rec:
                    continue
                self.records.append(rec)

        print(f"Loaded {len(self.records)} audio examples from {self.jsonl_path}")

    def __len__(self):
        return len(self.records)

    def _load_waveform(self, path: Path) -> torch.Tensor:
        wav, i = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if i != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=i, new_freq=self.sample_rate
            )

        if wav.shape[1] > self.max_samples:
            wav = wav[:, : self.max_samples]
        else:
            pad_len = self.max_samples - wav.shape[1]
            if pad_len > 0:
                wav = torch.nn.functional.pad(wav, (0, pad_len))

        wav = wav.squeeze(0)
        return wav

    def __getitem__(self, idx):
        rec = self.records[idx]
        audio_path = Path(rec[self.audio_key])
        tox_score = float(rec["toxicity"])
        label_id = bin_toxicity(tox_score)

        wav = self._load_waveform(audio_path)

        # enc = self.feature_extractor(
        #     wav,
        #     sampling_rate=self.sample_rate,
        #     padding="max_length",
        #     max_length=self.max_samples,
        #     return_tensors="pt",
        # )
        #
        #
        # return {
        #     "input_values": enc["input_values"].squeeze(0),
        #     "attention_mask": enc["attention_mask"].squeeze(0),
        #     "labels": torch.tensor(label_id, dtype=torch.long),
        # }

        enc = self.feature_extractor(
            wav,
            sampling_rate=self.sample_rate,
            padding="max_length",
            max_length=self.max_samples,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_values = enc["input_values"].squeeze(0)

        if "attention_mask" in enc:
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "toxicity_score": torch.tensor(tox_score, dtype=torch.float),
        }


def make_dataloaders():
    splits_root = Path("data") / "segments_text" / "audio_splits"

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    train_ds = ToxicityAudioDataset(
        splits_root / "train.jsonl",
        feature_extractor=feature_extractor,
        sample_rate=SAMPLE_RATE,
        max_duration=MAX_DURATION,
    )
    val_ds = ToxicityAudioDataset(
        splits_root / "val.jsonl",
        feature_extractor=feature_extractor,
        sample_rate=SAMPLE_RATE,
        max_duration=MAX_DURATION,
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return feature_extractor, train_dl, val_dl


def forward_step(model, batch, device, loss_fn):
    input_values = batch["input_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels,
    )
    logits = outputs.logits
    loss = outputs.loss
    return loss, logits, labels


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n_examples = 0
    n_correct = 0

    total_abs_err = 0.0 # for mae



    with torch.no_grad():
        for batch in dataloader:
            loss, logits, labels = forward_step(model, batch, device, loss_fn)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_examples += batch_size

            preds = torch.argmax(logits, dim=-1)
            n_correct += (preds == labels).sum().item()

            true_scores = batch["toxicity_score"].to(device)
            pred_scores = logits_to_continuous(logits)
            total_abs_err += torch.abs(pred_scores - true_scores).sum().item()

    avg_loss = total_loss / max(1, n_examples)
    accuracy = n_correct / max(1, n_examples)
    val_mae = total_abs_err / max(1, n_examples)

    return avg_loss, accuracy, val_mae


def main():

    import os
    import random
    import numpy as np
    import torch

    def set_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    print("HYPERS:", MAX_DURATION, BATCH_SIZE, EPOCHS, LEARNING_RATE)
    print("torch.initial_seed():", torch.initial_seed())
    print("python random:", random.random())
    print("numpy random:", np.random.rand())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    feature_extractor, train_dl, val_dl = make_dataloaders()

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        n_train_examples = 0
        n_train_correct = 0

        for batch in train_dl:
            optimizer.zero_grad()
            loss, logits, labels = forward_step(model, batch, device, loss_fn)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_train_loss += loss.item() * batch_size
            n_train_examples += batch_size

            preds = torch.argmax(logits, dim=-1)
            n_train_correct += (preds == labels).sum().item()

        avg_train_loss = total_train_loss / max(1, n_train_examples)
        train_acc = n_train_correct / max(1, n_train_examples)
        val_loss, val_acc, val_mae = evaluate(model, val_dl, device, loss_fn)


        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train loss={avg_train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}, val MAE={val_mae:.4f}"
        )



    out_dir = Path("models") / "audio_wav2vec2_cls"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    feature_extractor.save_pretrained(out_dir)
    plot_curves(history, out_dir, prefix="audio")
    print(f"Smodel Path: {out_dir}")


if __name__ == "__main__":
    main()
