import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

MODEL_NAME = "facebook/wav2vec2-base"
NUM_CLASSES = 5
SAMPLE_RATE = 16_000
MAX_DURATION = 15.0
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4


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
    )
    logits = outputs.logits
    loss = loss_fn(logits, labels)
    return loss, logits, labels


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n_examples = 0
    n_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, logits, labels = forward_step(model, batch, device, loss_fn)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_examples += batch_size

            preds = torch.argmax(logits, dim=-1)
            n_correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(1, n_examples)
    accuracy = n_correct / max(1, n_examples)
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    feature_extractor, train_dl, val_dl = make_dataloaders()

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

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

        val_loss, val_acc = evaluate(model, val_dl, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train loss={avg_train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

    out_dir = Path("models") / "audio_wav2vec2_cls"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    feature_extractor.save_pretrained(out_dir)
    print(f"Smodel Path: {out_dir}")


if __name__ == "__main__":
    main()
