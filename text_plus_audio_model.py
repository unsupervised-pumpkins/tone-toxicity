import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModel,
)

TEXT_MODEL_NAME = "roberta-base"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"
NUM_CLASSES = 5
SAMPLE_RATE = 16_000
MAX_DURATION = 15.0
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_TEXT_LENGTH = 256

BIN_MIDPOINTS = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

def logits_to_continuous(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    mids = BIN_MIDPOINTS.to(probs.device)
    return (probs * mids).sum(dim=-1)


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

class ToxicityMultimodalDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        feature_extractor,
        sample_rate: int = SAMPLE_RATE,
        max_duration: float = MAX_DURATION,
        audio_key: str = "audio_path",
        text_key: str = "text",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.audio_key = audio_key
        self.text_key = text_key

        self.records = []
        with self.jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "toxicity" not in rec or audio_key not in rec or text_key not in rec:
                    continue
                self.records.append(rec)

        print(f"Loaded {len(self.records)} multimodal examples from {self.jsonl_path}")

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

        text = rec[self.text_key]



        text_enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        input_ids = text_enc["input_ids"].squeeze(0)
        text_attention_mask = text_enc["attention_mask"].squeeze(0)


        audio_enc = self.feature_extractor(
            wav,
            sampling_rate=self.sample_rate,
            padding="max_length",
            max_length=self.max_samples,
            return_tensors="pt",
            return_attention_mask=True,
        )

        audio_input_values = audio_enc["input_values"].squeeze(0)

        if "attention_mask" in audio_enc:
            audio_attention_mask = audio_enc["attention_mask"].squeeze(0)
        else:
            audio_attention_mask = torch.ones_like(audio_input_values, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "text_attention_mask": text_attention_mask,
            "audio_input_values": audio_input_values,
            "audio_attention_mask": audio_attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "toxicity": torch.tensor(tox_score, dtype=torch.float),  # NEW
        }


class TextAudioToxicityModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = TEXT_MODEL_NAME,
        audio_model_name: str = AUDIO_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.audio_model = AutoModel.from_pretrained(audio_model_name)

        text_hidden_size = self.text_model.config.hidden_size
        audio_hidden_size = self.audio_model.config.hidden_size

        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(text_hidden_size + audio_hidden_size, num_classes)

    def forward(
        self,
        input_ids,
        text_attention_mask,
        audio_input_values,
        audio_attention_mask,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=text_attention_mask,
        )
        text_hidden = text_outputs.last_hidden_state
        text_cls = text_hidden[:, 0, :]

        # audio_outputs = self.audio_model(
        #     input_values=audio_input_values,
        #     attention_mask=audio_attention_mask,
        # )
        # audio_hidden = audio_outputs.last_hidden_state
        #
        # mask = audio_attention_mask.unsqueeze(-1).type_as(audio_hidden)
        # masked_hidden = audio_hidden * mask
        # sum_hidden = masked_hidden.sum(dim=1)
        # lengths = mask.sum(dim=1).clamp(min=1e-6)
        # audio_pooled = sum_hidden / lengths

        audio_outputs = self.audio_model(
            input_values=audio_input_values,
            attention_mask=audio_attention_mask,
        )
        audio_hidden = audio_outputs.last_hidden_state

        audio_pooled = audio_hidden.mean(dim=1)

        fused = torch.cat([text_cls, audio_pooled], dim=-1)
        # --------- if we want to add dropout --------
        # fused = self.dropout(fused)
        logits = self.classifier(fused)

        return logits


def make_dataloaders():
    splits_root = Path("data") / "segments_text" / "audio_splits"

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

    train_ds = ToxicityMultimodalDataset(
        splits_root / "train.jsonl",
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
    )
    val_ds = ToxicityMultimodalDataset(
        splits_root / "val.jsonl",
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return tokenizer, feature_extractor, train_dl, val_dl


def forward_step(model, batch, device, loss_fn):
    input_ids = batch["input_ids"].to(device)
    text_attention_mask = batch["text_attention_mask"].to(device)
    audio_input_values = batch["audio_input_values"].to(device)
    audio_attention_mask = batch["audio_attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        text_attention_mask=text_attention_mask,
        audio_input_values=audio_input_values,
        audio_attention_mask=audio_attention_mask,
    )
    loss = loss_fn(outputs, labels)
    return loss, outputs, labels


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n_examples = 0
    n_correct = 0
    total_abs_err = 0.0

    with torch.no_grad():
        for batch in dataloader:
            loss, logits, labels = forward_step(model, batch, device, loss_fn)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_examples += batch_size

            preds = torch.argmax(logits, dim=-1)
            n_correct += (preds == labels).sum().item()

            pred_scores = logits_to_continuous(logits)
            true_scores = batch["toxicity"].to(device)
            total_abs_err += torch.abs(pred_scores - true_scores).sum().item()

    avg_loss = total_loss / max(1, n_examples)
    accuracy = n_correct / max(1, n_examples)
    mae = total_abs_err / max(1, n_examples)
    return avg_loss, accuracy, mae

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, _, train_dl, val_dl = make_dataloaders()

    model = TextAudioToxicityModel().to(device)
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
        # val_loss, val_acc = evaluate(model, val_dl, device, loss_fn)
        val_loss, val_acc, val_mae = evaluate(model, val_dl, device, loss_fn)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss={avg_train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}, mae={val_mae:.4f}"
        )

    out_dir = Path("models") / "multimodal_text_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Smodel Path: {out_dir}")



if __name__ == "__main__":
    main()
