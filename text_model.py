import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "roberta-base"
BATCH_SIZE = 64
MAX_LENGTH = 256
EPOCHS = 200
LEARNING_RATE = 2e-5
NUM_CLASSES = 5

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

class ToxicityTextDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer, max_length: int = 256):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.records = []
        with self.jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "toxicity" not in rec or "text" not in rec:
                    continue
                self.records.append(rec)

        print(f"Loaded {len(self.records)} examples from {self.jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text"]
        tox_score = float(rec["toxicity"])
        label_id = bin_toxicity(tox_score)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
            "toxicity": torch.tensor(tox_score, dtype=torch.float),  # NEW
        }


def make_dataloaders():
    data_root = Path("data") / "segments_text" / "splits"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = ToxicityTextDataset(data_root / "train.jsonl", tokenizer, MAX_LENGTH)
    val_ds = ToxicityTextDataset(data_root / "val.jsonl",   tokenizer, MAX_LENGTH)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl= DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    return tokenizer, train_dl, val_dl


def forward_step(model, batch, device, loss_fn):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    logits = outputs.logits
    loss = loss_fn(logits, labels)
    return loss, logits, labels


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

    tokenizer, train_dl, val_dl = make_dataloaders()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

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
            f"training loss={avg_train_loss:.4f}, training accuracy={train_acc:.4f} | "
            f"validation loss={val_loss:.4f}, validation accuracy={val_acc:.4f}, "
            f"validation mae={val_mae:.4f}"
        )

    out_dir = Path("models") / "text_baseline_roberta_cls"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Smodel Path: {out_dir}")


if __name__ == "__main__":
    main()
