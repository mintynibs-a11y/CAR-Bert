"""
src/train.py — Fine-tune BERT on labeled car exterior reviews.

Usage (via main.py):
    python main.py train --data data/sample_reviews.csv

Or directly:
    python src/train.py --data data/sample_reviews.csv
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Allow running as a top-level script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import build_datasets, get_dataloader
from src.model import SentimentClassifier, get_tokenizer
from src.utils import compute_accuracy, get_device, print_classification_report, set_seed


def train(
    data_file: str = config.DEFAULT_DATA_FILE,
    save_dir: str = config.MODEL_DIR,
    num_epochs: int = config.NUM_EPOCHS,
    train_batch_size: int = config.TRAIN_BATCH_SIZE,
    eval_batch_size: int = config.EVAL_BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    warmup_ratio: float = config.WARMUP_RATIO,
    max_len: int = config.MAX_SEQ_LENGTH,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> None:
    """Fine-tune BERT and save the best checkpoint to *save_dir*."""

    set_seed(seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # ── Tokenizer & datasets ──────────────────────────────────────────────────
    print(f"[INFO] Loading tokenizer: {config.PRETRAINED_MODEL_NAME}")
    tokenizer = get_tokenizer()
    tokenizer.save_pretrained(save_dir)

    print(f"[INFO] Loading data from: {data_file}")
    train_ds, val_ds = build_datasets(data_file, tokenizer, max_len=max_len, test_split=test_split, seed=seed)
    print(f"[INFO] Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    train_loader = get_dataloader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=eval_batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Initialising model: {config.PRETRAINED_MODEL_NAME}")
    model = SentimentClassifier()
    model.to(device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # ─ Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % max(1, len(train_loader) // 5) == 0:
                print(
                    f"  Epoch {epoch}/{num_epochs} | Step {step}/{len(train_loader)} "
                    f"| Loss: {total_loss / step:.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Average training loss: {avg_train_loss:.4f}")

        # ─ Evaluate ──────────────────────────────────────────────────────────
        val_acc, all_preds, all_labels = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] Validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(save_dir)
            print(f"[Epoch {epoch}] ✓ Best model saved to: {save_dir}")

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print("[INFO] Final validation classification report:")
    print_classification_report(all_labels, all_preds, config.LABEL_NAMES)


def evaluate(
    model: SentimentClassifier,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Run evaluation and return (accuracy, predictions, true_labels)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = compute_accuracy(np.array(all_preds), np.array(all_labels))
    return acc, all_preds, all_labels


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for car exterior review sentiment analysis.")
    parser.add_argument("--data", default=config.DEFAULT_DATA_FILE, help="Path to labeled CSV file")
    parser.add_argument("--save-dir", default=config.MODEL_DIR, help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_file=args.data,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
