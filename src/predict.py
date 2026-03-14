"""
src/predict.py — Run sentiment inference on car exterior reviews using a fine-tuned BERT model.

Usage (via main.py):
    python main.py predict --data data/reviews_to_classify.csv --model saved_model/

Or directly:
    python src/predict.py --data data/reviews_to_classify.csv --model saved_model/

You can also pass review text directly on the command line:
    python main.py predict --text "车漆质量很好，颜色漂亮，非常满意！"
"""

import argparse
import os
import sys

import torch
from transformers import BertTokenizer

# Allow running as a top-level script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import ReviewDataset, build_inference_dataset, get_dataloader, load_csv
from src.model import SentimentClassifier, get_tokenizer
from src.utils import format_results_table, get_device, print_classification_report, set_seed


def predict_from_file(
    data_file: str,
    model_dir: str = config.MODEL_DIR,
    batch_size: int = config.EVAL_BATCH_SIZE,
    max_len: int = config.MAX_SEQ_LENGTH,
    output_file: str | None = None,
) -> list[dict]:
    """
    Run sentiment prediction on all reviews in *data_file*.

    Args:
        data_file:   Path to CSV with a ``review`` column.
        model_dir:   Directory containing the fine-tuned BERT model.
        batch_size:  Inference batch size.
        max_len:     Maximum token sequence length.
        output_file: If provided, write results CSV to this path.

    Returns:
        List of dicts with keys ``review``, ``label``, ``sentiment``, ``confidence``.
    """
    set_seed(config.RANDOM_SEED)
    device = get_device()

    tokenizer = _load_tokenizer(model_dir)
    model = _load_model(model_dir, device)

    dataset, reviews = build_inference_dataset(data_file, tokenizer, max_len=max_len)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    all_labels, all_confidences = _run_inference(model, loader, device)

    # Check if the source file has true labels for evaluation
    df = load_csv(data_file)
    true_labels = None
    if config.LABEL_COLUMN in df.columns:
        true_labels = df[config.LABEL_COLUMN].astype(int).tolist()

    results = _build_results(reviews, all_labels, all_confidences)
    _print_results(results, true_labels)

    if output_file:
        _save_results(results, output_file)

    return results


def predict_texts(
    texts: list[str],
    model_dir: str = config.MODEL_DIR,
    batch_size: int = config.EVAL_BATCH_SIZE,
    max_len: int = config.MAX_SEQ_LENGTH,
) -> list[dict]:
    """
    Run sentiment prediction on a list of review strings.

    Returns:
        List of dicts with keys ``review``, ``label``, ``sentiment``, ``confidence``.
    """
    device = get_device()
    tokenizer = _load_tokenizer(model_dir)
    model = _load_model(model_dir, device)

    dataset = ReviewDataset(texts, labels=None, tokenizer=tokenizer, max_len=max_len)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    all_labels, all_confidences = _run_inference(model, loader, device)
    results = _build_results(texts, all_labels, all_confidences)
    _print_results(results)
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_tokenizer(model_dir: str) -> BertTokenizer:
    """Load tokenizer from *model_dir*, falling back to pretrained if absent."""
    if os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "tokenizer_config.json")):
        print(f"[INFO] Loading tokenizer from: {model_dir}")
        return BertTokenizer.from_pretrained(model_dir)
    print(f"[INFO] Loading default tokenizer: {config.PRETRAINED_MODEL_NAME}")
    return get_tokenizer()


def _load_model(model_dir: str, device: torch.device) -> SentimentClassifier:
    """Load a fine-tuned model, or raise with a helpful message."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: '{model_dir}'. "
            "Run training first: python main.py train"
        )
    print(f"[INFO] Loading model from: {model_dir}")
    model = SentimentClassifier.load(model_dir)
    model.to(device)
    model.eval()
    return model


def _run_inference(
    model: SentimentClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Return (predicted_labels, confidence_scores) for every sample in *loader*."""
    all_labels: list[int] = []
    all_confidences: list[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            all_labels.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    return all_labels, all_confidences


def _build_results(
    reviews: list[str],
    labels: list[int],
    confidences: list[float],
) -> list[dict]:
    return [
        {
            "review": review,
            "label": label,
            "sentiment": config.LABEL_MAP[label],
            "confidence": round(confidence, 4),
        }
        for review, label, confidence in zip(reviews, labels, confidences)
    ]


def _print_results(results: list[dict], true_labels: list[int] | None = None) -> None:
    reviews = [r["review"] for r in results]
    pred_labels = [r["label"] for r in results]

    print("\n" + format_results_table(reviews, pred_labels, config.LABEL_MAP))

    # Sentiment distribution
    from collections import Counter
    dist = Counter(r["sentiment"] for r in results)
    print("\n[情感分布 / Sentiment Distribution]")
    for sentiment, count in sorted(dist.items()):
        pct = count / len(results) * 100
        print(f"  {sentiment:<25} {count:>4} ({pct:.1f}%)")

    if true_labels is not None:
        print("\n[评估报告 / Evaluation Report]")
        print_classification_report(true_labels, pred_labels, config.LABEL_NAMES)


def _save_results(results: list[dict], output_file: str) -> None:
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Results saved to: {output_file}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment inference on car exterior reviews.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", help="Path to CSV file with a 'review' column")
    group.add_argument("--text", help="A single review string to classify")
    parser.add_argument("--model", default=config.MODEL_DIR, help="Directory of fine-tuned BERT model")
    parser.add_argument("--output", default=None, help="Path to save results CSV (optional)")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.text:
        predict_texts([args.text], model_dir=args.model)
    else:
        predict_from_file(args.data, model_dir=args.model, output_file=args.output, batch_size=args.batch_size)
