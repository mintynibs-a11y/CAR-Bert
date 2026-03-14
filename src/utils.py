"""
src/utils.py — Shared utility functions.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Return fraction of correct predictions."""
    return float(np.sum(preds == labels)) / len(labels)


def format_results_table(reviews: list[str], predicted_labels: list[int], label_map: dict) -> str:
    """
    Format prediction results as a human-readable text table.

    Args:
        reviews:          List of review strings.
        predicted_labels: List of predicted integer labels.
        label_map:        Mapping from integer label to display string.

    Returns:
        Formatted string table.
    """
    sep = "─" * 80
    lines = [sep, f"{'No.':<5} {'Sentiment':<20} Review", sep]
    for i, (review, label) in enumerate(zip(reviews, predicted_labels), start=1):
        sentiment = label_map.get(label, str(label))
        # Truncate long reviews for display
        display = review if len(review) <= 50 else review[:47] + "..."
        lines.append(f"{i:<5} {sentiment:<20} {display}")
    lines.append(sep)
    return "\n".join(lines)


def print_classification_report(true_labels: list[int], pred_labels: list[int], label_names: list[str]) -> None:
    """Print a per-class classification report (requires scikit-learn)."""
    try:
        from sklearn.metrics import classification_report
        print(classification_report(true_labels, pred_labels, target_names=label_names, zero_division=0))
    except ImportError:
        # Fallback: simple accuracy
        acc = compute_accuracy(np.array(pred_labels), np.array(true_labels))
        print(f"Accuracy: {acc:.4f}")
