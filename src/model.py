"""
src/model.py — BERT-based sentiment classifier for car exterior reviews.

Wraps HuggingFace's BertForSequenceClassification with convenience helpers
for saving/loading fine-tuned weights.
"""

import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer

import config


class SentimentClassifier(nn.Module):
    """
    Fine-tuned BERT model for sentiment polarity classification.

    Labels:
        0 — 负面 (Negative)
        1 — 正面 (Positive)
        2 — 中性 (Neutral)
    """

    def __init__(self, pretrained_model_name: str = config.PRETRAINED_MODEL_NAME, num_labels: int = config.NUM_LABELS):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        """
        Forward pass.

        Returns a HuggingFace SequenceClassifierOutput object.
        When *labels* is supplied the output includes the cross-entropy loss.
        """
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, save_dir: str) -> None:
        """Save model weights to *save_dir*."""
        os.makedirs(save_dir, exist_ok=True)
        self.bert.save_pretrained(save_dir)

    @classmethod
    def load(cls, save_dir: str, num_labels: int = config.NUM_LABELS) -> "SentimentClassifier":
        """Load a previously saved fine-tuned model from *save_dir*."""
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Model directory not found: {save_dir}")
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.bert = BertForSequenceClassification.from_pretrained(save_dir, num_labels=num_labels)
        return instance


def get_tokenizer(model_name: str = config.PRETRAINED_MODEL_NAME) -> BertTokenizer:
    """Load the BERT tokenizer that matches the model."""
    return BertTokenizer.from_pretrained(model_name)
