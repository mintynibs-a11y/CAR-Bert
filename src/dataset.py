"""
src/dataset.py — PyTorch Dataset for car exterior review sentiment classification.

Loads reviews from a CSV file and tokenises them using a BERT tokeniser.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer

import config


class ReviewDataset(Dataset):
    """Dataset for BERT-based sentiment classification of car exterior reviews."""

    def __init__(self, reviews: list[str], labels: list[int] | None, tokenizer: BertTokenizer, max_len: int):
        """
        Args:
            reviews:   List of review strings.
            labels:    List of integer labels (0/1/2), or None for inference.
            tokenizer: HuggingFace BERT tokenizer.
            max_len:   Maximum token sequence length.
        """
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> dict:
        review = str(self.reviews[idx])
        encoding = self.tokenizer(
            review,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame, validating required columns."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path)
    if config.REVIEW_COLUMN not in df.columns:
        raise ValueError(
            f"CSV must contain a '{config.REVIEW_COLUMN}' column. "
            f"Found: {list(df.columns)}"
        )
    # Drop rows with missing review text
    df = df.dropna(subset=[config.REVIEW_COLUMN]).reset_index(drop=True)
    return df


def build_datasets(
    file_path: str,
    tokenizer: BertTokenizer,
    max_len: int = config.MAX_SEQ_LENGTH,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[Dataset, Dataset]:
    """
    Load labeled data from *file_path* and return (train_dataset, val_dataset).

    Raises ValueError if the CSV has no label column.
    """
    df = load_csv(file_path)
    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"CSV must contain a '{config.LABEL_COLUMN}' column for training. "
            f"For inference only, use build_inference_dataset()."
        )
    reviews = df[config.REVIEW_COLUMN].tolist()
    labels = df[config.LABEL_COLUMN].astype(int).tolist()

    dataset = ReviewDataset(reviews, labels, tokenizer, max_len)
    val_size = max(1, int(len(dataset) * test_split))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    return train_ds, val_ds


def build_inference_dataset(
    file_path: str,
    tokenizer: BertTokenizer,
    max_len: int = config.MAX_SEQ_LENGTH,
) -> tuple[Dataset, list[str]]:
    """
    Load unlabeled review data for inference.

    Returns:
        (dataset, reviews): dataset without labels, and original review strings.
    """
    df = load_csv(file_path)
    reviews = df[config.REVIEW_COLUMN].tolist()
    dataset = ReviewDataset(reviews, labels=None, tokenizer=tokenizer, max_len=max_len)
    return dataset, reviews


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Wrap a Dataset in a DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
