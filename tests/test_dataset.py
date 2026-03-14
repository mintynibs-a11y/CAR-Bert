"""
tests/test_dataset.py — Unit tests for the dataset module.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import (
    ReviewDataset,
    build_datasets,
    build_inference_dataset,
    get_dataloader,
    load_csv,
)
from tests.conftest import requires_network


# ── Test data ─────────────────────────────────────────────────────────────────

SAMPLE_REVIEWS = [
    "外观非常漂亮，车身线条流畅，非常满意！",
    "车漆质量很差，容易刮花，失望透顶。",
    "整体外观中规中矩，没什么特别。",
    "颜值很高，街上回头率极高！",
    "保险杠做工粗糙，缝隙明显，不推荐。",
]
SAMPLE_LABELS = [1, 0, 2, 1, 0]


def _write_csv(tmp_dir: str, include_labels: bool = True) -> str:
    path = os.path.join(tmp_dir, "reviews.csv")
    data = {config.REVIEW_COLUMN: SAMPLE_REVIEWS}
    if include_labels:
        data[config.LABEL_COLUMN] = SAMPLE_LABELS
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ── Tests: load_csv ───────────────────────────────────────────────────────────

class TestLoadCsv:
    def test_loads_valid_file(self, tmp_path):
        path = _write_csv(str(tmp_path))
        df = load_csv(path)
        assert len(df) == len(SAMPLE_REVIEWS)
        assert config.REVIEW_COLUMN in df.columns

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/reviews.csv")

    def test_missing_review_column_raises(self, tmp_path):
        path = os.path.join(str(tmp_path), "bad.csv")
        pd.DataFrame({"text": SAMPLE_REVIEWS}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="review"):
            load_csv(path)

    def test_drops_nan_rows(self, tmp_path):
        path = os.path.join(str(tmp_path), "nan.csv")
        data = {config.REVIEW_COLUMN: SAMPLE_REVIEWS + [None]}
        pd.DataFrame(data).to_csv(path, index=False)
        df = load_csv(path)
        assert len(df) == len(SAMPLE_REVIEWS)


# ── Tests: ReviewDataset ──────────────────────────────────────────────────────

@requires_network
class TestReviewDataset:
    def test_len(self, bert_tokenizer):
        ds = ReviewDataset(SAMPLE_REVIEWS, SAMPLE_LABELS, bert_tokenizer, max_len=64)
        assert len(ds) == len(SAMPLE_REVIEWS)

    def test_item_keys_with_labels(self, bert_tokenizer):
        ds = ReviewDataset(SAMPLE_REVIEWS, SAMPLE_LABELS, bert_tokenizer, max_len=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_item_keys_without_labels(self, bert_tokenizer):
        ds = ReviewDataset(SAMPLE_REVIEWS, labels=None, tokenizer=bert_tokenizer, max_len=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" not in item

    def test_input_ids_shape(self, bert_tokenizer):
        max_len = 64
        ds = ReviewDataset(SAMPLE_REVIEWS, SAMPLE_LABELS, bert_tokenizer, max_len=max_len)
        item = ds[0]
        assert item["input_ids"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)

    def test_label_values(self, bert_tokenizer):
        ds = ReviewDataset(SAMPLE_REVIEWS, SAMPLE_LABELS, bert_tokenizer, max_len=64)
        for i, expected in enumerate(SAMPLE_LABELS):
            assert ds[i]["labels"].item() == expected


# ── Tests: build_datasets ─────────────────────────────────────────────────────

@requires_network
class TestBuildDatasets:
    def test_split_sizes(self, tmp_path, bert_tokenizer):
        path = _write_csv(str(tmp_path))
        train_ds, val_ds = build_datasets(path, bert_tokenizer, max_len=64, test_split=0.2, seed=42)
        assert len(train_ds) + len(val_ds) == len(SAMPLE_REVIEWS)
        assert len(val_ds) >= 1

    def test_missing_label_column_raises(self, tmp_path, bert_tokenizer):
        path = _write_csv(str(tmp_path), include_labels=False)
        with pytest.raises(ValueError, match="label"):
            build_datasets(path, bert_tokenizer)


# ── Tests: build_inference_dataset ───────────────────────────────────────────

@requires_network
class TestBuildInferenceDataset:
    def test_returns_correct_reviews(self, tmp_path, bert_tokenizer):
        path = _write_csv(str(tmp_path), include_labels=False)
        dataset, reviews = build_inference_dataset(path, bert_tokenizer, max_len=64)
        assert reviews == SAMPLE_REVIEWS
        assert len(dataset) == len(SAMPLE_REVIEWS)

    def test_no_labels_in_items(self, tmp_path, bert_tokenizer):
        path = _write_csv(str(tmp_path), include_labels=False)
        dataset, _ = build_inference_dataset(path, bert_tokenizer, max_len=64)
        item = dataset[0]
        assert "labels" not in item


# ── Tests: get_dataloader ─────────────────────────────────────────────────────

@requires_network
class TestGetDataloader:
    def test_dataloader_batch_size(self, bert_tokenizer):
        ds = ReviewDataset(SAMPLE_REVIEWS, SAMPLE_LABELS, bert_tokenizer, max_len=64)
        loader = get_dataloader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 2
