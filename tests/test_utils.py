"""
tests/test_utils.py — Unit tests for the utility functions.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import compute_accuracy, format_results_table, set_seed


class TestSetSeed:
    def test_runs_without_error(self):
        set_seed(42)

    def test_produces_reproducible_results(self):
        import random
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b


class TestComputeAccuracy:
    def test_all_correct(self):
        preds = np.array([0, 1, 2, 1, 0])
        labels = np.array([0, 1, 2, 1, 0])
        assert compute_accuracy(preds, labels) == 1.0

    def test_all_wrong(self):
        preds = np.array([1, 0, 0])
        labels = np.array([0, 1, 1])
        assert compute_accuracy(preds, labels) == 0.0

    def test_partial_correct(self):
        preds = np.array([0, 1, 2, 1])
        labels = np.array([0, 0, 2, 1])
        assert compute_accuracy(preds, labels) == pytest.approx(0.75)


class TestFormatResultsTable:
    def test_returns_string(self):
        reviews = ["评价一", "评价二"]
        labels = [1, 0]
        label_map = {0: "负面 (Negative)", 1: "正面 (Positive)", 2: "中性 (Neutral)"}
        result = format_results_table(reviews, labels, label_map)
        assert isinstance(result, str)

    def test_contains_sentiment_labels(self):
        reviews = ["外观很好看", "车漆很差"]
        labels = [1, 0]
        label_map = {0: "负面 (Negative)", 1: "正面 (Positive)", 2: "中性 (Neutral)"}
        result = format_results_table(reviews, labels, label_map)
        assert "正面 (Positive)" in result
        assert "负面 (Negative)" in result

    def test_truncates_long_reviews(self):
        long_review = "外" * 60
        reviews = [long_review]
        labels = [1]
        label_map = {1: "正面 (Positive)"}
        result = format_results_table(reviews, labels, label_map)
        assert "..." in result

    def test_empty_input(self):
        result = format_results_table([], [], {})
        assert isinstance(result, str)
