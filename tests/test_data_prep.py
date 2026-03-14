"""
tests/test_data_prep.py — Unit tests for the data preparation module.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_prep import load_xlsx, split_and_save, POSITIVE_LABEL, NEGATIVE_LABEL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_mock_xlsx(tmp_dir: str, n_rows: int = 10) -> tuple[str, list[str], list[str]]:
    """
    Create a minimal xlsx file that matches the expected paired-review layout.

    The first row becomes the pandas column headers; it must contain real review
    text in positions 1 and 3 so that ``load_xlsx`` can recover them.
    """
    positive_reviews = [f"满意评价 {i}" for i in range(n_rows + 1)]
    negative_reviews = [f"不满意评价 {i}" for i in range(n_rows + 1)]

    # Build header row (index 0) and data rows
    header = {
        "满意": "满意",
        positive_reviews[0]: positive_reviews[0],  # first positive review = column name
        "不满意": "不满意",
        negative_reviews[0]: negative_reviews[0],  # first negative review = column name
    }
    data = {
        "满意": ["满意"] * n_rows,
        positive_reviews[0]: positive_reviews[1:],
        "不满意": ["不满意"] * n_rows,
        negative_reviews[0]: negative_reviews[1:],
    }
    df = pd.DataFrame(data)
    path = os.path.join(tmp_dir, "mock_reviews.xlsx")
    df.to_excel(path, index=False)
    return path, positive_reviews, negative_reviews


# ── Tests: load_xlsx ──────────────────────────────────────────────────────────

class TestLoadXlsx:
    def test_returns_dataframe_with_required_columns(self, tmp_path):
        path, _, _ = _write_mock_xlsx(str(tmp_path))
        df = load_xlsx(path)
        assert config.REVIEW_COLUMN in df.columns
        assert config.LABEL_COLUMN in df.columns

    def test_correct_total_row_count(self, tmp_path):
        n = 10
        path, positives, negatives = _write_mock_xlsx(str(tmp_path), n_rows=n)
        df = load_xlsx(path)
        # n+1 positives + n+1 negatives (header row included)
        assert len(df) == 2 * (n + 1)

    def test_label_values(self, tmp_path):
        path, _, _ = _write_mock_xlsx(str(tmp_path))
        df = load_xlsx(path)
        unique_labels = set(df[config.LABEL_COLUMN].unique())
        assert unique_labels == {POSITIVE_LABEL, NEGATIVE_LABEL}

    def test_positive_negative_counts_equal(self, tmp_path):
        n = 10
        path, _, _ = _write_mock_xlsx(str(tmp_path), n_rows=n)
        df = load_xlsx(path)
        assert (df[config.LABEL_COLUMN] == POSITIVE_LABEL).sum() == n + 1
        assert (df[config.LABEL_COLUMN] == NEGATIVE_LABEL).sum() == n + 1

    def test_first_review_from_header_is_included(self, tmp_path):
        path, positives, negatives = _write_mock_xlsx(str(tmp_path))
        df = load_xlsx(path)
        all_reviews = df[config.REVIEW_COLUMN].tolist()
        assert positives[0] in all_reviews
        assert negatives[0] in all_reviews

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_xlsx("/nonexistent/path/reviews.xlsx")

    def test_too_few_columns_raises(self, tmp_path):
        path = os.path.join(str(tmp_path), "bad.xlsx")
        pd.DataFrame({"review": ["text"]}).to_excel(path, index=False)
        with pytest.raises(ValueError, match="4 columns"):
            load_xlsx(path)


# ── Tests: split_and_save ─────────────────────────────────────────────────────

class TestSplitAndSave:
    def _make_df(self, n: int = 100) -> pd.DataFrame:
        reviews = [f"评价 {i}" for i in range(n)]
        labels = [POSITIVE_LABEL if i % 2 == 0 else NEGATIVE_LABEL for i in range(n)]
        return pd.DataFrame({config.REVIEW_COLUMN: reviews, config.LABEL_COLUMN: labels})

    def test_creates_both_csv_files(self, tmp_path):
        df = self._make_df()
        train_path, test_path = split_and_save(df, output_dir=str(tmp_path), test_split=0.2)
        assert os.path.isfile(train_path)
        assert os.path.isfile(test_path)

    def test_total_rows_preserved(self, tmp_path):
        df = self._make_df(100)
        train_path, test_path = split_and_save(df, output_dir=str(tmp_path), test_split=0.2)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        assert len(train_df) + len(test_df) == len(df)

    def test_test_fraction_approximate(self, tmp_path):
        df = self._make_df(100)
        _, test_path = split_and_save(df, output_dir=str(tmp_path), test_split=0.2, seed=42)
        test_df = pd.read_csv(test_path)
        assert abs(len(test_df) - 20) <= 2  # allow ±2 due to stratification

    def test_output_has_required_columns(self, tmp_path):
        df = self._make_df()
        train_path, test_path = split_and_save(df, output_dir=str(tmp_path))
        for path in (train_path, test_path):
            loaded = pd.read_csv(path)
            assert config.REVIEW_COLUMN in loaded.columns
            assert config.LABEL_COLUMN in loaded.columns

    def test_reproducible_with_same_seed(self, tmp_path):
        df = self._make_df(100)
        train_a, _ = split_and_save(df, output_dir=str(tmp_path), seed=42,
                                    train_file=str(tmp_path / "train_a.csv"),
                                    test_file=str(tmp_path / "test_a.csv"))
        train_b, _ = split_and_save(df, output_dir=str(tmp_path), seed=42,
                                    train_file=str(tmp_path / "train_b.csv"),
                                    test_file=str(tmp_path / "test_b.csv"))
        df_a = pd.read_csv(train_a)
        df_b = pd.read_csv(train_b)
        assert list(df_a[config.REVIEW_COLUMN]) == list(df_b[config.REVIEW_COLUMN])
