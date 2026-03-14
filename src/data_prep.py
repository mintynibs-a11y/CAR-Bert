"""
src/data_prep.py — Utilities for converting and splitting raw review data.

Supports reading the collected Excel dataset with paired positive/negative
reviews and splitting it into train and test CSV files ready for training.

Expected Excel layout (as collected by the scraper):
    Column 0: sentiment category for positive reviews ("满意" / "好评")
    Column 1: positive review text
    Column 2: sentiment category for negative reviews ("不满意" / "槽点")
    Column 3: negative review text

Each row of the file provides one positive review and one negative review.
Because pandas treats the first spreadsheet row as column headers, the text
that appears in the header position for columns 1 and 3 is itself a valid
review and is included in the output.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

import config


# ── Constants ──────────────────────────────────────────────────────────────────

POSITIVE_LABEL = 1   # 正面 / Positive
NEGATIVE_LABEL = 0   # 负面 / Negative

DEFAULT_XLSX_FILE = os.path.join(config.BASE_DIR, "满意以及不满意评论全部.xlsx")
DEFAULT_TRAIN_FILE = os.path.join(config.DATA_DIR, "train.csv")
DEFAULT_TEST_FILE = os.path.join(config.DATA_DIR, "test.csv")


# ── Public API ─────────────────────────────────────────────────────────────────

def load_xlsx(file_path: str) -> pd.DataFrame:
    """
    Read the paired positive/negative review Excel file and return a tidy
    DataFrame with columns ``review`` and ``label``.

    The first spreadsheet row is used as pandas column headers, so its review
    text (columns 1 and 3) is recovered from the column names and included in
    the output.

    Args:
        file_path: Path to the ``.xlsx`` file.

    Returns:
        DataFrame with columns ``review`` (str) and ``label`` (int).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file does not have the expected four-column layout.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    df_raw = pd.read_excel(file_path, engine="openpyxl")

    if df_raw.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 columns in '{file_path}', "
            f"got {df_raw.shape[1]}. "
            "The file should have paired positive/negative review columns."
        )

    col_names = list(df_raw.columns)

    # The first row of the spreadsheet was consumed as column headers.
    # Columns 1 and 3 are review texts, so recover them from the column names.
    first_positive = str(col_names[1])
    first_negative = str(col_names[3])

    # Collect positive reviews from the data rows (column 1)
    positive_reviews = [first_positive] + df_raw.iloc[:, 1].dropna().astype(str).tolist()

    # Collect negative reviews from the data rows (column 3)
    negative_reviews = [first_negative] + df_raw.iloc[:, 3].dropna().astype(str).tolist()

    records = (
        [{config.REVIEW_COLUMN: r, config.LABEL_COLUMN: POSITIVE_LABEL} for r in positive_reviews]
        + [{config.REVIEW_COLUMN: r, config.LABEL_COLUMN: NEGATIVE_LABEL} for r in negative_reviews]
    )

    result = pd.DataFrame(records).dropna(subset=[config.REVIEW_COLUMN]).reset_index(drop=True)
    return result


def split_and_save(
    df: pd.DataFrame,
    output_dir: str = config.DATA_DIR,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
    train_file: str | None = None,
    test_file: str | None = None,
) -> tuple[str, str]:
    """
    Stratified split of *df* into train and test sets and save them as CSV.

    Args:
        df:          DataFrame with ``review`` and ``label`` columns.
        output_dir:  Directory where the CSV files are written.
        test_split:  Fraction of data to hold out for testing.
        seed:        Random seed for reproducibility.
        train_file:  Override path for the training CSV.
        test_file:   Override path for the test CSV.

    Returns:
        (train_path, test_path) — absolute paths of the saved CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = train_file or os.path.join(output_dir, "train.csv")
    test_path = test_file or os.path.join(output_dir, "test.csv")

    train_df, test_df = train_test_split(
        df,
        test_size=test_split,
        random_state=seed,
        stratify=df[config.LABEL_COLUMN],
    )

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    return train_path, test_path


def prepare_data(
    xlsx_file: str = DEFAULT_XLSX_FILE,
    output_dir: str = config.DATA_DIR,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[str, str]:
    """
    End-to-end convenience function: load Excel → split → save CSV files.

    Args:
        xlsx_file:  Path to the raw Excel dataset.
        output_dir: Directory to write ``train.csv`` and ``test.csv``.
        test_split: Fraction of data reserved for the test set.
        seed:       Random seed.

    Returns:
        (train_path, test_path) — paths of the saved CSV files.
    """
    print(f"[INFO] Loading Excel data from: {xlsx_file}")
    df = load_xlsx(xlsx_file)
    print(f"[INFO] Total reviews loaded: {len(df)}")

    label_counts = df[config.LABEL_COLUMN].value_counts().sort_index()
    for lbl, cnt in label_counts.items():
        print(f"  Label {lbl} ({config.LABEL_MAP.get(lbl, lbl)}): {cnt}")

    print(f"[INFO] Splitting data — test fraction: {test_split:.0%}, seed: {seed}")
    train_df, test_df = train_test_split(
        df,
        test_size=test_split,
        random_state=seed,
        stratify=df[config.LABEL_COLUMN],
    )

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Train set: {len(train_df)} samples → {train_path}")
    print(f"[INFO] Test  set: {len(test_df)} samples → {test_path}")
    return train_path, test_path
