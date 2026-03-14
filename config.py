"""
Configuration for the Car Exterior Review Sentiment Analysis System.
汽车外饰评价情感分析配置文件
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# Default dataset file
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "sample_reviews.csv")

# ─── Model ────────────────────────────────────────────────────────────────────

# Pretrained BERT model for Chinese text
PRETRAINED_MODEL_NAME = "bert-base-chinese"

# Number of sentiment classes: 0=Negative, 1=Positive, 2=Neutral
NUM_LABELS = 3

LABEL_MAP = {
    0: "负面 (Negative)",
    1: "正面 (Positive)",
    2: "中性 (Neutral)",
}

LABEL_NAMES = ["负面", "正面", "中性"]

# ─── Training ─────────────────────────────────────────────────────────────────

MAX_SEQ_LENGTH = 128       # Maximum token sequence length
TRAIN_BATCH_SIZE = 16      # Training batch size
EVAL_BATCH_SIZE = 32       # Evaluation batch size
NUM_EPOCHS = 3             # Number of training epochs
LEARNING_RATE = 2e-5       # Learning rate for AdamW
WEIGHT_DECAY = 0.01        # Weight decay
WARMUP_RATIO = 0.1         # Fraction of total steps used for warmup
TEST_SPLIT = 0.2           # Fraction of data used for validation
RANDOM_SEED = 42           # Random seed for reproducibility

# ─── Columns ──────────────────────────────────────────────────────────────────

REVIEW_COLUMN = "review"   # Name of the text column in the CSV
LABEL_COLUMN = "label"     # Name of the label column in the CSV
