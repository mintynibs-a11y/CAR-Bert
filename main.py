"""
main.py — Entry point for the CAR-Bert sentiment analysis system.

Commands:
    prepare  — Convert raw Excel data to train/test CSV files
    train    — Fine-tune BERT on labeled car exterior reviews
    predict  — Run inference on new reviews (requires a trained model)
    evaluate — Run prediction on the held-out test set and show metrics

Examples:
    # Step 1: Prepare data from the uploaded Excel file
    python main.py prepare

    # Step 1 (custom paths):
    python main.py prepare --xlsx 满意以及不满意评论全部.xlsx --output-dir data/

    # Step 2: Train using the prepared training set
    python main.py train --data data/train.csv

    # Step 3: Evaluate on the held-out test set
    python main.py evaluate --data data/test.csv --output results.csv

    # Fine-tune on your own labeled CSV (review + label columns)
    python main.py train --data /path/to/your/reviews.csv

    # Predict sentiment for a CSV of reviews
    python main.py predict --data data/sample_reviews.csv --output results.csv

    # Predict a single review from the command line
    python main.py predict --text "车漆质量很好，颜色漂亮，非常满意！"
"""

import argparse
import sys

import config
from src.data_prep import DEFAULT_XLSX_FILE, prepare_data
from src.train import train
from src.predict import predict_from_file, predict_texts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="汽车外饰评价情感极性分类系统 — Car Exterior Review Sentiment Analysis (BERT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── prepare ────────────────────────────────────────────────────────────────
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Convert raw Excel review data into train/test CSV files",
    )
    prepare_parser.add_argument(
        "--xlsx",
        default=DEFAULT_XLSX_FILE,
        help=f"Path to the raw Excel file (default: {DEFAULT_XLSX_FILE})",
    )
    prepare_parser.add_argument(
        "--output-dir",
        default=config.DATA_DIR,
        help=f"Directory to write train.csv and test.csv (default: {config.DATA_DIR})",
    )
    prepare_parser.add_argument(
        "--test-split",
        type=float,
        default=config.TEST_SPLIT,
        help=f"Fraction of data to reserve as test set (default: {config.TEST_SPLIT})",
    )
    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed for reproducible splitting (default: {config.RANDOM_SEED})",
    )

    # ── train ──────────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Fine-tune BERT on labeled reviews")
    train_parser.add_argument(
        "--data",
        default=config.DEFAULT_DATA_FILE,
        help=f"Path to labeled CSV (default: {config.DEFAULT_DATA_FILE})",
    )
    train_parser.add_argument(
        "--save-dir",
        default=config.MODEL_DIR,
        help=f"Directory to save fine-tuned model (default: {config.MODEL_DIR})",
    )
    train_parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    train_parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")

    # ── predict ────────────────────────────────────────────────────────────────
    predict_parser = subparsers.add_parser("predict", help="Run sentiment inference on reviews")
    group = predict_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", help="Path to CSV file with a 'review' column")
    group.add_argument("--text", help="A single review string to classify")
    predict_parser.add_argument(
        "--model",
        default=config.MODEL_DIR,
        help=f"Directory of fine-tuned BERT model (default: {config.MODEL_DIR})",
    )
    predict_parser.add_argument("--output", default=None, help="Path to save results CSV (optional)")
    predict_parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)

    # ── evaluate ───────────────────────────────────────────────────────────────
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run prediction on the test set and display evaluation metrics",
    )
    evaluate_parser.add_argument(
        "--data",
        default=config.DEFAULT_TEST_FILE,
        help=f"Path to test CSV with 'review' and 'label' columns (default: {config.DEFAULT_TEST_FILE})",
    )
    evaluate_parser.add_argument(
        "--model",
        default=config.MODEL_DIR,
        help=f"Directory of fine-tuned BERT model (default: {config.MODEL_DIR})",
    )
    evaluate_parser.add_argument("--output", default=None, help="Path to save results CSV (optional)")
    evaluate_parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        prepare_data(
            xlsx_file=args.xlsx,
            output_dir=args.output_dir,
            test_split=args.test_split,
            seed=args.seed,
        )

    elif args.command == "train":
        train(
            data_file=args.data,
            save_dir=args.save_dir,
            num_epochs=args.epochs,
            train_batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    elif args.command == "predict":
        if args.text:
            predict_texts([args.text], model_dir=args.model)
        else:
            predict_from_file(
                data_file=args.data,
                model_dir=args.model,
                output_file=args.output,
                batch_size=args.batch_size,
            )

    elif args.command == "evaluate":
        predict_from_file(
            data_file=args.data,
            model_dir=args.model,
            output_file=args.output,
            batch_size=args.batch_size,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
