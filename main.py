"""
main.py — Entry point for the CAR-Bert sentiment analysis system.

Commands:
    train    — Fine-tune BERT on labeled car exterior reviews
    predict  — Run inference on new reviews (requires a trained model)

Examples:
    # Fine-tune on sample data
    python main.py train

    # Fine-tune on your own labeled data
    python main.py train --data /path/to/your/reviews.csv

    # Predict sentiment for a CSV of reviews
    python main.py predict --data data/sample_reviews.csv --output results.csv

    # Predict a single review from the command line
    python main.py predict --text "车漆质量很好，颜色漂亮，非常满意！"
"""

import argparse
import sys

import config
from src.train import train
from src.predict import predict_from_file, predict_texts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="汽车外饰评价情感极性分类系统 — Car Exterior Review Sentiment Analysis (BERT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
