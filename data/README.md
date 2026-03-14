# 汽车外饰用户评价情感极性分类系统
# Car Exterior Review Sentiment Polarity Classification System

This directory contains sample data and instructions for the dataset format.

## Dataset Format

The training data should be a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `review` | string | The user review text (汽车外饰评价文本) |
| `label` | int | Sentiment label: `0`=Negative(负面), `1`=Positive(正面), `2`=Neutral(中性) |

## Files

- `sample_reviews.csv` — Sample labeled car exterior reviews for training/evaluation

## Sentiment Labels

| Label | Chinese | Description |
|-------|---------|-------------|
| 0 | 负面 | Negative sentiment |
| 1 | 正面 | Positive sentiment |
| 2 | 中性 | Neutral sentiment |

## Preparing Your Own Data

Place your crawled reviews in a CSV file with at minimum a `review` column.
For training, a `label` column is also required.
For inference (prediction only), only the `review` column is needed.
