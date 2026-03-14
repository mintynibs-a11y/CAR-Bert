# CAR-Bert — 汽车外饰用户评价情感极性分类系统

A BERT-based sentiment polarity classification system for Chinese car exterior user reviews.

## 项目简介 Overview

本项目通过 BERT 架构对已爬取的汽车外饰相关用户评价进行情感极性分类，将每条评价自动分为：

| Label | 情感 | Sentiment |
|-------|------|-----------|
| `1`   | 正面 | Positive  |
| `0`   | 负面 | Negative  |
| `2`   | 中性 | Neutral   |

## 项目结构 Project Structure

```
CAR-Bert/
├── config.py               # 全局配置 (模型路径、超参数等)
├── main.py                 # 命令行入口
├── requirements.txt        # Python 依赖
├── data/
│   ├── sample_reviews.csv  # 示例标注数据集 (50 条汽车外饰评价)
│   └── README.md           # 数据格式说明
├── src/
│   ├── dataset.py          # 数据集加载与 Tokenization
│   ├── model.py            # BERT 情感分类模型
│   ├── train.py            # 模型训练脚本
│   ├── predict.py          # 推理/预测脚本
│   └── utils.py            # 工具函数
└── tests/                  # 单元测试
```

## 快速开始 Quick Start

### 1. 安装依赖 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 训练模型 Train the Model

使用内置的示例数据集训练：

```bash
python main.py train
```

使用自己的标注数据训练（CSV 文件需包含 `review` 和 `label` 列）：

```bash
python main.py train --data /path/to/your/labeled_reviews.csv
```

训练完成后，模型会保存到 `saved_model/` 目录。

### 3. 情感预测 Predict Sentiment

#### 对 CSV 文件中的评价批量预测

```bash
python main.py predict --data data/sample_reviews.csv --output results.csv
```

#### 直接预测单条评价文本

```bash
python main.py predict --text "车漆质量很好，颜色漂亮，非常满意！"
```

#### 输出示例

```
────────────────────────────────────────────────────────────────────────────────
No.   Sentiment            Review
────────────────────────────────────────────────────────────────────────────────
1     正面 (Positive)      外观非常漂亮，颜色鲜艳，车身线条流畅，很有质感，...
2     负面 (Negative)      车漆质量很差，买了没多久就出现了细小划痕，不耐用...
3     中性 (Neutral)       外观设计中规中矩，没有特别出彩的地方，也没有特别...
────────────────────────────────────────────────────────────────────────────────

[情感分布 / Sentiment Distribution]
  中性 (Neutral)            5 (10.0%)
  正面 (Positive)          30 (60.0%)
  负面 (Negative)          15 (30.0%)
```

## 数据格式 Data Format

训练数据为 CSV 格式，需包含以下列：

| 列名     | 类型   | 说明                                      |
|---------|--------|------------------------------------------|
| `review` | string | 评价文本                                  |
| `label`  | int    | 情感标签：`0`=负面, `1`=正面, `2`=中性  |

推理数据只需 `review` 列。

## 配置 Configuration

在 `config.py` 中调整参数：

| 参数                  | 默认值              | 说明                 |
|----------------------|---------------------|---------------------|
| `PRETRAINED_MODEL_NAME` | `bert-base-chinese` | 预训练 BERT 模型     |
| `NUM_LABELS`         | `3`                 | 分类数量 (正/负/中性) |
| `MAX_SEQ_LENGTH`     | `128`               | 最大序列长度          |
| `NUM_EPOCHS`         | `3`                 | 训练轮数              |
| `LEARNING_RATE`      | `2e-5`              | 学习率               |
| `TRAIN_BATCH_SIZE`   | `16`                | 训练批次大小          |

## 命令行参数 CLI Arguments

### `train`

```
python main.py train [--data DATA] [--save-dir DIR] [--epochs N]
                     [--batch-size N] [--lr LR]
```

### `predict`

```
python main.py predict (--data DATA | --text TEXT)
                       [--model DIR] [--output OUTPUT] [--batch-size N]
```

## 测试 Tests

```bash
python -m pytest tests/ -v
```

> 部分测试需要网络连接以下载 BERT 模型权重，在离线环境中会自动跳过。
