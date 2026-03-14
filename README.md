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
├── 满意以及不满意评论全部.xlsx  # 原始爬取数据
├── data/
│   ├── sample_reviews.csv  # 示例标注数据集 (50 条汽车外饰评价)
│   ├── train.csv           # 训练集 (由 prepare 命令生成)
│   ├── test.csv            # 测试集 (由 prepare 命令生成)
│   └── README.md           # 数据格式说明
├── src/
│   ├── data_prep.py        # Excel 数据加载与训练/测试集分割
│   ├── dataset.py          # 数据集加载与 Tokenization
│   ├── model.py            # BERT 情感分类模型
│   ├── train.py            # 模型训练脚本
│   ├── predict.py          # 推理/预测脚本
│   └── utils.py            # 工具函数
└── tests/                  # 单元测试
```

## 完整工作流程 Full Workflow

### 第一步 Step 1：安装依赖 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 第二步 Step 2：准备数据 Prepare Data

将原始 Excel 文件（`满意以及不满意评论全部.xlsx`）自动转换为训练集与测试集 CSV：

```bash
python main.py prepare
```

此命令会：
1. 读取 Excel 文件中的满意（正面）与不满意（负面）评价
2. 按 **80% 训练 / 20% 测试** 的比例进行分层随机划分
3. 将结果保存至 `data/train.csv` 和 `data/test.csv`

**可选参数：**

```bash
python main.py prepare \
    --xlsx 满意以及不满意评论全部.xlsx \
    --output-dir data/ \
    --test-split 0.2 \
    --seed 42
```

**输出示例：**

```
[INFO] Loading Excel data from: 满意以及不满意评论全部.xlsx
[INFO] Total reviews loaded: 18074
  Label 0 (负面 (Negative)): 9037
  Label 1 (正面 (Positive)): 9037
[INFO] Splitting data — test fraction: 20%, seed: 42
[INFO] Train set: 14459 samples → data/train.csv
[INFO] Test  set: 3615 samples → data/test.csv
```

---

### 第三步 Step 3：自动训练模型 Train the Model

使用上一步生成的训练集自动微调 BERT 模型：

```bash
python main.py train --data data/train.csv
```

训练完成后，最优模型会自动保存到 `saved_model/` 目录。

**可选参数：**

```bash
python main.py train \
    --data data/train.csv \    # 训练数据路径
    --save-dir saved_model/   # 模型保存目录
    --epochs 3                # 训练轮数（默认 3）
    --batch-size 16           # 批次大小（默认 16）
    --lr 2e-5                 # 学习率（默认 2e-5）
```

---

### 第四步 Step 4：使用测试集生成分析结果 Evaluate on Test Set

训练完成后，使用测试集评估模型效果并生成分析报告：

```bash
python main.py evaluate --data data/test.csv --output results.csv
```

此命令会：
1. 对测试集中的每条评价运行情感预测
2. 输出情感分布统计
3. 输出精确率、召回率、F1 等评估指标（如测试集包含真实标签）
4. 将结果保存至指定 CSV 文件（可选）

**输出示例：**

```
────────────────────────────────────────────────────────────────────────────────
No.   Sentiment            Review
────────────────────────────────────────────────────────────────────────────────
1     正面 (Positive)      外观非常漂亮，颜色鲜艳，车身线条流畅，很有质感，...
2     负面 (Negative)      车漆质量很差，买了没多久就出现了细小划痕，不耐用...
────────────────────────────────────────────────────────────────────────────────

[情感分布 / Sentiment Distribution]
  负面 (Negative)          1808 (50.0%)
  正面 (Positive)          1807 (50.0%)

[评估报告 / Evaluation Report]
              precision    recall  f1-score   support
          负面       0.92      0.91      0.91      1808
          正面       0.91      0.92      0.91      1807
```

---

## 快速开始 Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（分割训练集和测试集）
python main.py prepare

# 3. 自动训练模型
python main.py train --data data/train.csv

# 4. 使用测试集生成分析结果
python main.py evaluate --data data/test.csv --output results.csv
```

---

## 其他用法 Other Usage

### 对自定义 CSV 文件进行批量预测

```bash
python main.py predict --data data/sample_reviews.csv --output results.csv
```

### 直接预测单条评价文本

```bash
python main.py predict --text "车漆质量很好，颜色漂亮，非常满意！"
```

### 使用示例数据训练

```bash
python main.py train --data data/sample_reviews.csv
```

---

## 数据格式 Data Format

原始 Excel 文件（`满意以及不满意评论全部.xlsx`）格式：

| 列 A | 列 B（正面评价） | 列 C | 列 D（负面评价） |
|------|----------------|------|----------------|
| 满意 | 正面评价文本…   | 不满意 | 负面评价文本… |

生成的 CSV 格式（`train.csv` / `test.csv`）：

| 列名     | 类型   | 说明                                      |
|---------|--------|------------------------------------------|
| `review` | string | 评价文本                                  |
| `label`  | int    | 情感标签：`0`=负面, `1`=正面, `2`=中性  |

---

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
| `TEST_SPLIT`         | `0.2`               | 测试集比例            |

---

## 命令行参数 CLI Arguments

### `prepare`

```
python main.py prepare [--xlsx XLSX] [--output-dir DIR]
                       [--test-split FLOAT] [--seed INT]
```

### `train`

```
python main.py train [--data DATA] [--save-dir DIR] [--epochs N]
                     [--batch-size N] [--lr LR]
```

### `evaluate`

```
python main.py evaluate [--data DATA] [--model DIR]
                        [--output OUTPUT] [--batch-size N]
```

### `predict`

```
python main.py predict (--data DATA | --text TEXT)
                       [--model DIR] [--output OUTPUT] [--batch-size N]
```

---

## 测试 Tests

```bash
python -m pytest tests/ -v
```

> 部分测试需要网络连接以下载 BERT 模型权重，在离线环境中会自动跳过。
