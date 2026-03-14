"""
tests/test_model.py — Unit tests for the model module.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import SentimentClassifier, get_tokenizer
from tests.conftest import requires_network


@requires_network
class TestGetTokenizer:
    def test_returns_tokenizer(self):
        tokenizer = get_tokenizer()
        assert tokenizer is not None

    def test_encodes_chinese_text(self):
        tokenizer = get_tokenizer()
        encoding = tokenizer("外观非常漂亮", return_tensors="pt")
        assert "input_ids" in encoding
        assert encoding["input_ids"].shape[1] > 0


@requires_network
class TestSentimentClassifier:
    @pytest.fixture(scope="class")
    def model(self):
        return SentimentClassifier()

    def test_instantiates(self, model):
        assert model is not None

    def test_output_logits_shape(self, model, bert_tokenizer):
        encoding = bert_tokenizer(
            "车漆很好看",
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding.get("token_type_ids"),
            )
        assert output.logits.shape == (1, config.NUM_LABELS)

    def test_forward_with_labels_returns_loss(self, model, bert_tokenizer):
        encoding = bert_tokenizer(
            "外观设计一般",
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor([1], dtype=torch.long)
        with torch.no_grad():
            output = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding.get("token_type_ids"),
                labels=labels,
            )
        assert output.loss is not None
        assert output.loss.item() > 0

    def test_save_and_load(self, model, tmp_path, bert_tokenizer):
        save_dir = str(tmp_path / "model")
        model.save(save_dir)

        loaded = SentimentClassifier.load(save_dir)
        assert loaded is not None

        encoding = bert_tokenizer(
            "车漆颜色很漂亮",
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            original_logits = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            ).logits
            loaded_logits = loaded(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            ).logits
        assert torch.allclose(original_logits, loaded_logits, atol=1e-5)


# ── Tests that do NOT require network ────────────────────────────────────────

class TestSentimentClassifierOffline:
    def test_load_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            SentimentClassifier.load("/nonexistent/model/dir")
