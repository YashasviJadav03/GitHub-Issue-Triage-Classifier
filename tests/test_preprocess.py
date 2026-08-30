"""Unit tests for dataset preprocessing, text cleaning, and label mapping."""

import numpy as np
import pandas as pd
import pytest

import config
from src.data.preprocess import clean_markdown_and_html, map_raw_labels, iterative_multilabel_split


def test_clean_markdown_and_html_basic():
    text = "<h1>Fatal crash</h1><p>Check this out <a href='https://example.com'>link</a></p>"
    cleaned, has_code = clean_markdown_and_html(text)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "Fatal crash" in cleaned
    assert "link" in cleaned
    assert has_code is False


def test_clean_markdown_code_detection():
    text = "Running this code crashes:\n```python\nimport torch\ntorch.cuda.init()\n```\nDetails here."
    cleaned, has_code = clean_markdown_and_html(text)
    assert has_code is True
    assert "[CODE_BLOCK]" in cleaned
    assert "import torch" in cleaned


def test_map_raw_labels_exact_and_heuristic():
    # Exact mappings
    assert "bug" in map_raw_labels(["kind/bug", "p0"])
    assert "feature_request" in map_raw_labels(["type: enhancement"])
    assert "documentation" in map_raw_labels(["area: docs"])
    assert "question" in map_raw_labels(["help wanted"])
    assert "duplicate" in map_raw_labels(["status: duplicate"])
    assert "needs_more_info" in map_raw_labels(["waiting-for-user-response"])
    assert "critical" in map_raw_labels(["priority: critical"])

    # Multi-label mapping
    combo = map_raw_labels(["type: bug", "severity: high", "waiting-for-user-response"])
    assert "bug" in combo
    assert "critical" in combo
    assert "needs_more_info" in combo


def test_iterative_multilabel_split():
    # Create synthetic test dataframe
    data = {
        "text": [f"Issue number {i}" for i in range(100)],
        "bug": [1 if i % 2 == 0 else 0 for i in range(100)],
        "feature_request": [1 if i % 3 == 0 else 0 for i in range(100)],
        "documentation": [1 if i % 5 == 0 else 0 for i in range(100)],
        "question": [1 if i % 7 == 0 else 0 for i in range(100)],
        "duplicate": [1 if i % 11 == 0 else 0 for i in range(100)],
        "needs_more_info": [1 if i % 4 == 0 else 0 for i in range(100)],
        "critical": [1 if i % 9 == 0 else 0 for i in range(100)],
    }
    df = pd.DataFrame(data)

    train_df, val_df, test_df = iterative_multilabel_split(
        df=df,
        label_cols=config.TARGET_LABELS,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    assert len(train_df) + len(val_df) + len(test_df) == 100
    assert len(train_df) == 70
    assert len(val_df) >= 14
    assert len(test_df) >= 14

    # Every label should have at least 1 positive sample in train
    for lbl in config.TARGET_LABELS:
        assert train_df[lbl].sum() > 0
