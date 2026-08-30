import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LORA_ADAPTER_DIR = MODELS_DIR / "lora-adapter"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Target multi-label classes
# Multi-label taxonomy designed for practical GitHub issue triage
TARGET_LABELS = [
    "bug",
    "feature_request",
    "documentation",
    "question",
    "duplicate",
    "needs_more_info",
    "critical",
]

NUM_LABELS = len(TARGET_LABELS)
LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(TARGET_LABELS)}

# Repository raw label mapping dictionary
# Maps repository-specific label variations into our unified 7-label schema
RAW_LABEL_MAPPING = {
    # Bug variants
    "bug": "bug",
    "type: bug": "bug",
    "kind/bug": "bug",
    "defect": "bug",
    "issue": "bug",
    "type: defect": "bug",
    "status: confirmed-bug": "bug",
    "type-bug": "bug",
    "p: bug": "bug",
    "bug: confirmed": "bug",
    "component: bug": "bug",

    # Feature request / enhancement variants
    "enhancement": "feature_request",
    "feature": "feature_request",
    "type: feature": "feature_request",
    "type: enhancement": "feature_request",
    "kind/feature": "feature_request",
    "kind/enhancement": "feature_request",
    "proposal": "feature_request",
    "feature-request": "feature_request",
    "new feature": "feature_request",

    # Documentation variants
    "documentation": "documentation",
    "docs": "documentation",
    "type: docs": "documentation",
    "type: documentation": "documentation",
    "kind/documentation": "documentation",
    "area: documentation": "documentation",
    "area: docs": "documentation",
    "component: docs": "documentation",

    # Question / help / support variants
    "question": "question",
    "type: question": "question",
    "kind/question": "question",
    "help wanted": "question",
    "support": "question",
    "usage": "question",
    "need help": "question",
    "discussion": "question",

    # Duplicate variants
    "duplicate": "duplicate",
    "status: duplicate": "duplicate",
    "type: duplicate": "duplicate",
    "resolution: duplicate": "duplicate",
    "closed: duplicate": "duplicate",

    # Needs more info variants
    "needs-more-info": "needs_more_info",
    "needs more info": "needs_more_info",
    "status: needs-info": "needs_more_info",
    "status: more-info-needed": "needs_more_info",
    "info-needed": "needs_more_info",
    "waiting-for-user-response": "needs_more_info",
    "needs repro": "needs_more_info",
    "status: needs-reproduce": "needs_more_info",
    "clarification needed": "needs_more_info",

    # Critical / high priority variants
    "critical": "critical",
    "priority: critical": "critical",
    "priority: high": "critical",
    "priority: p0": "critical",
    "p0": "critical",
    "p1": "critical",
    "severity: critical": "critical",
    "severity: high": "critical",
    "urgent": "critical",
    "high priority": "critical",
    "blocker": "critical",
    "crash": "critical",
}

# Default Repositories for dataset acquisition
DEFAULT_REPOS = [
    "facebook/react",
    "microsoft/vscode",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "golang/go",
]

# Model & Training Hyperparameters
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DEFAULT_THRESHOLD = 0.5

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_lin", "k_lin", "v_lin", "out_lin"]

# Artifact Paths
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV_PATH = PROCESSED_DATA_DIR / "test.csv"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
THRESHOLDS_PATH = RESULTS_DIR / "label_thresholds.json"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.csv"
MISCLASSIFIED_PATH = RESULTS_DIR / "misclassified_examples.csv"
COMPARISON_TABLE_PATH = RESULTS_DIR / "comparison_table.md"
