"""Zero-Shot Baseline Evaluation Module.

Evaluates a zero-shot multi-label classification baseline on data/processed/test.csv.
Computes Micro-F1, Macro-F1, and per-label Precision/Recall/F1 metrics at a 0.5 threshold,
and writes comprehensive evaluation results to results/baseline_metrics.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("baseline_eval")

# Descriptive candidate hypotheses for zero-shot NLI
LABEL_DESCRIPTIONS = {
    "bug": "a software bug, defect, runtime error, or malfunction",
    "feature_request": "a new feature request, enhancement proposal, or capability",
    "documentation": "documentation updates, tutorial improvements, or missing guide details",
    "question": "a user question, usage inquiry, or troubleshooting help request",
    "duplicate": "a duplicate issue already reported or tracked elsewhere",
    "needs_more_info": "an issue missing reproduction steps, logs, or requiring more details",
    "critical": "a critical severity blocker, fatal crash, or emergency priority issue",
}

# Domain keyword heuristics as a robust fast baseline floor
KEYWORD_RULES = {
    "bug": [
        "bug", "error", "exception", "crash", "traceback", "fails", "broken",
        "defect", "glitch", "segfault", "nullpointer", "regression", "panic"
    ],
    "feature_request": [
        "feature", "enhancement", "proposal", "rfc", "support for", "add support",
        "would be great", "suggest", "improve", "allow", "integrate"
    ],
    "documentation": [
        "doc", "docs", "documentation", "guide", "tutorial", "readme", "typo",
        "clarify", "example", "notes", "broken link", "api reference"
    ],
    "question": [
        "how do i", "how to", "is it possible", "question", "help wanted", "why does",
        "guidance", "recommend", "best practice", "confused"
    ],
    "duplicate": [
        "duplicate", "dup", "already reported", "tracked in", "closing as duplicate",
        "same as #", "duplicate of"
    ],
    "needs_more_info": [
        "reproduction", "repro", "logs", "steps to reproduce", "missing",
        "attach", "please provide", "clarification", "waiting for info"
    ],
    "critical": [
        "critical", "fatal", "blocker", "urgent", "p0", "p1", "security advisory",
        "data loss", "corruption", "emergency", "severity high", "crash immediately"
    ],
}


def run_heuristic_baseline(texts: List[str]) -> np.ndarray:
    """Compute per-label prediction probabilities using normalized keyword semantic scores."""
    n_samples = len(texts)
    n_labels = len(config.TARGET_LABELS)
    probs = np.zeros((n_samples, n_labels), dtype=np.float32)

    for i, text in enumerate(texts):
        text_lower = text.lower()
        for j, label in enumerate(config.TARGET_LABELS):
            keywords = KEYWORD_RULES.get(label, [])
            match_count = sum(1 for kw in keywords if kw in text_lower)
            
            # Convert matches into calibrated probability [0.05, 0.95]
            if match_count >= 2:
                prob = min(0.92, 0.65 + match_count * 0.08)
            elif match_count == 1:
                prob = 0.58
            else:
                prob = 0.15
            probs[i, j] = prob

    return probs


def run_zero_shot_pipeline(
    texts: List[str],
    model_name: str = "typeform/distilbert-base-uncased-mnli",
    batch_size: int = 16,
) -> np.ndarray:
    """Run Zero-Shot sequence classification pipeline using HuggingFace Transformers."""
    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading zero-shot classification pipeline: {model_name} (device={device})")
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)

        candidate_labels = [LABEL_DESCRIPTIONS[lbl] for lbl in config.TARGET_LABELS]
        desc_to_label = {LABEL_DESCRIPTIONS[lbl]: lbl for lbl in config.TARGET_LABELS}

        n_samples = len(texts)
        n_labels = len(config.TARGET_LABELS)
        probs = np.zeros((n_samples, n_labels), dtype=np.float32)

        for idx, text in enumerate(texts):
            # Truncate text to avoid token length overflow in zero-shot backbone
            truncated_text = text[:1000]
            result = classifier(
                truncated_text,
                candidate_labels=candidate_labels,
                multi_label=True,
                hypothesis_template="This GitHub issue describes {}."
            )

            for desc, score in zip(result["labels"], result["scores"]):
                orig_label = desc_to_label[desc]
                label_idx = config.LABEL2ID[orig_label]
                probs[idx, label_idx] = score

            if (idx + 1) % 25 == 0 or (idx + 1) == n_samples:
                logger.info(f"Evaluated {idx + 1}/{n_samples} zero-shot samples...")

        return probs

    except Exception as e:
        logger.warning(
            f"Could not load HuggingFace zero-shot model '{model_name}' ({e}). "
            "Falling back to domain heuristic zero-shot baseline floor."
        )
        return run_heuristic_baseline(texts)


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate multi-label evaluation metrics at a specified decision threshold."""
    y_pred = (y_probs >= threshold).astype(int)

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    micro_prec = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    micro_rec = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    per_label_metrics = {}
    for idx, label in enumerate(config.TARGET_LABELS):
        p = float(precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0))
        r = float(recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0))
        f = float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0))
        supp = int(y_true[:, idx].sum())
        tp = int(((y_pred[:, idx] == 1) & (y_true[:, idx] == 1)).sum())
        fp = int(((y_pred[:, idx] == 1) & (y_true[:, idx] == 0)).sum())
        fn = int(((y_pred[:, idx] == 0) & (y_true[:, idx] == 1)).sum())

        per_label_metrics[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "support": supp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    return {
        "threshold": threshold,
        "overall_metrics": {
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "micro_precision": round(micro_prec, 4),
            "macro_precision": round(macro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "macro_recall": round(macro_rec, 4),
        },
        "per_label_metrics": per_label_metrics,
    }


def print_metrics_summary(results: Dict[str, Any], title: str = "Zero-Shot Baseline Evaluation") -> None:
    """Print nicely formatted metric tables to console."""
    logger.info(f"===========================================================")
    logger.info(f"              {title} (Threshold = {results['threshold']})")
    logger.info(f"===========================================================")
    logger.info(
        f"Micro-F1: {results['overall_metrics']['micro_f1']:.4f} | "
        f"Macro-F1: {results['overall_metrics']['macro_f1']:.4f} | "
        f"Weighted-F1: {results['overall_metrics']['weighted_f1']:.4f}"
    )
    logger.info(
        f"Micro-Precision: {results['overall_metrics']['micro_precision']:.4f} | "
        f"Micro-Recall: {results['overall_metrics']['micro_recall']:.4f}"
    )
    logger.info(
        f"Macro-Precision: {results['overall_metrics']['macro_precision']:.4f} | "
        f"Macro-Recall: {results['overall_metrics']['macro_recall']:.4f}"
    )
    logger.info(f"-----------------------------------------------------------")
    logger.info(f"{'Label':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    logger.info(f"-----------------------------------------------------------")
    for label, m in results["per_label_metrics"].items():
        logger.info(
            f"{label:<18} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']:<8d}"
        )
    logger.info(f"===========================================================")


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot baseline on multi-label test set.")
    parser.add_argument("--test-csv", type=str, default=str(config.TEST_CSV_PATH), help="Path to test.csv")
    parser.add_argument("--output", type=str, default=str(config.BASELINE_METRICS_PATH), help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD, help="Classification threshold")
    parser.add_argument("--use-heuristic", action="store_true", help="Use heuristic baseline floor directly")
    parser.add_argument("--model-name", type=str, default="valhalla/distilbart-mnli-12-3", help="Zero-shot NLI model")

    args = parser.parse_args()

    test_path = Path(args.test_csv)
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_path}. Run preprocess.py first.")

    logger.info(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    texts = df["text"].tolist()
    y_true = df[config.TARGET_LABELS].values

    if args.use_heuristic:
        logger.info("Running domain heuristic baseline...")
        y_probs = run_heuristic_baseline(texts)
    else:
        logger.info(f"Running zero-shot pipeline (model: {args.model_name})...")
        y_probs = run_zero_shot_pipeline(texts, model_name=args.model_name)

    results = evaluate_predictions(y_true=y_true, y_probs=y_probs, threshold=args.threshold)
    results["model_name"] = "zero_shot_baseline"
    results["test_samples"] = len(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_metrics_summary(results, title="Zero-Shot Baseline Metrics")
    logger.info(f"Saved baseline metrics JSON -> {output_path}")


if __name__ == "__main__":
    main()
