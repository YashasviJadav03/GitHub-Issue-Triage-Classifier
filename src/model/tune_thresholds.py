"""Per-Label Multi-Label Threshold Optimization Module.

In multi-label classification with severe class imbalance (e.g., high-frequency 'bug'
vs rare 'duplicate' or 'critical' labels), a static global threshold of 0.5 is suboptimal:
1. Rare classes often produce lower predicted probabilities because negative examples
   dominate the BCE loss gradient. Setting threshold=0.5 causes high false-negative rates
   (near-zero recall) on rare classes.
2. High-prevalence classes may produce overconfident probabilities, causing false positives.
3. Tuning per-label thresholds on validation probabilities decouples each class's
   precision-recall tradeoff, allowing rare classes to use lower cutoffs (e.g. 0.25 - 0.35)
   and frequent classes to use higher cutoffs (e.g. 0.55 - 0.65), significantly boosting Macro-F1.

This script runs inference on data/processed/val.csv, evaluates thresholds from 0.10 to 0.90
(step 0.05) independently for each label, and saves the optimal thresholds to results/label_thresholds.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
logger = logging.getLogger("tune_thresholds")


def predict_probabilities(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Run batched forward pass and apply Sigmoid to obtain multi-label probabilities."""
    model.eval()
    model.to(device)
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_range: Tuple[float, float, float] = (0.10, 0.90, 0.05),
) -> Dict[str, Any]:
    """Search for the F1-maximizing threshold per label independently."""
    min_th, max_th, step = threshold_range
    candidate_thresholds = np.arange(min_th, max_th + 1e-5, step)

    optimal_thresholds = {}
    tuning_details = {}

    default_preds = (y_probs >= 0.5).astype(int)
    tuned_preds = np.zeros_like(y_true, dtype=int)

    for idx, label in enumerate(config.TARGET_LABELS):
        y_true_col = y_true[:, idx]
        probs_col = y_probs[:, idx]

        best_th = 0.50
        best_f1 = -1.0
        best_prec = 0.0
        best_rec = 0.0

        th_history = []

        for th in candidate_thresholds:
            th = round(float(th), 2)
            preds_col = (probs_col >= th).astype(int)
            f1 = f1_score(y_true_col, preds_col, zero_division=0)
            prec = precision_score(y_true_col, preds_col, zero_division=0)
            rec = recall_score(y_true_col, preds_col, zero_division=0)

            th_history.append({"threshold": th, "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)})

            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_prec = prec
                best_rec = rec

        optimal_thresholds[label] = float(best_th)
        tuned_preds[:, idx] = (probs_col >= best_th).astype(int)

        # Baseline metrics at 0.5
        base_f1 = f1_score(y_true_col, default_preds[:, idx], zero_division=0)
        base_prec = precision_score(y_true_col, default_preds[:, idx], zero_division=0)
        base_rec = recall_score(y_true_col, default_preds[:, idx], zero_division=0)

        tuning_details[label] = {
            "optimal_threshold": float(best_th),
            "tuned_f1": round(float(best_f1), 4),
            "tuned_precision": round(float(best_prec), 4),
            "tuned_recall": round(float(best_rec), 4),
            "baseline_0.5_f1": round(float(base_f1), 4),
            "baseline_0.5_precision": round(float(base_prec), 4),
            "baseline_0.5_recall": round(float(base_rec), 4),
            "f1_improvement": round(float(best_f1 - base_f1), 4),
        }

    # Summary macro and micro F1
    base_macro_f1 = f1_score(y_true, default_preds, average="macro", zero_division=0)
    tuned_macro_f1 = f1_score(y_true, tuned_preds, average="macro", zero_division=0)
    base_micro_f1 = f1_score(y_true, default_preds, average="micro", zero_division=0)
    tuned_micro_f1 = f1_score(y_true, tuned_preds, average="micro", zero_division=0)

    return {
        "optimal_thresholds": optimal_thresholds,
        "summary": {
            "baseline_0.5_macro_f1": round(float(base_macro_f1), 4),
            "tuned_macro_f1": round(float(tuned_macro_f1), 4),
            "macro_f1_gain": round(float(tuned_macro_f1 - base_macro_f1), 4),
            "baseline_0.5_micro_f1": round(float(base_micro_f1), 4),
            "tuned_micro_f1": round(float(tuned_micro_f1), 4),
            "micro_f1_gain": round(float(tuned_micro_f1 - base_micro_f1), 4),
        },
        "per_label_details": tuning_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune per-label classification thresholds on validation set.")
    parser.add_argument("--adapter-dir", type=str, default=str(config.LORA_ADAPTER_DIR), help="Trained LoRA adapter path")
    parser.add_argument("--val-csv", type=str, default=str(config.VAL_CSV_PATH), help="Validation CSV path")
    parser.add_argument("--output", type=str, default=str(config.THRESHOLDS_PATH), help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")

    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir)
    val_path = Path(args.val_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        raise FileNotFoundError(f"Validation file {val_path} not found.")

    logger.info(f"Loading validation data from {val_path}...")
    df = pd.read_csv(val_path)
    texts = df["text"].tolist()
    y_true = df[config.TARGET_LABELS].values

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading fine-tuned model and adapter from {adapter_path} (device={device})...")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Computing validation probabilities via Sigmoid activation...")
    y_probs = predict_probabilities(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=args.batch_size,
        device=device,
    )

    logger.info("Optimizing per-label thresholds over range [0.10, 0.90]...")
    results = find_optimal_thresholds(y_true=y_true, y_probs=y_probs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("===========================================================")
    logger.info("            PER-LABEL THRESHOLD TUNING SUMMARY             ")
    logger.info("===========================================================")
    logger.info(
        f"Macro-F1: {results['summary']['baseline_0.5_macro_f1']:.4f} -> {results['summary']['tuned_macro_f1']:.4f} "
        f"(+{results['summary']['macro_f1_gain']:.4f})"
    )
    logger.info(
        f"Micro-F1: {results['summary']['baseline_0.5_micro_f1']:.4f} -> {results['summary']['tuned_micro_f1']:.4f} "
        f"(+{results['summary']['micro_f1_gain']:.4f})"
    )
    logger.info("-----------------------------------------------------------")
    logger.info(f"{'Label':<18} | {'Opt Thresh':<10} | {'Base F1':<10} | {'Tuned F1':<10} | {'Gain':<8}")
    logger.info("-----------------------------------------------------------")
    for lbl, det in results["per_label_details"].items():
        logger.info(
            f"{lbl:<18} | {det['optimal_threshold']:<10.2f} | {det['baseline_0.5_f1']:<10.4f} | "
            f"{det['tuned_f1']:<10.4f} | +{det['f1_improvement']:<8.4f}"
        )
    logger.info("===========================================================")
    logger.info(f"Saved optimal per-label thresholds to {output_path}")


if __name__ == "__main__":
    main()
