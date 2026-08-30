"""Multi-Label LoRA Fine-Tuning Module.

Fine-tunes a pretrained transformer (distilbert-base-uncased) with LoRA parameter-efficient
adapters for multi-label GitHub issue classification using BCEWithLogitsLoss and Sigmoid activation.
Tracks Micro-F1 and Macro-F1 per epoch, persists adapters, and logs experiment runs to results/experiment_log.csv.
"""

import argparse
from datetime import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

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
logger = logging.getLogger("train")


class MultiLabelDataset:
    """Helper to convert CSV pandas DataFrames into HuggingFace tokenized Datasets."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_dataset(self, csv_path: Path) -> Dataset:
        df = pd.read_csv(csv_path)
        labels = df[config.TARGET_LABELS].values.astype(np.float32)
        texts = df["text"].tolist()

        raw_dict = {"text": texts, "labels": labels.tolist()}
        hf_dataset = Dataset.from_dict(raw_dict)

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
            )
            tokenized["labels"] = examples["labels"]
            return tokenized

        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {csv_path.name}",
        )
        return tokenized_dataset


def compute_multilabel_metrics(eval_pred, threshold: float = 0.5) -> Dict[str, float]:
    """Compute Micro-F1, Macro-F1, Precision, and Recall for multi-label predictions."""
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    micro_prec = precision_score(labels, preds, average="micro", zero_division=0)
    macro_prec = precision_score(labels, preds, average="macro", zero_division=0)
    micro_rec = recall_score(labels, preds, average="micro", zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro", zero_division=0)

    metrics = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_precision": float(micro_prec),
        "macro_precision": float(macro_prec),
        "micro_recall": float(micro_rec),
        "macro_recall": float(macro_rec),
    }

    for idx, label_name in enumerate(config.TARGET_LABELS):
        f1 = f1_score(labels[:, idx], preds[:, idx], zero_division=0)
        metrics[f"f1_{label_name}"] = float(f1)

    return metrics


def log_experiment_to_csv(entry: Dict[str, Any], log_path: Path = config.EXPERIMENT_LOG_PATH) -> None:
    """Append experiment hyperparameter and performance metrics to CSV."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([entry])
    if log_path.exists():
        df_existing = pd.read_csv(log_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(log_path, index=False)
    logger.info(f"Appended run result to experiment log -> {log_path}")


def train(
    run_id: Optional[str] = None,
    model_name: str = config.BASE_MODEL_NAME,
    train_csv: Path = config.TRAIN_CSV_PATH,
    val_csv: Path = config.VAL_CSV_PATH,
    output_dir: Path = config.LORA_ADAPTER_DIR,
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    lora_r: int = config.LORA_R,
    lora_alpha: int = config.LORA_ALPHA,
    lora_dropout: float = config.LORA_DROPOUT,
    max_length: int = config.MAX_LENGTH,
    weight_decay: float = config.WEIGHT_DECAY,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train DistilBERT with LoRA adapters for multi-label classification."""
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_id:
        run_id = f"lora_r{lora_r}_lr{lr}_ep{epochs}_{datetime.now().strftime('%H%M%S')}"

    logger.info(f"Loading tokenizer for base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Tokenizing train and validation splits...")
    data_prep = MultiLabelDataset(tokenizer=tokenizer, max_length=max_length)
    train_dataset = data_prep.prepare_dataset(train_csv)
    val_dataset = data_prep.prepare_dataset(val_csv)

    logger.info(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation examples.")

    logger.info(f"Initializing base model '{model_name}' for multi-label classification...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        modules_to_save=["classifier", "pre_classifier"],
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        save_total_limit=2,
        seed=seed,
        dataloader_drop_last=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_multilabel_metrics,
    )

    logger.info(
        f"Starting training run '{run_id}': epochs={epochs}, lr={lr}, batch_size={batch_size}, "
        f"LoRA(r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})"
    )

    train_result = trainer.train()

    logger.info("Evaluating fine-tuned model on validation set...")
    eval_metrics = trainer.evaluate()
    logger.info(
        f"Validation Results: Micro-F1: {eval_metrics.get('eval_micro_f1', 0):.4f} | "
        f"Macro-F1: {eval_metrics.get('eval_macro_f1', 0):.4f} | "
        f"Loss: {eval_metrics.get('eval_loss', 0):.4f}"
    )

    logger.info(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_length": max_length,
        "eval_micro_f1": round(float(eval_metrics.get("eval_micro_f1", 0)), 4),
        "eval_macro_f1": round(float(eval_metrics.get("eval_macro_f1", 0)), 4),
        "eval_loss": round(float(eval_metrics.get("eval_loss", 0)), 4),
        "train_runtime_sec": round(float(train_result.metrics.get("train_runtime", 0)), 2),
    }

    summary_file = output_dir / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    log_experiment_to_csv(run_summary)
    logger.info(f"Training complete! Summary saved to {summary_file}")
    return run_summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT with LoRA for multi-label issue triage.")
    parser.add_argument("--run-id", type=str, default=None, help="Identifier for experiment run")
    parser.add_argument("--model-name", type=str, default=config.BASE_MODEL_NAME, help="Pretrained model name")
    parser.add_argument("--train-csv", type=str, default=str(config.TRAIN_CSV_PATH), help="Train CSV path")
    parser.add_argument("--val-csv", type=str, default=str(config.VAL_CSV_PATH), help="Validation CSV path")
    parser.add_argument("--output-dir", type=str, default=str(config.LORA_ADAPTER_DIR), help="Output adapter dir")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=config.LORA_R, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=config.LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=config.LORA_DROPOUT, help="LoRA dropout")
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train(
        run_id=args.run_id,
        model_name=args.model_name,
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
