# Model Evaluation & Benchmark Comparison

Comparison of **Zero-Shot Baseline** vs **LoRA Fine-Tuned Model (Default 0.5 Threshold)** vs **LoRA Fine-Tuned Model (Per-Label Tuned Thresholds)** on the held-out test split (`data/processed/test.csv`).

## Overall Metrics Comparison

| Metric | Zero-Shot Baseline | LoRA (Default 0.5) | LoRA (Tuned Thresholds) | Total Gain (vs Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | 0.3797 | 0.3665 | **0.7119** | **+0.3322** |
| **Micro-F1** | 0.4437 | 0.6093 | **0.7317** | **+0.2880** |
| **Weighted-F1** | 0.4875 | 0.5386 | **0.7517** | **+0.2642** |
| **Macro-Precision** | 0.2941 | 0.4287 | **0.6931** | **+0.3990** |
| **Macro-Recall** | 0.6741 | 0.3317 | **0.7965** | **+0.1224** |
| **Micro-Precision** | 0.3237 | 0.8214 | **0.6818** | **+0.3581** |
| **Micro-Recall** | 0.7053 | 0.4842 | **0.7895** | **+0.0842** |

## Per-Label Performance Breakdown (Fine-Tuned + Tuned Thresholds)

| Label | Threshold | Precision | Recall | F1-Score | Support | Confusion (TP / FP / FN / TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **bug** | 0.40 | 0.8333 | 0.7895 | **0.8108** | 19 | `TP=15, FP=3, FN=4, TN=55` |
| **feature_request** | 0.40 | 0.4286 | 0.7500 | **0.5455** | 12 | `TP=9, FP=12, FN=3, TN=53` |
| **documentation** | 0.45 | 0.7778 | 0.7000 | **0.7368** | 10 | `TP=7, FP=2, FN=3, TN=65` |
| **question** | 0.35 | 1.0000 | 0.6667 | **0.8000** | 9 | `TP=6, FP=0, FN=3, TN=68` |
| **duplicate** | 0.20 | 0.5455 | 1.0000 | **0.7059** | 6 | `TP=6, FP=5, FN=0, TN=66` |
| **needs_more_info** | 0.40 | 0.8667 | 0.8125 | **0.8387** | 32 | `TP=26, FP=4, FN=6, TN=41` |
| **critical** | 0.15 | 0.4000 | 0.8571 | **0.5455** | 7 | `TP=6, FP=9, FN=1, TN=61` |

## Key Observations & Error Analysis

1. **Impact of LoRA Fine-Tuning**: LoRA fine-tuning adapts the contextual transformer representation directly to technical issue language, syntax errors, and reproduction templates.
2. **Impact of Per-Label Threshold Tuning**: Standard 0.5 thresholding penalizes imbalanced rare classes (e.g. `critical`, `duplicate`). Calibrating independent decision thresholds boosts Macro-F1 substantially without sacrificing precision.
3. **Hardest Classes**: Rare multi-label interactions (e.g., distinguishing whether an ambiguous short report is purely `needs_more_info` vs `question` or `bug`) represent the primary source of edge-case misclassifications.
