# GitHub Issue Triage Classifier

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_App-6366f1.svg?logo=render&logoColor=white)](https://github-issue-triage-api.onrender.com/)
[![CI Pipeline](https://github.com/YashasviJadav03/GitHub-Issue-Triage-Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/YashasviJadav03/GitHub-Issue-Triage-Classifier/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace PEFT](https://img.shields.io/badge/PEFT-LoRA-FFD21E.svg)](https://huggingface.co/docs/peft)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end multi-label machine learning system for automated GitHub issue classification and priority triage. The architecture fine-tunes a transformer backbone (`distilbert-base-uncased`) using **Parameter-Efficient Fine-Tuning (PEFT / LoRA)** paired with **independent per-label decision threshold calibration** to simultaneously classify multi-category issue types and predict priority tiers.

> **Try it live:** [https://github-issue-triage-api.onrender.com](https://github-issue-triage-api.onrender.com/) | **API Docs:** [/docs](https://github-issue-triage-api.onrender.com/docs)

---

## 1. Problem Formulation: Multi-Label vs Multi-Class

Real-world GitHub issues do not conform to mutually exclusive categories. A single issue frequently exhibits multiple valid attributes simultaneously:
* `bug` + `critical` + `needs_more_info`
* `feature_request` + `documentation`

### Limitations of Standard Multi-Class Formulations
* **Softmax Output Constraint**: Single-label multi-class architectures enforce $\sum_i P(y_i) = 1$, artificially suppressing concurrent tags.
* **Class Imbalance Sensitivity**: High-frequency labels (`bug`, `needs_more_info`) dominate cross-entropy loss gradients, yielding near-zero recall on rare yet critical labels (`critical`, `duplicate`).

### Multi-Label Architectural Design
* **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) applied independently across all $K$ output heads.
* **Activation**: Sigmoid function $\sigma(z_k) = \frac{1}{1 + e^{-z_k}}$ generating unconstrained per-class posterior probabilities $\hat{y}_k \in [0, 1]$.
* **Per-Label Threshold Calibration**: In place of a static $0.5$ cutoff, each class decision threshold $t_k \in [0.10, 0.90]$ is independently optimized on validation data to maximize its respective $F_1$ score.

---

## 2. Key Features and Technical Highlights

### Parameter-Efficient LoRA Adaptation
* Wraps the `distilbert-base-uncased` attention projection layers (`q_lin`, `k_lin`, `v_lin`, `out_lin`) with low-rank decomposition matrices ($r=16$, $\alpha=32$).
* Trains under 2% of total model parameters while matching full fine-tuning representation capacity and preventing catastrophic forgetting.

### Independent Per-Label Decision Calibration
* Solves multi-label class imbalance without artificial oversampling by grid-searching validation probability space ($0.10 \le t \le 0.90$, step $0.05$).
* Decouples the precision-recall trade-off per class, allowing low-prevalence classes (`critical`, `duplicate`) to trigger at lower probability boundaries without inflating false positives for high-prevalence classes.

### Iterative Multi-Label Stratification
* Implements multi-label greedy stratification across train (70%), validation (15%), and test (15%) partitions to preserve identical label co-occurrence distributions and cardinality across splits.

### Automated Noise Cleaning and Markdown Normalization
* Cleans raw HTML tags, formats markdown image/hyperlink artifacts, flags code snippets with structured tokens (`[CODE_BLOCK]`), and normalizes issue titles and bodies into unified composite input sequences.

### Systematic Error Analysis and Uncertainty Ranking
* Automatically calculates per-sample Hamming distance and cross-entropy divergence on the test split to surface and export the top misclassified edge cases for qualitative inspection.

### Enterprise-Grade Serving and CI/CD
* Fully containerized asynchronous REST API (FastAPI + Uvicorn) with sub-25ms inference latency, Pydantic validation, structured health monitoring, automated Pytest suite, and GitHub Actions CI.

---

## 3. Taxonomy and Label Schema

| Category | Description | Sample Keywords / Raw Tags |
| :--- | :--- | :--- |
| `bug` | Software defects, runtime crashes, and unexpected behaviors | `type: bug`, `defect`, `kind/bug`, `segfault` |
| `feature_request` | New capabilities, enhancements, and architectural proposals | `enhancement`, `proposal`, `RFC`, `type: feature` |
| `documentation` | Documentation updates, missing guides, and tutorial corrections | `area: docs`, `documentation`, `tutorial` |
| `question` | Usage inquiries, troubleshooting questions, and discussions | `help wanted`, `question`, `usage`, `discussion` |
| `duplicate` | Duplicate issues already tracked in existing reports | `duplicate`, `status: duplicate`, `closed: duplicate` |
| `needs_more_info` | Issues missing reproduction steps, system logs, or environment details | `needs-repro`, `waiting-for-user-response`, `info-needed` |
| `critical` | P0/P1 blockers, data loss risks, and security vulnerabilities | `severity: critical`, `blocker`, `p0`, `urgent` |

---

## 4. Project Structure

```text
github-issue-triage/
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI pipeline (Pytest + Docker build validation)
├── data/
│   ├── raw/                       # Raw issue records fetched from GitHub API
│   └── processed/                 # Multi-label stratified splits (train.csv, val.csv, test.csv)
├── models/
│   └── lora-adapter/              # Trained PEFT LoRA adapter weights, config, and tokenizer
├── results/
│   ├── baseline_metrics.json      # Zero-shot baseline evaluation metrics
│   ├── label_thresholds.json      # Calibrated per-label decision thresholds
│   ├── experiment_log.csv         # Hyperparameter ablation run tracking
│   ├── misclassified_examples.csv # Error analysis dataset for edge cases
│   └── comparison_table.md        # Benchmark comparison report
├── src/
│   ├── api/
│   │   └── main.py                # FastAPI microservice (POST /triage, GET /health)
│   ├── data/
│   │   ├── fetch_issues.py        # GitHub REST API data acquisition engine
│   │   └── preprocess.py          # Markdown/HTML cleaning & multi-label stratification
│   └── model/
│       ├── baseline_eval.py       # Zero-shot NLI baseline evaluator
│       ├── train.py               # Multi-label LoRA fine-tuning with BCEWithLogitsLoss
│       ├── tune_thresholds.py     # Independent per-label F1 threshold optimizer
│       ├── compare_runs.py        # Hyperparameter run comparator ranked by Macro-F1
│       └── evaluate.py            # Final test evaluation & error analysis
├── tests/
│   ├── test_api.py                # FastAPI endpoint integration tests
│   └── test_preprocess.py         # Text preprocessing & stratification unit tests
├── Dockerfile                     # Production containerization specification
├── config.py                      # Centralized paths, hyperparameters & taxonomy
└── requirements.txt               # Pinned project dependencies
```

---

## 5. Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Model & Fine-Tuning** | PyTorch, HuggingFace Transformers, PEFT (LoRA), Accelerate |
| **Data Engineering** | Scikit-learn, Pandas, NumPy, Regex, Python Requests |
| **API & Serving** | FastAPI, Uvicorn, Pydantic v2 |
| **Testing & CI/CD** | Docker, GitHub Actions, Pytest, HTTPX |