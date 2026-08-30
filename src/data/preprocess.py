"""Data Preprocessing & Multi-Label Stratification Module.

Maps raw issue labels to a standard taxonomy, cleans and normalizes issue text
(HTML/markdown stripping), constructs multi-hot binary vectors, and produces
stratified train/val/test splits.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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
logger = logging.getLogger("preprocess")


def clean_markdown_and_html(text: str) -> Tuple[str, bool]:
    """Clean markdown artifacts, HTML tags, and detect presence of code blocks.

    Returns:
        Tuple[str, bool]: (cleaned_text, has_code_block)
    """
    if not text or not isinstance(text, str):
        return "", False

    has_code = False

    # Detect code blocks
    if re.search(r"```[\s\S]*?```", text) or re.search(r"`[^`]+`", text):
        has_code = True

    # Replace markdown code blocks with placeholder to preserve semantic intent without token bloat
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", r" [CODE_BLOCK] \1 [/CODE_BLOCK] ", text)

    # Remove HTML tags (e.g. <img ...>, <div>, <details>, <summary>)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Replace markdown image links ![alt](url) and links [text](url) -> text
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned)

    # Normalize URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", "[URL]", cleaned)

    # Normalize excessive whitespace, tabs, and newlines
    cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    return cleaned.strip(), has_code


def map_raw_labels(raw_labels: List[str]) -> Set[str]:
    """Map a list of raw GitHub labels to the unified target label set."""
    mapped = set()
    for lbl in raw_labels:
        lbl_clean = lbl.strip().lower()

        # Direct lookup in mapping
        if lbl_clean in config.RAW_LABEL_MAPPING:
            mapped.add(config.RAW_LABEL_MAPPING[lbl_clean])
            continue

        # Substring / pattern fallback rules
        if any(w in lbl_clean for w in ["bug", "defect", "fault", "error", "fail", "crash"]):
            mapped.add("bug")
        if any(w in lbl_clean for w in ["feature", "enhancement", "proposal", "rfc"]):
            mapped.add("feature_request")
        if any(w in lbl_clean for w in ["doc", "docs", "documentation", "guide"]):
            mapped.add("documentation")
        if any(w in lbl_clean for w in ["question", "help", "support", "discussion", "usage"]):
            mapped.add("question")
        if "duplicate" in lbl_clean:
            mapped.add("duplicate")
        if any(w in lbl_clean for w in ["need", "info", "repro", "clarification", "waiting"]):
            mapped.add("needs_more_info")
        if any(w in lbl_clean for w in ["critical", "p0", "p1", "urgent", "blocker", "high priority", "severity: high"]):
            mapped.add("critical")

    return {l for l in mapped if l in config.TARGET_LABELS}


def iterative_multilabel_split(
    df: pd.DataFrame,
    label_cols: List[str],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Multi-label stratified split using iterative stratification to balance label distributions."""
    np.random.seed(random_state)
    n_samples = len(df)

    desired_ratios = np.array([train_size, val_size, test_size])
    desired_ratios = desired_ratios / desired_ratios.sum()
    target_counts = (desired_ratios * n_samples).astype(int)

    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    split_names = ["train", "val", "test"]

    Y = df[label_cols].values
    unassigned = set(range(n_samples))

    # Iterative stratification per label starting with rarest
    for l_idx in np.argsort(Y.sum(axis=0)):
        pos_indices = [i for i in unassigned if Y[i, l_idx] == 1]
        np.random.shuffle(pos_indices)

        for idx in pos_indices:
            if idx not in unassigned:
                continue

            # Assign to split with smallest current proportion relative to target
            proportions_current = [
                len(split_indices[s]) / max(target_counts[s_i], 1)
                for s_i, s in enumerate(split_names)
            ]
            chosen_split = split_names[np.argmin(proportions_current)]
            split_indices[chosen_split].append(idx)
            unassigned.remove(idx)

    # Distribute remaining samples
    unassigned_list = list(unassigned)
    np.random.shuffle(unassigned_list)
    for idx in unassigned_list:
        proportions_current = [
            len(split_indices[s]) / max(target_counts[s_i], 1)
            for s_i, s in enumerate(split_names)
        ]
        chosen_split = split_names[np.argmin(proportions_current)]
        split_indices[chosen_split].append(idx)

    train_df = df.iloc[split_indices["train"]].copy().reset_index(drop=True)
    val_df = df.iloc[split_indices["val"]].copy().reset_index(drop=True)
    test_df = df.iloc[split_indices["test"]].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def print_dataset_statistics(df: pd.DataFrame, split_name: str = "Dataset") -> None:
    """Print multi-label distribution and cardinality."""
    total = len(df)
    logger.info(f"=== {split_name} Statistics (Total samples: {total}) ===")

    label_counts = {lbl: int(df[lbl].sum()) for lbl in config.TARGET_LABELS}
    for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total > 0 else 0
        logger.info(f"  - {lbl:20s}: {count:5d} ({pct:5.1f}%)")

    labels_per_issue = df[config.TARGET_LABELS].sum(axis=1)
    logger.info(f"  * Average labels per issue (Cardinality): {labels_per_issue.mean():.2f}")
    logger.info(f"  * Issues with >1 label (Multi-label): {(labels_per_issue > 1).sum()} ({(labels_per_issue > 1).mean() * 100:.1f}%)")
    logger.info(f"  * Issues with exactly 1 label: {(labels_per_issue == 1).sum()} ({(labels_per_issue == 1).mean() * 100:.1f}%)")


def preprocess_issues(
    raw_path: Path = config.RAW_DATA_DIR / "github_issues_raw.json",
    train_output: Path = config.TRAIN_CSV_PATH,
    val_output: Path = config.VAL_CSV_PATH,
    test_output: Path = config.TEST_CSV_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end preprocessing pipeline for raw GitHub issues."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run fetch_issues.py first.")

    logger.info(f"Loading raw issues from {raw_path}...")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_issues = json.load(f)

    logger.info(f"Loaded {len(raw_issues)} raw issue records.")

    processed_records = []
    dropped_no_label = 0

    for issue in raw_issues:
        raw_labels = issue.get("labels", [])
        mapped_labels = map_raw_labels(raw_labels)

        if not mapped_labels:
            dropped_no_label += 1
            continue

        title = issue.get("title", "") or ""
        body = issue.get("body", "") or ""

        clean_title, _ = clean_markdown_and_html(title)
        clean_body, has_code = clean_markdown_and_html(body)

        full_text = f"Title: {clean_title}\n\nBody: {clean_body}".strip()

        if len(full_text) < 5:
            continue

        record = {
            "issue_id": issue.get("id"),
            "repo": issue.get("repo"),
            "text": full_text,
            "has_code": int(has_code),
        }

        for lbl in config.TARGET_LABELS:
            record[lbl] = 1 if lbl in mapped_labels else 0

        processed_records.append(record)

    logger.info(f"Processed {len(processed_records)} valid labeled issues (Dropped {dropped_no_label} unmapped issues).")

    df = pd.DataFrame(processed_records)

    train_df, val_df, test_df = iterative_multilabel_split(
        df=df,
        label_cols=config.TARGET_LABELS,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)

    export_cols = ["text"] + config.TARGET_LABELS
    train_df[export_cols].to_csv(train_output, index=False, encoding="utf-8")
    val_df[export_cols].to_csv(val_output, index=False, encoding="utf-8")
    test_df[export_cols].to_csv(test_output, index=False, encoding="utf-8")

    logger.info(f"Saved train split ({len(train_df)} rows) -> {train_output}")
    logger.info(f"Saved val split   ({len(val_df)} rows) -> {val_output}")
    logger.info(f"Saved test split  ({len(test_df)} rows) -> {test_output}")

    print_dataset_statistics(df, split_name="Entire Dataset")
    print_dataset_statistics(train_df, split_name="Train Split")
    print_dataset_statistics(val_df, split_name="Validation Split")
    print_dataset_statistics(test_df, split_name="Test Split")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw GitHub issues into multi-label dataset.")
    parser.add_argument("--raw-file", type=str, default=str(config.RAW_DATA_DIR / "github_issues_raw.json"))
    parser.add_argument("--train-out", type=str, default=str(config.TRAIN_CSV_PATH))
    parser.add_argument("--val-out", type=str, default=str(config.VAL_CSV_PATH))
    parser.add_argument("--test-out", type=str, default=str(config.TEST_CSV_PATH))

    args = parser.parse_args()

    preprocess_issues(
        raw_path=Path(args.raw_file),
        train_output=Path(args.train_out),
        val_output=Path(args.val_out),
        test_output=Path(args.test_out),
    )


if __name__ == "__main__":
    main()
