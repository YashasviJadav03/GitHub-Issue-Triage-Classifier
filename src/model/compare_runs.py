"""Experiment Run Comparison & Model Selection Module.

Compares fine-tuning ablation runs logged in results/experiment_log.csv,
ranks configurations by Macro-F1 (which penalizes poor performance on rare categories
like 'duplicate' and 'critical'), and flags the optimal hyperparameter combination.
"""

import argparse
import logging
import sys
from pathlib import Path
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
logger = logging.getLogger("compare_runs")


def compare_experiments(log_path: Path = config.EXPERIMENT_LOG_PATH) -> pd.DataFrame:
    """Read experiment log, sort by Macro-F1, and print comparison summary."""
    if not log_path.exists():
        raise FileNotFoundError(f"Experiment log file not found at {log_path}. Run train.py first.")

    df = pd.read_csv(log_path)
    if df.empty:
        logger.warning("Experiment log is empty.")
        return df

    # Sort descending by validation macro-f1, then micro-f1
    df_sorted = df.sort_values(by=["eval_macro_f1", "eval_micro_f1"], ascending=[False, False]).reset_index(drop=True)

    best_run = df_sorted.iloc[0]

    print("\n" + "=" * 90)
    print("                      EXPERIMENT RUN COMPARISON (PEFT / LoRA)                      ")
    print("=" * 90)
    
    # Display table
    display_cols = [
        "run_id", "lora_r", "lora_alpha", "learning_rate", "epochs",
        "eval_macro_f1", "eval_micro_f1", "eval_loss", "train_runtime_sec"
    ]
    avail_cols = [c for c in display_cols if c in df_sorted.columns]
    
    formatted_df = df_sorted[avail_cols].copy()
    print(formatted_df.to_string(index=False))
    print("=" * 90)
    print(f"🏆 BEST CONFIGURATION (by Macro-F1): {best_run['run_id']}")
    print(f"   - LoRA Rank (r):       {best_run.get('lora_r')}")
    print(f"   - LoRA Alpha:          {best_run.get('lora_alpha')}")
    print(f"   - Learning Rate:       {best_run.get('learning_rate')}")
    print(f"   - Epochs:              {best_run.get('epochs')}")
    print(f"   - Val Macro-F1:        {best_run.get('eval_macro_f1'):.4f}")
    print(f"   - Val Micro-F1:        {best_run.get('eval_micro_f1'):.4f}")
    print(f"   - Val BCE Loss:        {best_run.get('eval_loss'):.4f}")
    print("=" * 90)
    print("💡 Note: Macro-F1 is our primary selection metric because it weights all 7 categories")
    print("   equally, preventing frequent labels (e.g. bug) from masking poor recall on rare labels.")
    print("=" * 90 + "\n")

    return df_sorted


def main():
    parser = argparse.ArgumentParser(description="Compare logged experiment runs and identify best LoRA config.")
    parser.add_argument("--log-path", type=str, default=str(config.EXPERIMENT_LOG_PATH), help="Path to experiment_log.csv")
    args = parser.parse_args()

    compare_experiments(log_path=Path(args.log_path))


if __name__ == "__main__":
    main()
