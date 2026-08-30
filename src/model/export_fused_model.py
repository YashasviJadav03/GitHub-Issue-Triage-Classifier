"""Model Fusion & Memory Optimization Utility.

Merges the trained LoRA adapter weights directly into the base DistilBERT model
and saves a consolidated, standalone model to models/fused-model/.
This eliminates PEFT runtime memory overhead and reduces RAM footprint from ~600MB to <200MB,
allowing seamless deployment on 512MB RAM free-tier instances (Render, HuggingFace Spaces).
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
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
logger = logging.getLogger("export_fused_model")


def export_fused_model(
    adapter_dir: Path = config.LORA_ADAPTER_DIR,
    output_dir: Path = config.MODELS_DIR / "fused-model",
    base_model_name: str = config.BASE_MODEL_NAME,
) -> Path:
    """Merge LoRA adapter into base model and save consolidated checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model '{base_model_name}' and LoRA adapter from {adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=config.NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        low_cpu_mem_usage=True,
    )

    lora_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    logger.info("Merging LoRA adapter weights into base model...")
    fused_model = lora_model.merge_and_unload()
    fused_model.eval()

    logger.info(f"Saving standalone fused model to {output_dir}...")
    fused_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"Fused model saved successfully! Total standalone size: {sum(f.stat().st_size for f in output_dir.glob('*')) / (1024 * 1024):.1f} MB")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Export standalone fused model for low-memory cloud deployment.")
    parser.add_argument("--adapter-dir", type=str, default=str(config.LORA_ADAPTER_DIR))
    parser.add_argument("--output-dir", type=str, default=str(config.MODELS_DIR / "fused-model"))
    args = parser.parse_args()

    export_fused_model(
        adapter_dir=Path(args.adapter_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
