"""FastAPI Multi-Label GitHub Issue Triage Service.

Exposes REST endpoints for automated triage:
- GET  /health : Health and model status
- POST /triage : Classifies issue title and body into multi-label categories
                 using fine-tuned LoRA adapters and optimal per-label decision thresholds.
"""

import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
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
logger = logging.getLogger("api")

# Global model state
state: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "thresholds": {lbl: 0.5 for lbl in config.TARGET_LABELS},
    "device": "cpu",
    "ready": False,
}


def clean_text_input(title: str, body: str) -> str:
    """Preprocess and format raw issue input into clean composite sequence."""
    title_clean = re.sub(r"<[^>]+>", " ", title or "").strip()
    body_clean = re.sub(r"<[^>]+>", " ", body or "").strip()
    # Normalize markdown links and URLs
    body_clean = re.sub(r"!\[.*?\]\(.*?\)", "", body_clean)
    body_clean = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", body_clean)
    body_clean = re.sub(r"https?://\S+", "[URL]", body_clean)
    body_clean = re.sub(r"[ \t]+", " ", body_clean)

    return f"Title: {title_clean}\n\nBody: {body_clean}".strip()


def load_model_and_thresholds():
    """Load tokenizer, model weights (fused or LoRA), and per-label tuned thresholds."""
    fused_dir = config.MODELS_DIR / "fused-model"
    adapter_dir = config.LORA_ADAPTER_DIR
    thresholds_file = config.THRESHOLDS_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device
    torch.set_num_threads(1)  # Restrict multi-threading memory spikes on 512MB instances

    logger.info(f"Initializing model on device: {device}")

    # 1. Load tuned thresholds if present
    if thresholds_file.exists():
        try:
            with open(thresholds_file, "r", encoding="utf-8") as f:
                th_data = json.load(f)
                state["thresholds"] = th_data.get("optimal_thresholds", state["thresholds"])
                logger.info(f"Loaded tuned per-label thresholds: {state['thresholds']}")
        except Exception as e:
            logger.warning(f"Could not parse thresholds file ({e}), using default 0.5")

    # 2. Load tokenizer and model
    try:
        if fused_dir.exists() and (fused_dir / "model.safetensors").exists():
            logger.info(f"Loading standalone fused model from {fused_dir} (low memory mode)...")
            tokenizer = AutoTokenizer.from_pretrained(str(fused_dir))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(fused_dir),
                num_labels=config.NUM_LABELS,
                problem_type="multi_label_classification",
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID,
                low_cpu_mem_usage=True,
            )
        elif adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter from {adapter_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.BASE_MODEL_NAME,
                num_labels=config.NUM_LABELS,
                problem_type="multi_label_classification",
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        else:
            logger.warning(f"Adapter not found. Loading base model as fallback.")
            tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                config.BASE_MODEL_NAME,
                num_labels=config.NUM_LABELS,
                problem_type="multi_label_classification",
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID,
                low_cpu_mem_usage=True,
            )

        model.eval()
        model.to(device)
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["ready"] = True
        logger.info("GitHub Issue Triage model loaded and ready for inference!")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        state["ready"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle model startup and teardown."""
    logger.info("Starting up GitHub Issue Triage Service...")
    load_model_and_thresholds()
    yield
    logger.info("Shutting down service...")


# Pydantic Schemas
class TriageRequest(BaseModel):
    title: str = Field(..., description="Issue title", min_length=2, example="Fatal crash: NullPointerException in ConcurrentMode")
    body: Optional[str] = Field("", description="Issue body description", example="Renderer throws uncaught error on route switch under Suspense.")


class TriageResponse(BaseModel):
    predicted_labels: List[str] = Field(..., description="List of predicted category labels")
    confidence_scores: Dict[str, float] = Field(..., description="Predicted probabilities per category")
    thresholds_applied: Dict[str, float] = Field(..., description="Threshold applied for each category")
    execution_time_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    target_labels: List[str]
    thresholds: Dict[str, float]


# Initialize FastAPI application
app = FastAPI(
    title="GitHub Issue Triage API",
    description="Multi-label GitHub Issue Classifier fine-tuned with LoRA on DistilBERT using per-label decision thresholds.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root():
    """Serve the interactive web dashboard."""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if state["ready"] else "initializing",
        model_loaded=state["ready"],
        device=state["device"],
        target_labels=config.TARGET_LABELS,
        thresholds=state["thresholds"],
    )


@app.post("/triage", response_model=TriageResponse, tags=["Inference"])
async def triage_issue(request: TriageRequest):
    if not state["ready"] or state["model"] is None:
        # Re-attempt lazy load if model was saved post-startup
        load_model_and_thresholds()
        if not state["ready"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is still initializing or adapter not found.",
            )

    start_time = time.time()
    full_text = clean_text_input(title=request.title, body=request.body)

    try:
        inputs = state["tokenizer"](
            full_text,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        ).to(state["device"])

        with torch.no_grad():
            outputs = state["model"](**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        predicted_labels = []
        confidence_scores = {}

        for idx, label in enumerate(config.TARGET_LABELS):
            score = round(float(probs[idx]), 4)
            th = state["thresholds"].get(label, 0.5)
            confidence_scores[label] = score
            if score >= th:
                predicted_labels.append(label)

        # Fallback to top-scoring label if no label met its individual threshold
        if not predicted_labels:
            best_idx = int(np.argmax(probs))
            predicted_labels.append(config.TARGET_LABELS[best_idx])

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return TriageResponse(
            predicted_labels=predicted_labels,
            confidence_scores=confidence_scores,
            thresholds_applied=state["thresholds"],
            execution_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
