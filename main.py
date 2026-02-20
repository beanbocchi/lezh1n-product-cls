from contextlib import asynccontextmanager
from typing import List, Optional
import logging
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from categories import CATEGORIES, get_category_id, get_category_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "Lezh1n/xlm-roberta-ecommerce-classifier"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release resources on shutdown."""
    logger.info("=" * 70)
    logger.info("Loading model: %s on %s", MODEL_NAME, DEVICE)
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    app.state.classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
    )
    app.state.model = model
    app.state.tokenizer = tokenizer

    logger.info("✓ Model loaded — %d categories", len(CATEGORIES))
    logger.info("=" * 70)

    yield

    del app.state.classifier, app.state.model, app.state.tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    logger.info("Model unloaded.")


app = FastAPI(
    title="E-Commerce Product Classifier API",
    description="Classify product text into one of 32 e-commerce categories.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ClassificationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "Sony WH-1000XM5 wireless headphones", "top_k": 5}
        }
    )

    text: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(5, ge=1, le=32)


class BatchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "iPhone 15 Pro Max 256GB",
                    "Nike running shoes size 10",
                    "Samsung 4K Smart TV",
                ],
                "top_k": 5,
            }
        }
    )

    texts: List[str] = Field(..., min_length=1, max_length=100)
    top_k: Optional[int] = Field(5, ge=1, le=32)


class Prediction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category_id": 0,
                "category_name": "electronics",
                "display_name": "Electronics",
                "score": 0.9534,
            }
        }
    )

    category_id: int = Field(..., description="Category ID (0-31)")
    category_name: str = Field(..., description="Category internal name")
    display_name: str = Field(..., description="Category display name")
    score: float = Field(..., description="Confidence score (0-1)")


class ClassificationResponse(BaseModel):
    text: str
    predictions: List[Prediction]
    inference_time: float


class BatchResponse(BaseModel):
    results: List[ClassificationResponse]
    total_time: float
    count: int


class CategoryInfo(BaseModel):
    category_id: int
    name: str
    display_name: str
    description: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_prediction(label: str, score: float) -> Prediction:
    """Map a raw pipeline label + score to a Prediction object."""
    category_id = get_category_id(label)
    if category_id is None:
        logger.warning("Unknown category label from model: %s", label)
        return Prediction(
            category_id=-1,
            category_name=label,
            display_name="Unknown",
            score=score,
        )
    info = get_category_info(category_id)
    return Prediction(
        category_id=category_id,
        category_name=label,
        display_name=info["display_name"],
        score=score,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "E-Commerce Product Classification API",
        "model": MODEL_NAME,
        "version": "2.0.0",
        "categories": len(CATEGORIES),
        "docs": "/docs",
    }


@app.get("/health", tags=["General"])
async def health(req: Request):
    if not hasattr(req.app.state, "classifier"):
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "categories": len(CATEGORIES),
    }


@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify(body: ClassificationRequest, req: Request):
    """Classify a single product description. Returns top-K category predictions."""
    classifier = req.app.state.classifier
    start = time.perf_counter()
    try:
        raw = classifier(body.text, top_k=body.top_k)
        predictions = [_to_prediction(p["label"], p["score"]) for p in raw[: body.top_k]]
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(500, f"Classification failed: {exc}") from exc
    return ClassificationResponse(
        text=body.text,
        predictions=predictions,
        inference_time=time.perf_counter() - start,
    )


@app.post("/classify/batch", response_model=BatchResponse, tags=["Classification"])
async def classify_batch(body: BatchRequest, req: Request):
    """Classify multiple product descriptions in a single batched inference call."""
    classifier = req.app.state.classifier
    start = time.perf_counter()
    try:
        # True batch inference — all texts processed in one forward pass
        all_raw = classifier(body.texts, top_k=body.top_k, batch_size=32)
        results = [
            ClassificationResponse(
                text=text,
                predictions=[
                    _to_prediction(p["label"], p["score"])
                    for p in raw[: body.top_k]
                ],
                inference_time=0.0,
            )
            for text, raw in zip(body.texts, all_raw)
        ]
    except Exception as exc:
        logger.exception("Batch classification failed")
        raise HTTPException(500, f"Batch classification failed: {exc}") from exc
    return BatchResponse(
        results=results,
        total_time=time.perf_counter() - start,
        count=len(body.texts),
    )


# NOTE: specific routes must be registered BEFORE parameterised ones to avoid
# FastAPI matching "/categories/name/electronics" as category_id="name".
@app.get("/categories/name/{category_name}", response_model=CategoryInfo, tags=["Categories"])
async def get_category_by_name(category_name: str):
    """Look up a category by its internal name, e.g. `electronics`."""
    category_id = get_category_id(category_name)
    if category_id is None:
        raise HTTPException(404, f"Category '{category_name}' not found")
    info = get_category_info(category_id)
    return CategoryInfo(
        category_id=category_id,
        name=info["name"],
        display_name=info["display_name"],
        description=info["description"],
    )


@app.get("/categories/{category_id}", response_model=CategoryInfo, tags=["Categories"])
async def get_category(category_id: int):
    """Look up a category by its numeric ID (0–31)."""
    if category_id < 0 or category_id >= len(CATEGORIES):
        raise HTTPException(404, f"Category ID {category_id} not found. Valid range: 0-31")
    info = get_category_info(category_id)
    return CategoryInfo(
        category_id=category_id,
        name=info["name"],
        display_name=info["display_name"],
        description=info["description"],
    )


@app.get("/categories", response_model=List[CategoryInfo], tags=["Categories"])
async def list_categories():
    """List all 32 categories."""
    return [
        CategoryInfo(
            category_id=cat_id,
            name=info["name"],
            display_name=info["display_name"],
            description=info["description"],
        )
        for cat_id, info in CATEGORIES.items()
    ]


@app.get("/model-info", tags=["Information"])
async def model_info(req: Request):
    """Return model metadata and performance metrics."""
    if not hasattr(req.app.state, "model"):
        raise HTTPException(503, "Model not loaded")
    m = req.app.state.model
    t = req.app.state.tokenizer
    return {
        "model_name": MODEL_NAME,
        "base_model": "FacebookAI/xlm-roberta-base",
        "device": DEVICE,
        "num_parameters": m.num_parameters(),
        "num_categories": len(CATEGORIES),
        "category_id_range": "0-31",
        "max_sequence_length": t.model_max_length,
        "accuracy": "90.14%",
        "f1_score": "90.00%",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9999, log_level="info")
