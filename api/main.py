"""
AMFTA REST API — Inference Service
=====================================

Exposes the trained global model as a FastAPI endpoint for real-time
IoT intrusion detection inference.

Endpoints:
  POST /predict         — Single-sample prediction
  POST /predict/batch   — Batch prediction (up to 1000 samples)
  GET  /health          — Health check
  GET  /model/info      — Model metadata
  GET  /metrics         — Prometheus-format metrics counter

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker-compose up amfta-api
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from amfta.models.local_mlp import LocalMLP

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("amfta.api")

app = FastAPI(
    title="AMFTA Intrusion Detection API",
    description=(
        "Byzantine-Resilient Federated Learning for Smart City IoT Networks. "
        "Submit network flow feature vectors; receive attack probability scores."
    ),
    version="1.0.0",
    contact={"name": "AMFTA Research Team"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe model registry with lazy loading."""

    def __init__(self):
        self._model: Optional[LocalMLP] = None
        self._model_path: Optional[Path] = None
        self._loaded_at: Optional[float] = None
        self._inference_count: int = 0
        self._total_latency: float = 0.0

    def load(self, model_path: Optional[Path] = None) -> None:
        """Load model from checkpoint or initialise a fresh untrained model."""
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location="cpu")
            model = LocalMLP()
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
            self._model_path = model_path
            logger.info("Model loaded from %s", model_path)
        else:
            logger.warning(
                "No checkpoint found at '%s'. Using freshly initialised model. "
                "Predictions will be random until a trained model is loaded.",
                model_path,
            )
            model = LocalMLP()

        model.eval()
        self._model = model
        self._loaded_at = time.time()

    @property
    def model(self) -> LocalMLP:
        if self._model is None:
            self.load()
        return self._model

    def record_inference(self, latency_s: float) -> None:
        self._inference_count += 1
        self._total_latency += latency_s

    @property
    def avg_latency_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return 1000.0 * self._total_latency / self._inference_count

    def get_info(self) -> dict:
        return {
            "model_path": str(self._model_path) if self._model_path else "in-memory",
            "loaded_at": self._loaded_at,
            "inference_count": self._inference_count,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "num_parameters": self.model.num_parameters(),
            "architecture": str(self.model),
        }


registry = ModelRegistry()


@app.on_event("startup")
async def startup_event():
    """Load model at startup."""
    model_path = os.environ.get("MODEL_CHECKPOINT_PATH", "checkpoints/amfta_best.pt")
    registry.load(Path(model_path))
    logger.info("AMFTA API started. Model ready.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-sample prediction request."""
    features: List[float] = Field(
        ...,
        min_items=1,
        max_items=200,
        description="Normalised network flow feature vector (45 features for TON_IoT).",
        example=[0.1] * 45,
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for binary classification.",
    )

    @validator("features")
    def check_feature_range(cls, v):
        if any(not (0.0 <= x <= 1.0) for x in v):
            raise ValueError("All features must be in [0, 1] (MinMax normalised).")
        return v


class PredictResponse(BaseModel):
    attack_probability: float = Field(..., description="P(attack) in [0, 1].")
    is_attack: bool = Field(..., description="True if attack_probability >= threshold.")
    threshold: float
    latency_ms: float
    model_version: str = "1.0.0"


class BatchPredictRequest(BaseModel):
    samples: List[List[float]] = Field(
        ..., min_items=1, max_items=1000,
        description="List of feature vectors.",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_samples: int
    attack_count: int
    normal_count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    inference_count: int
    avg_latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health and readiness check."""
    return HealthResponse(
        status="healthy",
        model_loaded=registry._model is not None,
        inference_count=registry._inference_count,
        avg_latency_ms=registry.avg_latency_ms,
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return loaded model metadata."""
    return registry.get_info()


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """Single-sample intrusion detection prediction.

    Submit a 45-dimensional MinMax-normalised network flow feature vector.
    Receive the attack probability and binary classification result.
    """
    start = time.perf_counter()

    try:
        x = torch.tensor([request.features], dtype=torch.float32)
        with torch.no_grad():
            prob = registry.model(x).item()
    except Exception as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    latency = time.perf_counter() - start
    registry.record_inference(latency)

    return PredictResponse(
        attack_probability=round(prob, 6),
        is_attack=prob >= request.threshold,
        threshold=request.threshold,
        latency_ms=round(latency * 1000, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple network flow samples.

    Processes up to 1000 samples in a single request for throughput-efficient
    inference on IoT gateway devices or monitoring pipelines.
    """
    start = time.perf_counter()

    try:
        X = torch.tensor(request.samples, dtype=torch.float32)
        with torch.no_grad():
            probs = registry.model(X).numpy()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )

    latency = time.perf_counter() - start
    registry.record_inference(latency)

    preds = (probs >= request.threshold).astype(bool)
    predictions = [
        {
            "sample_idx": i,
            "attack_probability": round(float(probs[i]), 6),
            "is_attack": bool(preds[i]),
        }
        for i in range(len(probs))
    ]

    return BatchPredictResponse(
        predictions=predictions,
        total_samples=len(probs),
        attack_count=int(preds.sum()),
        normal_count=int((~preds).sum()),
        latency_ms=round(latency * 1000, 3),
    )


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus-compatible plaintext metrics."""
    info = registry.get_info()
    lines = [
        "# HELP amfta_inference_total Total inference requests",
        "# TYPE amfta_inference_total counter",
        f"amfta_inference_total {info['inference_count']}",
        "# HELP amfta_latency_ms_avg Average inference latency in milliseconds",
        "# TYPE amfta_latency_ms_avg gauge",
        f"amfta_latency_ms_avg {info['avg_latency_ms']}",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))
