"""Inspection API: POST /inspect (image) -> decision, score, defect type, anomaly map."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .inference import OnnxInspector, overlay, png_base64

MODELS_DIR = Path(os.getenv("DV_MODELS_DIR", "models"))
WEB_DIR = Path(os.getenv("DV_WEB_DIR", Path(__file__).resolve().parents[2] / "web"))
MAX_BYTES = 10 * 1024 * 1024

INSPECTIONS = Counter("dv_inspections_total", "Inspections performed", ["model", "decision"])
LATENCY = Histogram(
    "dv_inference_seconds", "ONNX inference duration", ["model"], buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
)

app = FastAPI(
    title="DefectVision", version="0.1.0", description="Visual inspection of magnetic tiles (PatchCore, U-Net)."
)
_models: dict[str, OnnxInspector] = {}


def models() -> dict[str, OnnxInspector]:
    if not _models:
        for path in sorted(MODELS_DIR.glob("*.onnx")):
            _models[path.stem] = OnnxInspector(path)
    return _models


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": sorted(models())}


@app.get("/models")
def list_models() -> list[dict]:
    return [
        {"name": n, "kind": m.kind, "threshold": m.threshold, "image_size": m.image_size} for n, m in models().items()
    ]


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), model: str = Query("unet"), heatmap: bool = Query(True)) -> dict:
    available = models()
    if model not in available:
        raise HTTPException(404, f"unknown model: {model} (available: {', '.join(sorted(available)) or 'none'})")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image too large (10 MB maximum)")
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(422, "unreadable file: please send a JPEG or PNG image")
    t0 = time.perf_counter()
    result = available[model].inspect(gray)
    LATENCY.labels(model).observe(time.perf_counter() - t0)
    body = result.to_dict()
    INSPECTIONS.labels(model, body["decision"]).inc()
    if heatmap:
        body["overlay_png"] = png_base64(overlay(gray, result.anomaly_map, result.threshold))
    return body


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if (WEB_DIR / "samples").exists():
    app.mount("/samples", StaticFiles(directory=WEB_DIR / "samples"), name="samples")
