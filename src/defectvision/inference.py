"""Inférence de production : ONNX Runtime + NumPy + OpenCV uniquement (pas de PyTorch dans l'image Docker)."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .data import CLASSES, preprocess


@dataclass
class Result:
    model: str
    score: float
    threshold: float
    is_defect: bool
    defect_type: str | None
    defect_area_pct: float
    latency_ms: float
    anomaly_map: np.ndarray  # (S, S), même échelle que le score

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "anomaly_map"}
        d["score"], d["threshold"] = round(self.score, 5), round(self.threshold, 5)
        d["defect_area_pct"], d["latency_ms"] = round(self.defect_area_pct, 2), round(self.latency_ms, 1)
        d["decision"] = "REJET" if self.is_defect else "CONFORME"
        return d


class OnnxInspector:
    def __init__(self, path: str | Path, threads: int = 0):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.kind = meta["kind"]
        self.threshold = float(meta["threshold"])
        self.image_size = int(meta["image_size"])
        self.outputs = [o.name for o in self.session.get_outputs()]

    def inspect(self, gray: np.ndarray) -> Result:
        t0 = time.perf_counter()
        x = preprocess(gray, self.image_size)[None]
        out = dict(zip(self.outputs, self.session.run(None, {"image": x}), strict=True))
        score, amap = float(out["score"][0]), out["anomaly_map"][0]
        is_defect = score >= self.threshold
        defect_type = None
        if is_defect and "class_probs" in out:
            probs = out["class_probs"][0]
            defect_type = CLASSES[int((probs[1:] * amap[None]).sum((1, 2)).argmax()) + 1]
        area = float((amap >= self.threshold).mean() * 100) if is_defect else 0.0
        return Result(
            self.kind, score, self.threshold, is_defect, defect_type, area, (time.perf_counter() - t0) * 1000, amap
        )


def overlay(gray: np.ndarray, amap: np.ndarray, threshold: float, alpha: float = 0.45) -> np.ndarray:
    """Superpose la carte d'anomalie (en couleur, saturée au seuil x 1,5) et le contour des zones rejetées."""
    h, w = gray.shape[:2]
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    m = cv2.resize(amap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((np.clip(m / (1.5 * threshold), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
    contours, _ = cv2.findContours((m >= threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def png_base64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("encodage PNG impossible")
    return base64.b64encode(buf.tobytes()).decode()
