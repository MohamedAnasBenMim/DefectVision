import importlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from .conftest import texture


@pytest.fixture
def client(tmp_path, monkeypatch):
    import torch

    from defectvision.backbone import resnet18
    from defectvision.export import export
    from defectvision.patchcore import PatchCore
    from defectvision.segmentation import Inspector, UNet

    rng = np.random.default_rng(0)
    from defectvision import data

    pc = PatchCore(resnet18(pretrained=False), image_size=64, blur_sigma=2)
    pc.fit(
        [torch.from_numpy(np.stack([data.preprocess(texture(rng, 64), 64) for _ in range(4)]))],
        0.1,
        log=lambda *_: None,
    )
    export(
        pc, tmp_path / "patchcore.onnx", "patchcore", threshold=0.0, image_size=64
    )  # threshold 0 -> everything is rejected
    export(
        Inspector(UNet(pretrained=False)), tmp_path / "unet.onnx", "unet", threshold=2.0, image_size=64
    )  # never rejected
    monkeypatch.setenv("DV_MODELS_DIR", str(tmp_path))
    from prometheus_client import REGISTRY

    import defectvision.api as api

    # reloading re-creates the metrics: unregister the previous ones from the global registry first
    for collector in (api.INSPECTIONS, api.LATENCY):
        REGISTRY.unregister(collector)
    importlib.reload(api)
    return TestClient(api.app)


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok", "models": ["patchcore", "unet"]}
    kinds = {m["name"]: m["kind"] for m in client.get("/models").json()}
    assert kinds == {"patchcore": "patchcore", "unet": "unet"}


def test_inspect_returns_decision_and_overlay(client):
    img = texture(np.random.default_rng(1), 120)
    r = client.post("/inspect?model=patchcore", files={"file": ("tile.png", png(img), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "REJECT" and body["is_defect"] is True and body["defect_area_pct"] > 0
    overlay = cv2.imdecode(
        np.frombuffer(__import__("base64").b64decode(body["overlay_png"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert overlay.shape[:2] == img.shape  # overlay at the original image size

    r = client.post("/inspect?model=unet&heatmap=false", files={"file": ("tile.png", png(img), "image/png")})
    assert r.json()["decision"] == "PASS" and "overlay_png" not in r.json()


def test_inspect_errors(client):
    assert client.post("/inspect?model=nope", files={"file": ("a.png", b"x", "image/png")}).status_code == 404
    assert client.post("/inspect", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 422


def test_metrics_endpoint(client):
    client.post(
        "/inspect?model=unet", files={"file": ("t.png", png(texture(np.random.default_rng(2), 64)), "image/png")}
    )
    text = client.get("/metrics").text
    assert 'dv_inspections_total{decision="PASS",model="unet"}' in text
