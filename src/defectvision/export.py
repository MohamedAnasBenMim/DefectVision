"""Export ONNX autonome : un seul fichier par modèle, seuil de décision et métadonnées inclus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def export(
    model: torch.nn.Module, path: str | Path, kind: str, threshold: float, image_size: int, extra: dict | None = None
) -> Path:
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    x = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        n_out = len(model(x))
    names = ["score", "anomaly_map", "class_probs"][:n_out]
    torch.onnx.export(
        model,
        (x,),
        str(path),
        input_names=["image"],
        output_names=names,
        dynamic_axes={"image": {0: "batch"}, **{n: {0: "batch"} for n in names}},
        opset_version=17,
        dynamo=False,
    )
    m = onnx.load(str(path))
    meta = {
        "kind": kind,
        "threshold": f"{threshold:.6f}",
        "image_size": str(image_size),
        **{k: json.dumps(v) for k, v in (extra or {}).items()},
    }
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(path))
    return path


def parity(model: torch.nn.Module, path: str | Path, image_size: int, n: int = 2, seed: int = 0) -> float:
    """Écart maximal entre PyTorch et ONNX Runtime sur le score et la carte d'anomalie."""
    import onnxruntime as ort

    x = np.random.default_rng(seed).standard_normal((n, 3, image_size, image_size)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["score", "anomaly_map"], {"image": x})
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x))[:2]
    return max(float(np.abs(a - b.numpy()).max()) for a, b in zip(onnx_out, torch_out, strict=True))
