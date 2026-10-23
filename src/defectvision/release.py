"""Passage en production : export ONNX des modèles entraînés et mesure de latence CPU."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from .inference import OnnxInspector


def _threshold(name: str, artifacts: Path) -> float:
    return json.loads((artifacts / f"{name}_metrics.json").read_text())["operating_point"]["threshold"]


def export_all(cfg: dict, artifacts: str | Path = "artifacts", models_dir: str | Path = "models") -> dict:
    import torch

    from .backbone import resnet18
    from .export import export, parity
    from .patchcore import PatchCore
    from .segmentation import Inspector, UNet

    artifacts, models_dir = Path(artifacts), Path(models_dir)
    size, out = cfg["image_size"], {}

    if (artifacts / "patchcore.pt").exists():
        state = torch.load(artifacts / "patchcore.pt")
        pc = PatchCore(resnet18(pretrained=False), image_size=size, blur_sigma=cfg["patchcore"]["blur_sigma"])
        pc.memory, pc.memory_sq = state["memory"], state["memory_sq"]
        pc.load_state_dict(state)
        path = export(pc.eval(), models_dir / "patchcore.onnx", "patchcore", _threshold("patchcore", artifacts), size)
        out["patchcore"] = {
            "max_abs_diff_vs_torch": parity(pc, path, size),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }

    if (artifacts / "unet.pt").exists():
        net = UNet(pretrained=False)
        net.load_state_dict(torch.load(artifacts / "unet.pt"))
        model = Inspector(net.eval())
        path = export(model, models_dir / "unet.onnx", "unet", _threshold("unet", artifacts), size)
        out["unet"] = {
            "max_abs_diff_vs_torch": parity(model, path, size),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/export.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def benchmark(cfg: dict, models_dir: str | Path = "models", runs: int = 50, threads: int = 0) -> dict:
    """Latence de bout en bout d'une inspection (prétraitement + ONNX Runtime + post-traitement), lot de 1."""
    import os

    from .data import index, read_gray

    images = [read_gray(s.image) for s in index(cfg["data_dir"])[::97][:10]]
    res = {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(), "runs": runs}
    for path in sorted(Path(models_dir).glob("*.onnx")):
        insp = OnnxInspector(path, threads=threads)
        for img in images[:3]:
            insp.inspect(img)  # préchauffage
        lat = []
        for i in range(runs):
            t0 = time.perf_counter()
            insp.inspect(images[i % len(images)])
            lat.append((time.perf_counter() - t0) * 1000)
        res[path.stem] = {
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "images_per_s": round(1000 / float(np.mean(lat)), 1),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark.json").write_text(json.dumps(res, indent=2))
    return res


def quantize_int8(src: str | Path, dst: str | Path, calibration: list[np.ndarray]) -> Path:
    """Quantification statique INT8 (format QDQ, par canal), calibrée sur des images de VALIDATION."""
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": x[None]} for x in calibration])

        def get_next(self):
            return next(self.it, None)

    src, dst = Path(src), Path(dst)
    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_static(
        str(pre),
        str(dst),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    pre.unlink()
    # recopie des métadonnées (seuil, type de modèle) perdues par la quantification
    meta = {p.key: p.value for p in onnx.load(str(src), load_external_data=False).metadata_props}
    m = onnx.load(str(dst))
    del m.metadata_props[:]
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(dst))
    return dst
