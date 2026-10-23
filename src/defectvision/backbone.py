"""ImageNet-pretrained ResNet-18, without depending on a model hub.

The weights come from the ONNX Model Zoo (pinned file, SHA-256 checksum verified) and are copied into
the equivalent torchvision architecture. `tests/test_models.py` checks that the PyTorch and ONNX Runtime
outputs match.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

WEIGHTS_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
WEIGHTS_SHA256 = "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c"
DEFAULT_PATH = Path("weights/resnet18-v1-7.onnx")


def fetch_weights(path: str | Path = DEFAULT_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"unexpected checksum for {path}: {digest}")
    return path


def onnx_to_state_dict(onnx_path: str | Path) -> dict:
    """Map each Conv/BatchNorm/Gemm node of the ONNX graph to the corresponding torchvision module."""
    import onnx
    import torch
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    producer = {}  # tensor name -> node that produces it
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    # convolution order in torchvision; the 1x1 convolutions are the shortcuts (downsample)
    blocks = [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)]
    downsample = iter(["layer2.0.downsample", "layer3.0.downsample", "layer4.0.downsample"])
    convs3 = iter([f"{b}.conv{k}" for b in blocks for k in (1, 2)])
    conv_name: dict[str, str] = {}
    sd: dict[str, torch.Tensor] = {}
    first = True
    for node in model.graph.node:
        if node.op_type == "Conv":
            w = init[node.input[1]]
            if first:
                name, first = "conv1", False
            elif w.shape[-1] == 1:
                name = next(downsample) + ".0"
            else:
                name = next(convs3)
            sd[f"{name}.weight"] = torch.from_numpy(w.copy())
            conv_name[node.output[0]] = name
        elif node.op_type == "BatchNormalization":
            conv = conv_name[producer[node.input[0]].output[0]]
            bn = {"conv1": "bn1"}.get(conv) or (conv[:-2] + ".1" if conv.endswith(".0") else conv.replace("conv", "bn"))
            for key, idx in (("weight", 1), ("bias", 2), ("running_mean", 3), ("running_var", 4)):
                sd[f"{bn}.{key}"] = torch.from_numpy(init[node.input[idx]].copy())
            sd[f"{bn}.num_batches_tracked"] = torch.tensor(0)
        elif node.op_type == "Gemm":
            sd["fc.weight"] = torch.from_numpy(init[node.input[1]].copy())
            sd["fc.bias"] = torch.from_numpy(init[node.input[2]].copy())
    return sd


def resnet18(onnx_path: str | Path | None = None, pretrained: bool = True):
    """torchvision ResNet-18, with ImageNet weights if `pretrained`."""
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    if pretrained:
        net.load_state_dict(onnx_to_state_dict(fetch_weights(onnx_path or DEFAULT_PATH)))
    return net.eval()


def check_parity(onnx_path: str | Path, n: int = 2, seed: int = 0) -> float:
    """Maximum absolute difference between PyTorch logits (converted weights) and ONNX Runtime (original file)."""
    import onnxruntime as ort
    import torch

    x = np.random.default_rng(seed).standard_normal((n, 3, 224, 224)).astype(np.float32)
    ref = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"data": x})[0]
    with torch.no_grad():
        out = resnet18(onnx_path)(torch.from_numpy(x)).numpy()
    return float(np.abs(out - ref).max())
