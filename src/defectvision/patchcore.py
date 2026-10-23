"""PatchCore (Roth et al., CVPR 2022), implemented from scratch.

Idea: learn from GOOD pieces only. Local descriptors (patches) are extracted from a pretrained network,
and a representative subset (greedy k-center coreset) is kept in a memory bank; at test time, the
distance from a patch to its nearest good neighbour measures how anomalous it is.

The whole pipeline (feature extraction, distances, anomaly map, blur) is a single `nn.Module`, so it
exports as-is to one ONNX file, memory bank included.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, int(round(4 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    return torch.outer(k, k)[None, None]


class PatchCore(nn.Module):
    def __init__(self, backbone: nn.Module, image_size: int = 256, blur_sigma: float = 4.0, feat_dim: int = 384):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2, self.layer3 = backbone.layer2, backbone.layer3
        self.image_size = image_size
        self.register_buffer("memory", torch.zeros(1, feat_dim))
        self.register_buffer("memory_sq", torch.zeros(1))
        self.register_buffer("kernel", gaussian_kernel(blur_sigma))
        self.eval()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) -> (B, H*W, 384): layers 2 and 3, 3x3 local smoothing, same resolution."""
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        return f.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image score (B,), anomaly map (B, S, S))."""
        b = x.shape[0]
        f = self.features(x)  # (B, N, D)
        side = int(math.isqrt(f.shape[1]))
        d2 = (f * f).sum(-1, keepdim=True) - 2 * f @ self.memory.T + self.memory_sq  # (B, N, K)
        dist = d2.min(dim=-1).values.clamp_min(0).sqrt()
        amap = dist.reshape(b, 1, side, side)
        amap = F.interpolate(amap, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        pad = self.kernel.shape[-1] // 2
        amap = F.conv2d(F.pad(amap, (pad, pad, pad, pad), mode="replicate"), self.kernel)[:, 0]
        return amap.flatten(1).max(dim=1).values, amap

    @torch.no_grad()
    def fit(self, batches, coreset_ratio: float = 0.02, projection_dim: int = 128, seed: int = 0, log=print) -> None:
        feats = torch.cat([self.features(x).reshape(-1, self.memory.shape[1]) for x in batches])
        log(f"good patches: {feats.shape[0]:,} x {feats.shape[1]}")
        n = max(1, int(feats.shape[0] * coreset_ratio))
        idx = greedy_coreset(feats, n, projection_dim, seed)
        self.memory = feats[idx].contiguous()
        self.memory_sq = (self.memory**2).sum(-1)
        log(f"memory bank: {n:,} patches ({coreset_ratio:.1%})")

    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
        scores, maps = [], []
        for i in range(0, len(x), batch_size):
            s, m = self(x[i : i + batch_size])
            scores.append(s)
            maps.append(m)
        return torch.cat(scores).numpy(), torch.cat(maps).numpy()


def greedy_coreset(feats: torch.Tensor, n: int, projection_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center selection: each new point is the one farthest from the points already selected.

    Covers the space of good patches with n points. A random projection (Johnson-Lindenstrauss) reduces
    the dimension to speed up the distance computations.
    """
    g = torch.Generator().manual_seed(seed)
    if projection_dim and projection_dim < feats.shape[1]:
        proj = torch.randn(feats.shape[1], projection_dim, generator=g) / math.sqrt(projection_dim)
        z = feats @ proj
    else:
        z = feats
    z = z.contiguous()
    sq = (z * z).sum(-1)
    selected = torch.empty(n, dtype=torch.long)
    selected[0] = torch.randint(len(z), (1,), generator=g)
    # ||z - c||^2 = ||z||^2 - 2 z.c + ||c||^2: one matrix-vector product, no temporary (N, d) array
    min_d = sq - 2 * (z @ z[selected[0]]) + sq[selected[0]]
    for i in range(1, n):
        c = int(torch.argmax(min_d))
        selected[i] = c
        torch.minimum(min_d, sq - 2 * (z @ z[c]) + sq[c], out=min_d)
    return selected
