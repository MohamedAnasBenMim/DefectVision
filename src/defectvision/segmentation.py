"""Supervised U-Net (ImageNet ResNet-18 encoder): per-pixel defect segmentation and classification.

Output: 6 channels (good + 5 defect types). Pixel anomaly score = 1 - P(good); image score = maximum of
that map after light smoothing (a single isolated pixel is not enough to reject a piece).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import data, metrics
from .backbone import resnet18

N_CLASSES = len(data.CLASSES)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, pretrained: bool = True, n_classes: int = N_CLASSES):
        super().__init__()
        r = resnet18(pretrained=pretrained)
        self.enc0 = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2, 64
        self.enc1 = nn.Sequential(r.maxpool, r.layer1)  # /4, 64
        self.enc2, self.enc3, self.enc4 = r.layer2, r.layer3, r.layer4  # /8 128, /16 256, /32 512
        self.dec4 = conv_block(512 + 256, 256)
        self.dec3 = conv_block(256 + 128, 128)
        self.dec2 = conv_block(128 + 64, 64)
        self.dec1 = conv_block(64 + 64, 32)
        self.dec0 = conv_block(32, 16)
        self.head = nn.Conv2d(16, n_classes, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.dec4(self._up(e4, e3))
        d = self.dec3(self._up(d, e2))
        d = self.dec2(self._up(d, e1))
        d = self.dec1(self._up(d, e0))
        d = self.dec0(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(d)


class Inspector(nn.Module):
    """Exportable inference wrapper: (image score, anomaly map, per-class probabilities)."""

    def __init__(self, unet: UNet, smooth: int = 5):
        super().__init__()
        self.unet, self.smooth = unet.eval(), smooth

    def forward(self, x: torch.Tensor):
        probs = self.unet(x).softmax(1)
        amap = 1 - probs[:, 0]
        pooled = F.avg_pool2d(amap[:, None], self.smooth, stride=1, padding=self.smooth // 2)[:, 0]
        return pooled.flatten(1).max(1).values, amap, probs


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean Dice over the defect classes present in the batch (the background is handled by cross-entropy)."""
    probs = logits.softmax(1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((0, 2, 3))
    total = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = (2 * inter + eps) / (total + eps)
    present = onehot.sum((0, 2, 3))[1:] > 0
    return 1 - dice[1:][present].mean() if present.any() else logits.sum() * 0


def augment(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Flips and brightness/contrast jitter (lighting varies from one shot to the next)."""
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1], mask[::-1]
    alpha, beta = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def load_pair(s: data.Sample, size: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    img = cv2.resize(data.read_gray(s.image), (size, size), interpolation=cv2.INTER_AREA)
    mask = data.read_mask(s.mask, size) * s.class_index  # binary mask -> class index
    return img, mask.astype(np.int64)


def to_tensor(gray: np.ndarray, size: int) -> np.ndarray:
    return data.preprocess(gray, size)


@torch.no_grad()
def predict(model: Inspector, samples: list[data.Sample], size: int, batch_size: int = 8):
    scores, maps, classes = [], [], []
    for i in range(0, len(samples), batch_size):
        x = torch.from_numpy(
            np.stack([data.preprocess(data.read_gray(s.image), size) for s in samples[i : i + batch_size]])
        )
        sc, amap, probs = model(x)
        scores.append(sc.numpy())
        maps.append(amap.numpy())
        # defect type = most likely non-good class, weighted over the whole image
        classes.append((probs[:, 1:] * amap[:, None]).sum((2, 3)).argmax(1).numpy() + 1)
    return np.concatenate(scores), np.concatenate(maps), np.concatenate(classes)


def quick_val(model: Inspector, samples: list[data.Sample], size: int) -> dict:
    scores, maps, _ = predict(model, samples, size)
    y = np.array([s.is_defect for s in samples])
    dices = []
    for s, m in zip(samples, maps, strict=True):
        if s.is_defect:
            gt = data.read_mask(s.mask, size).astype(bool)
            pred = m > 0.5
            dices.append(2 * (gt & pred).sum() / max(gt.sum() + pred.sum(), 1))
    return {"image_auroc": metrics.auroc(y, scores), "dice": float(np.mean(dices))}


def train(
    cfg: dict, split: dict[str, list[data.Sample]], out_dir: str | Path = "artifacts", log=print
) -> tuple[Inspector, dict]:
    size, ucfg = cfg["image_size"], cfg["unet"]
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    defects = [s for s in split["train"] if s.is_defect]
    goods = [s for s in split["train"] if not s.is_defect]
    cache = {s.image: load_pair(s, size) for s in split["train"]}

    net = UNet(pretrained=True)
    opt = torch.optim.AdamW(net.parameters(), lr=ucfg["lr"], weight_decay=ucfg["weight_decay"])
    steps_per_epoch = (len(defects) + ucfg["good_per_epoch"]) // ucfg["batch_size"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ucfg["lr"], total_steps=ucfg["epochs"] * steps_per_epoch)
    best, history = (-1.0, None), []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    for epoch in range(ucfg["epochs"]):
        net.train()
        items = defects + rng.sample(goods, min(len(goods), ucfg["good_per_epoch"]))
        rng.shuffle(items)
        losses = []
        for i in range(0, steps_per_epoch * ucfg["batch_size"], ucfg["batch_size"]):
            batch = [augment(*cache[s.image], rng) for s in items[i : i + ucfg["batch_size"]]]
            x = torch.from_numpy(np.stack([to_tensor(b[0], size) for b in batch]))
            y = torch.from_numpy(np.stack([b[1] for b in batch]))
            logits = net(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        net.eval()
        val = quick_val(Inspector(net), split["val"], size)
        sel = (val["image_auroc"] + val["dice"]) / 2
        history.append(
            {"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 4), **{k: round(v, 4) for k, v in val.items()}}
        )
        log(
            f"epoch {epoch + 1:>2}/{ucfg['epochs']}  loss {np.mean(losses):.4f}  val AUROC {val['image_auroc']:.4f}  Dice {val['dice']:.4f}  ({time.perf_counter() - t_start:.0f} s)"
        )
        if sel > best[0]:
            best = (sel, epoch + 1)
            torch.save(net.state_dict(), out / "unet.pt")
    net.load_state_dict(torch.load(out / "unet.pt"))
    (out / "unet_history.json").write_text(json.dumps(history, indent=1))
    return Inspector(net.eval()), {
        "best_epoch": best[1],
        "train_seconds": round(time.perf_counter() - t_start, 1),
        "history": history,
    }
