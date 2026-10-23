"""Magnetic Tile Defects dataset (Huang et al., 2018): download, indexing and splitting.

Key point: each physical tile is photographed under 6 lighting conditions (prefixes exp1...exp6), with
very close `num` identifiers. A random split by IMAGE puts the same piece in both the training and the
test set, which inflates the scores. Images are therefore grouped by piece (`group`) and split by group.
"""

from __future__ import annotations

import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = "https://github.com/abin24/Magnetic-tile-defect-datasets..git"
COMMIT = "cc4eb530b57b9cb7225bf76023873043ffd50655"

CLASSES = ["good", "blowhole", "break", "crack", "fray", "uneven"]  # index 0 = good piece
FOLDERS = {
    "MT_Free": "good",
    "MT_Blowhole": "blowhole",
    "MT_Break": "break",
    "MT_Crack": "crack",
    "MT_Fray": "fray",
    "MT_Uneven": "uneven",
}
_NAME = re.compile(r"exp(\d+)_num_(\d+)\.jpg$")
GROUP_GAP = 200  # maximum identifier gap between two shots of the same piece

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    label: str  # "good" or defect type
    exposure: int
    num: int
    group: str  # physical piece identifier

    @property
    def is_defect(self) -> bool:
        return self.label != "good"

    @property
    def class_index(self) -> int:
        return CLASSES.index(self.label)


def fetch(data_dir: str | Path) -> Path:
    """Clone the public dataset repository at a pinned commit (reproducibility)."""
    data_dir = Path(data_dir)
    if (data_dir / "MT_Free").exists():
        return data_dir
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", REPO, str(data_dir)], check=True)
    subprocess.run(["git", "-C", str(data_dir), "checkout", "--quiet", COMMIT], check=True)
    return data_dir


def index(data_dir: str | Path) -> list[Sample]:
    """List all images with their mask, class and group (physical piece)."""
    data_dir = Path(data_dir)
    samples: list[Sample] = []
    for folder, label in FOLDERS.items():
        items = []
        for img in (data_dir / folder / "Imgs").glob("*.jpg"):
            m = _NAME.search(img.name)
            if m:
                items.append((int(m.group(2)), int(m.group(1)), img))
        items.sort()
        group_id, last_num, exposures = -1, None, set()
        for num, exp, img in items:
            # new piece if the identifier jumps, or if this lighting condition is already taken in the current group
            if last_num is None or num - last_num > GROUP_GAP or exp in exposures:
                group_id += 1
                exposures = set()
            exposures.add(exp)
            last_num = num
            samples.append(Sample(img, img.with_suffix(".png"), label, exp, num, f"{label}-{group_id:03d}"))
    if not samples:
        raise FileNotFoundError(f"no images found in {data_dir} (run `dv fetch-data`)")
    return samples


def split(samples: list[Sample], val: float, test: float, seed: int, by_group: bool = True) -> dict[str, list[Sample]]:
    """Class-stratified split. `by_group=False` reproduces the common mistake (splitting by image)."""
    rng = random.Random(seed)
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    by_label: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)
    for label in sorted(by_label):
        units: dict[str, list[Sample]] = defaultdict(list)
        for s in by_label[label]:
            units[s.group if by_group else str(s.image)].append(s)
        keys = sorted(units)
        rng.shuffle(keys)
        n_test = max(1, round(len(keys) * test))
        n_val = max(1, round(len(keys) * val))
        for i, k in enumerate(keys):
            part = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out[part].extend(units[k])
    return out


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img


def read_mask(path: str | Path, size: int) -> np.ndarray:
    m = read_gray(path)
    return (cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)


def preprocess(gray: np.ndarray, size: int) -> np.ndarray:
    """Grayscale image (H, W) -> ImageNet-normalized float32 tensor (3, size, size)."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.repeat(x[None], 3, axis=0)
    return (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
