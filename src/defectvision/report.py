"""Report figures (README) built from the predictions saved in artifacts/."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import data
from .inference import overlay

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COLORS = {"patchcore": "#2f6f9f", "unet": "#c4572a"}
NAMES = {"patchcore": "PatchCore (unsupervised)", "unet": "U-Net (supervised)"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _load(name: str, artifacts: Path):
    path = artifacts / f"{name}_test.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {
        "scores": z["scores"],
        "maps": z["maps"].astype(np.float32),
        "files": [str(f) for f in z["files"]],
        **json.loads((artifacts / f"{name}_metrics.json").read_text()),
    }


def roc_points(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s)
    y = y[order].astype(bool)
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(~y) / (~y).sum()])
    return fpr, tpr


def figures(cfg: dict, out: Path, artifacts: str | Path = "artifacts") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = Path(artifacts)
    out.mkdir(parents=True, exist_ok=True)
    runs = {n: r for n in ("patchcore", "unet") if (r := _load(n, artifacts))}
    by_file = {str(s.image): s for s in data.index(cfg["data_dir"])}

    # 1. ROC curves (image level) with operating point
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    for name, r in runs.items():
        y = np.array([by_file[f].is_defect for f in r["files"]])
        fpr, tpr = roc_points(y, r["scores"])
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{NAMES[name]} — AUROC {r['image_auroc']:.3f}")
        op = r["operating_point"]
        ax.plot(op["false_reject_rate"], op["recall"], "o", color=COLORS[name], ms=7, mec="white", mew=1.5)
    ax.plot([0, 1], [0, 1], ls="--", color=GRID, lw=1)
    ax.set_xlabel("Good pieces wrongly rejected", color=INK)
    ax.set_ylabel("Defects detected", color=INK)
    ax.set_title("Piece-level detection (test set)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "roc.png")
    plt.close(fig)

    # 2. data leakage: split by image vs by piece
    leak = artifacts / "patchcore_random_split_metrics.json"
    if leak.exists() and "patchcore" in runs:
        rnd = json.loads(leak.read_text())
        grp = runs["patchcore"]
        labels = ["Image AUROC", "Pixel AUROC", "AUPRO (30%)"]
        keys = ["image_auroc", "pixel_auroc", "aupro_30"]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=150)
        b1 = ax.bar(x - 0.19, [rnd[k] for k in keys], 0.36, color="#b8c4ce", label="Random split by image (leakage)")
        b2 = ax.bar(x + 0.19, [grp[k] for k in keys], 0.36, color=COLORS["patchcore"], label="Split by physical piece")
        for bars in (b1, b2):
            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{b.get_height():.3f}",
                    ha="center",
                    fontsize=8,
                    color=INK,
                )
        ax.set_xticks(x, labels)
        ax.set_ylim(0.5, 1.04)
        ax.set_title("PatchCore: effect of train/test leakage", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        _style(ax)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(out / "leakage.png")
        plt.close(fig)

    # 3. qualitative examples: one defect of each type
    if runs:
        ref = next(iter(runs.values()))
        rng = np.random.default_rng(cfg["seed"])
        picks = []
        for c in data.CLASSES[1:] + ["good"]:
            idx = [i for i, f in enumerate(ref["files"]) if by_file[f].label == c]
            if idx:
                picks.append(int(rng.choice(idx)))
        cols = 2 + len(runs)
        fig, axes = plt.subplots(len(picks), cols, figsize=(2.1 * cols, 2.1 * len(picks)), dpi=130)
        size = cfg["image_size"]
        for row, i in enumerate(picks):
            s = by_file[ref["files"][i]]
            gray = cv2.resize(data.read_gray(s.image), (size, size), interpolation=cv2.INTER_AREA)
            panels = [(gray, f"{s.label}"), (data.read_mask(s.mask, size) * 255, "ground truth")]
            for name, r in runs.items():
                j = r["files"].index(ref["files"][i])
                thr = r["operating_point"]["threshold"]
                verdict = "REJECT" if r["scores"][j] >= thr else "pass"
                panels.append((cv2.cvtColor(overlay(gray, r["maps"][j], thr), cv2.COLOR_BGR2RGB), f"{name}: {verdict}"))
            for col, (img, title) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.set_title(title, fontsize=8, color=INK)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "examples.png")
        plt.close(fig)

    # 4. U-Net learning curve
    hist = artifacts / "unet_history.json"
    if hist.exists():
        h = json.loads(hist.read_text())
        ep = [e["epoch"] for e in h]
        fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150)
        ax.plot(ep, [e["image_auroc"] for e in h], color=COLORS["unet"], lw=2, label="Image AUROC (validation)")
        ax.plot(ep, [e["dice"] for e in h], color="#7a8b99", lw=2, label="Defect Dice (validation)")
        ax.set_xlabel("Epoch", color=INK)
        ax.set_ylim(0, 1)
        ax.set_title("U-Net: training", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        _style(ax)
        fig.tight_layout()
        fig.savefig(out / "unet_training.png")
        plt.close(fig)
