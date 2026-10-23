"""Expériences reproductibles : entraînement, évaluation et rapport de métriques (JSON)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import data, metrics
from .backbone import resnet18
from .patchcore import PatchCore


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_split(cfg: dict, by_group: bool = True) -> dict[str, list[data.Sample]]:
    samples = data.index(cfg["data_dir"])
    return data.split(samples, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"], by_group=by_group)


def load_images(samples: list[data.Sample], size: int) -> torch.Tensor:
    return torch.from_numpy(np.stack([data.preprocess(data.read_gray(s.image), size) for s in samples]))


def load_masks(samples: list[data.Sample], size: int) -> np.ndarray:
    return np.stack([data.read_mask(s.mask, size) for s in samples])


def split_summary(split: dict[str, list[data.Sample]]) -> dict:
    out = {}
    for part, items in split.items():
        out[part] = {
            "images": len(items),
            "pieces": len({s.group for s in items}),
            "defect_images": sum(s.is_defect for s in items),
            "good_images": sum(not s.is_defect for s in items),
        }
    return out


def evaluate_scores(
    cfg: dict, val: list[data.Sample], val_scores: np.ndarray, test: list[data.Sample], test_scores, test_maps, size
) -> dict:
    """Métriques image + pixel sur le test, seuil calibré sur la validation."""
    y_val = np.array([s.is_defect for s in val])
    y_test = np.array([s.is_defect for s in test])
    thr = metrics.threshold_for_recall(y_val, val_scores, cfg["operating_point"]["target_recall"])
    masks = load_masks(test, size)
    lo, hi = metrics.bootstrap_auroc(y_test, test_scores, [s.group for s in test])
    per_class = {}
    for c in data.CLASSES[1:]:
        sel = np.array([s.label in ("good", c) for s in test])
        per_class[c] = round(metrics.auroc(y_test[sel], test_scores[sel]), 4)
    return {
        "image_auroc": round(metrics.auroc(y_test, test_scores), 4),
        "image_auroc_ci95": [round(lo, 4), round(hi, 4)],
        "image_auroc_per_defect": per_class,
        "pixel_auroc": round(metrics.pixel_auroc(masks, test_maps), 4),
        "aupro_30": round(metrics.aupro(masks, test_maps), 4),
        "operating_point": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.operating_point(y_test, test_scores, thr).items()
        },
    }


def run_patchcore(
    cfg: dict, by_group: bool = True, out_dir: str | Path = "artifacts", log=print
) -> tuple[PatchCore, dict]:
    seed_everything(cfg["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    size = cfg["image_size"]
    pc_cfg = cfg["patchcore"]
    split = get_split(cfg, by_group)
    train_good = [s for s in split["train"] if not s.is_defect]  # PatchCore ne voit AUCUN défaut
    model = PatchCore(resnet18(), image_size=size, blur_sigma=pc_cfg["blur_sigma"])

    t0 = time.perf_counter()
    batches = (load_images(train_good[i : i + 16], size) for i in range(0, len(train_good), 16))
    model.fit(batches, pc_cfg["coreset_ratio"], pc_cfg["projection_dim"], cfg["seed"], log=log)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_scores, _ = model.predict(load_images(split["val"], size))
    test_scores, test_maps = model.predict(load_images(split["test"], size))
    infer_ms = (time.perf_counter() - t0) / (len(split["val"]) + len(split["test"])) * 1000

    report = {
        "model": "patchcore",
        "split": "pièces (groupes)" if by_group else "images (aléatoire)",
        "split_summary": split_summary(split),
        "training_images": len(train_good),
        "defect_labels_used": 0,
        "memory_bank": int(model.memory.shape[0]),
        "fit_seconds": round(fit_s, 1),
        "torch_cpu_ms_per_image": round(infer_ms, 1),
        **evaluate_scores(cfg, split["val"], val_scores, split["test"], test_scores, test_maps, size),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "patchcore" if by_group else "patchcore_random_split"
    np.savez_compressed(
        out / f"{tag}_test.npz",
        scores=test_scores,
        maps=test_maps.astype(np.float16),
        files=[str(s.image) for s in split["test"]],
    )
    torch.save(model.state_dict(), out / f"{tag}.pt")
    (out / f"{tag}_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return model, report


def run_unet(cfg: dict, out_dir: str | Path = "artifacts", log=print):
    from . import segmentation

    seed_everything(cfg["seed"])
    size = cfg["image_size"]
    split = get_split(cfg, True)
    model, train_info = segmentation.train(cfg, split, out_dir, log=log)

    val_scores, _, _ = segmentation.predict(model, split["val"], size)
    t0 = time.perf_counter()
    test_scores, test_maps, test_cls = segmentation.predict(model, split["test"], size)
    infer_ms = (time.perf_counter() - t0) / len(split["test"]) * 1000

    # segmentation : IoU par type de défaut (pixels prédits de ce type vs vérité terrain)
    masks = load_masks(split["test"], size)
    per_class_iou = {}
    with torch.no_grad():
        for c in data.CLASSES[1:]:
            ious = []
            for s, amap, m in zip(split["test"], test_maps, masks, strict=True):
                if s.label == c:
                    ious.append(metrics.iou(amap > 0.5, m.astype(bool)))
            per_class_iou[c] = round(float(np.nanmean(ious)), 4) if ious else None
    defects = [i for i, s in enumerate(split["test"]) if s.is_defect]
    type_acc = float(np.mean([test_cls[i] == split["test"][i].class_index for i in defects]))
    confusion = np.zeros((len(data.CLASSES) - 1, len(data.CLASSES) - 1), dtype=int)
    for i in defects:
        confusion[split["test"][i].class_index - 1, test_cls[i] - 1] += 1

    report = {
        "model": "unet",
        "split": "pièces (groupes)",
        "split_summary": split_summary(split),
        "training_images": len(split["train"]),
        "defect_labels_used": sum(s.is_defect for s in split["train"]),
        "best_epoch": train_info["best_epoch"],
        "fit_seconds": train_info["train_seconds"],
        "torch_cpu_ms_per_image": round(infer_ms, 1),
        **evaluate_scores(cfg, split["val"], val_scores, split["test"], test_scores, test_maps, size),
        "mean_iou_per_defect": per_class_iou,
        "defect_type_accuracy": round(type_acc, 4),
        "defect_type_confusion": {"labels": data.CLASSES[1:], "matrix": confusion.tolist()},
    }
    out = Path(out_dir)
    np.savez_compressed(
        out / "unet_test.npz",
        scores=test_scores,
        maps=test_maps.astype(np.float16),
        classes=test_cls,
        files=[str(s.image) for s in split["test"]],
    )
    (out / "unet_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return model, report
