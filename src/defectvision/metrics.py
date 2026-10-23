"""Métriques d'inspection : AUROC image et pixel, AUPRO, point de fonctionnement, intervalles de confiance."""

from __future__ import annotations

import cv2
import numpy as np


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Aire sous la courbe ROC par la statistique de Mann-Whitney (rangs moyens en cas d'égalité)."""
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # rangs moyens pour les ex aequo
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray, max_pixels: int = 4_000_000, seed: int = 0) -> float:
    m, s = masks.ravel(), maps.ravel()
    if len(m) > max_pixels:  # sous-échantillonnage uniforme, sans biais
        idx = np.random.default_rng(seed).choice(len(m), max_pixels, replace=False)
        m, s = m[idx], s[idx]
    return auroc(m, s)


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, steps: int = 300) -> float:
    """Aire sous la courbe PRO (per-region overlap) jusqu'à un taux de faux positifs de 30 %, normalisée.

    Contrairement à l'AUROC pixel, chaque défaut compte autant, quelle que soit sa surface : un petit trou
    raté pèse autant qu'une grande zone bien détectée.
    """
    regions = []
    for mask, amap in zip(masks, maps, strict=True):
        n, lab = cv2.connectedComponents(mask.astype(np.uint8))
        for r in range(1, n):
            regions.append(np.sort(amap[lab == r]))
    if not regions:
        return float("nan")
    normal = np.sort(maps[masks == 0])
    lo = np.quantile(normal, 1 - fpr_limit)
    thresholds = np.linspace(normal[-1], lo, steps)
    fpr = 1 - np.searchsorted(normal, thresholds, side="right") / len(normal)
    pro = np.mean([1 - np.searchsorted(r, thresholds, side="right") / len(r) for r in regions], axis=0)
    fpr = np.concatenate([[0.0], fpr, [fpr_limit]])
    pro = np.concatenate([[0.0], pro, [pro[-1]]])
    keep = fpr <= fpr_limit
    return float(np.trapezoid(pro[keep], fpr[keep]) / fpr_limit)


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Plus grand seuil qui détecte au moins `target_recall` des pièces défectueuses."""
    pos = np.sort(np.asarray(scores)[np.asarray(labels).astype(bool)])
    k = int(np.floor((1 - target_recall) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def operating_point(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp, fn = int((pred & labels).sum()), int((~pred & labels).sum())
    fp, tn = int((pred & ~labels).sum()), int((~pred & ~labels).sum())
    return {
        "threshold": float(threshold),
        "recall": tp / max(tp + fn, 1),  # défauts détectés
        "escape_rate": fn / max(tp + fn, 1),  # défauts qui partent chez le client
        "false_reject_rate": fp / max(fp + tn, 1),  # pièces saines rejetées à tort
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def bootstrap_auroc(
    labels: np.ndarray, scores: np.ndarray, groups: list[str], n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """IC à 95 % de l'AUROC image, en rééchantillonnant des PIÈCES (et non des images corrélées)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        v = auroc(np.asarray(labels)[idx], np.asarray(scores)[idx])
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else float("nan")
