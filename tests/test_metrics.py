import numpy as np
import pytest

from defectvision import metrics


def test_auroc_matches_sklearn_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(y * 0.8, 1), 1)  # rounding -> many ties
    assert metrics.auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)


def test_aupro_perfect_and_random():
    masks = np.zeros((4, 32, 32), np.uint8)
    masks[0, 5:10, 5:10] = 1
    masks[1, 20:30, 2:4] = 1
    assert metrics.aupro(masks, masks.astype(float)) == pytest.approx(1.0, abs=1e-6)
    noise = np.random.default_rng(0).random(masks.shape)
    assert metrics.aupro(masks, noise) < 0.3


def test_threshold_for_recall_and_operating_point():
    y = np.array([1] * 20 + [0] * 80)
    s = np.concatenate([np.linspace(0.5, 1.0, 20), np.linspace(0.0, 0.6, 80)])
    thr = metrics.threshold_for_recall(y, s, 0.95)
    op = metrics.operating_point(y, s, thr)
    assert op["recall"] >= 0.95
    assert op["escape_rate"] == pytest.approx(1 - op["recall"])
    assert op["tp"] + op["fn"] == 20 and op["fp"] + op["tn"] == 80


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 60)
    s = y + rng.normal(0, 0.8, 120)
    groups = [f"g{i // 6}" for i in range(120)]
    lo, hi = metrics.bootstrap_auroc(y, s, groups, n=300)
    assert lo <= metrics.auroc(y, s) <= hi
