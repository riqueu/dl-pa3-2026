"""Testes unitários do módulo `src/metrics.py`."""

import numpy as np

from src.dataset import DEFAULT_CLASSES
from src.metrics import (
    compute_confusion_matrix,
    compute_macro_f1_and_accuracy,
    compute_ood_scores,
    compute_per_class_f1,
)


def test_macro_f1_penalizes_missing_rare_class():
    y_true = [0] * 10 + [1] * 2
    y_pred = [0] * 10 + [0] * 2
    res = compute_macro_f1_and_accuracy(y_true, y_pred, num_classes=2)
    assert res["accuracy"] > 0.8
    assert res["f1_macro"] < 0.6


def test_per_class_f1_covers_all_21_species():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 0]
    res = compute_per_class_f1(y_true, y_pred, DEFAULT_CLASSES)
    assert set(res.keys()) == set(DEFAULT_CLASSES)
    assert res[DEFAULT_CLASSES[1]] == 1.0
    assert res[DEFAULT_CLASSES[2]] == 0.0


def test_confusion_matrix_shape_and_normalization():
    y_true = [0, 0, 1, 2]
    y_pred = [0, 1, 1, 2]
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=21, normalize=False)
    assert cm.shape == (21, 21)
    assert cm.sum() == 4

    cm_norm = compute_confusion_matrix(y_true, y_pred, num_classes=21, normalize=True)
    assert np.isclose(cm_norm[0].sum(), 1.0)
    assert np.isclose(cm_norm[5].sum(), 0.0)


def test_ood_scores_distinguish_confident_from_uniform():
    confident_logits = np.array([[8.0] + [0.0] * 20], dtype=np.float32)
    uniform_logits = np.zeros((1, 21), dtype=np.float32)

    id_scores = compute_ood_scores(confident_logits, temperature=1.0)
    ood_scores = compute_ood_scores(uniform_logits, temperature=1.0)

    assert id_scores["msp"][0] > ood_scores["msp"][0]
    assert ood_scores["entropy"][0] > id_scores["entropy"][0]
    assert ood_scores["energy"][0] > id_scores["energy"][0]
