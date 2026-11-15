"""Módulo de métricas de classificação desbalanceada e sinais OOD (PA3).

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-baseline
"""

from typing import Dict, Sequence

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_macro_f1_and_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 21,
) -> Dict[str, float]:
    """Calcula F1 Macro (métrica principal do PA3) e acurácia global.

    F1 Macro atribui o mesmo peso a todas as `num_classes` classes, evitando que
    classes raras (ex.: `horse`) sejam escondidas por classes frequentes (`urso`).
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    if y_true_arr.size == 0:
        raise ValueError("y_true não pode ser vazio.")

    labels = list(range(num_classes))
    f1_macro = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    return {"f1_macro": f1_macro, "accuracy": acc}


def compute_per_class_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, float]:
    """Calcula o F1 score individual de cada espécie."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(len(class_names)))
    scores = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {str(name): float(score) for name, score in zip(class_names, scores)}


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 21,
    normalize: bool = False,
) -> np.ndarray:
    """Computa a matriz de confusão `(num_classes, num_classes)`."""
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if not normalize:
        return cm

    cm_float = cm.astype(np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    return np.divide(
        cm_float,
        row_sums,
        out=np.zeros_like(cm_float),
        where=row_sums > 0,
    )


def compute_ood_scores(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Calcula sinais Out-of-Distribution (OOD) para responder ao caso da capivara.

    Retorna:
        - `msp`: Maximum Softmax Probability (menor em amostras OOD).
        - `entropy`: Entropia preditiva `-sum(p * log(p))` (maior em amostras OOD).
        - `energy`: Energia livre `-T * logsumexp(z / T)` (maior/menos negativa em OOD).
    """
    if temperature <= 0.0:
        raise ValueError("temperature deve ser estritamente positiva.")

    logits_arr = np.asarray(logits, dtype=np.float64)
    if logits_arr.ndim == 1:
        logits_arr = logits_arr[np.newaxis, :]
    if logits_arr.ndim != 2:
        raise ValueError(f"logits deve ser 1D ou 2D, recebido shape={logits_arr.shape}.")

    scaled = logits_arr / float(temperature)
    lse = logsumexp(scaled, axis=-1, keepdims=True)
    probs = np.exp(scaled - lse)

    msp = np.max(probs, axis=-1)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
    energy = -float(temperature) * lse.squeeze(-1)

    return {
        "msp": msp.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "energy": energy.astype(np.float32),
    }
