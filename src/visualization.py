"""Módulo de visualização para o PA3 (Partes 0, 1, 2 e 3).

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-baseline
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent_dir(save_path: Optional[str]) -> None:
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def plot_training_curves(
    history: Dict[str, List[float]],
    overfit_epoch: Optional[int] = None,
    title: str = "Curvas de Treino e Validação",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota Loss e F1 Macro (Treino vs. Validação) com marcação de overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    ax_loss = axes[0]
    if "train_loss" in history:
        ax_loss.plot(epochs, history["train_loss"], label="Treino Loss", marker="o", markersize=3)
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    if overfit_epoch is not None:
        ax_loss.axvline(
            overfit_epoch,
            color="crimson",
            linestyle="--",
            alpha=0.8,
            label=f"Início memorização (época {overfit_epoch})",
        )
    ax_loss.set_title("Perda (Cross-Entropy)")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_metric = axes[1]
    if "train_f1" in history:
        ax_metric.plot(epochs, history["train_f1"], label="Treino F1 Macro", marker="o", markersize=3)
    if "val_f1" in history:
        ax_metric.plot(epochs, history["val_f1"], label="Val F1 Macro", marker="s", markersize=3)
    if overfit_epoch is not None:
        ax_metric.axvline(
            overfit_epoch,
            color="crimson",
            linestyle="--",
            alpha=0.8,
            label=f"Época {overfit_epoch}",
        )
    ax_metric.set_title("F1 Macro")
    ax_metric.set_xlabel("Época")
    ax_metric.set_ylabel("F1 Macro")
    ax_metric.set_ylim(0.0, 1.05)
    ax_metric.grid(True, alpha=0.3)
    ax_metric.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusão",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota matriz de confusão 21×21 com anotações."""
    cm_arr = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        display_cm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums > 0)
        fmt = ".2f"
        cbar_label = "Proporção por Classe Verdadeira"
    else:
        display_cm = cm_arr
        fmt = ".0f"
        cbar_label = "Contagem"

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.52), max(8.5, n_classes * 0.46)))
    im = ax.imshow(
        display_cm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=(1.0 if normalize else None),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ticks = np.arange(n_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = float(display_cm.max()) * 0.55 if display_cm.size > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = display_cm[i, j]
            if val > 0:
                text_val = format(val, fmt) if normalize else str(int(round(cm_arr[i, j])))
                ax.text(
                    j,
                    i,
                    text_val,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if val > thresh else "black",
                )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Classe Predita")
    ax.set_ylabel("Classe Verdadeira")
    fig.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_budget_curve(
    df_results: pd.DataFrame,
    ceiling_f1: float = 0.686,
    teacher_f1: Optional[float] = None,
    title: str = "Curva de Orçamento — F1 Macro no test_public",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota o gráfico obrigatório da Parte 2 (eixo X em escala logarítmica)."""
    required = {"metodo", "orcamento", "f1_macro"}
    missing = required - set(df_results.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em df_results: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    palette = {
        "piso": "#1f77b4",
        "baseline": "#1f77b4",
        "trilha_c": "#2ca02c",
        "distill": "#2ca02c",
    }

    for method_name, group in df_results.groupby("metodo", sort=False):
        agg = (
            group.groupby("orcamento")["f1_macro"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("orcamento")
        )
        agg["std"] = agg["std"].fillna(0.0)
        color = palette.get(str(method_name).lower(), None)

        ax.errorbar(
            agg["orcamento"],
            agg["mean"],
            yerr=agg["std"],
            marker="o",
            linewidth=2.2,
            capsize=5,
            label=str(method_name),
            color=color,
        )
        for _, row in agg.iterrows():
            ax.annotate(
                f"{row['mean']:.3f}",
                (row["orcamento"], row["mean"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
            )

    ax.axhline(
        ceiling_f1,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Teto (todos os rótulos, ResNet-18): {ceiling_f1:.3f}",
    )
    if teacher_f1 is not None:
        ax.axhline(
            teacher_f1,
            color="darkorange",
            linestyle="-.",
            linewidth=1.6,
            label=f"Professor Zero-Shot (BioCLIP): {teacher_f1:.3f}",
        )

    ax.set_xscale("log")
    budgets = sorted(df_results["orcamento"].unique().tolist())
    ax.set_xticks(budgets)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Orçamento (rótulos por classe, escala log)", fontsize=10.5)
    ax.set_ylabel("F1 Macro (test_public)", fontsize=10.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, max(1.0, ceiling_f1 + 0.12))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_gains(
    per_class_baseline: Dict[str, float],
    per_class_method: Dict[str, float],
    per_class_teacher: Optional[Dict[str, float]] = None,
    title: str = "F1 por Classe no Orçamento de 5 Rótulos",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compara F1 por espécie e ganho ΔF1 entre Piso e Trilha C (Partes 1.3 e 2.3)."""
    classes = sorted(
        per_class_method.keys(),
        key=lambda c: per_class_method[c] - per_class_baseline.get(c, 0.0),
        reverse=True,
    )
    y = np.arange(len(classes))
    base_vals = np.array([per_class_baseline.get(c, 0.0) for c in classes])
    meth_vals = np.array([per_class_method.get(c, 0.0) for c in classes])
    deltas = meth_vals - base_vals

    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(6.5, len(classes) * 0.34)), sharey=True
    )

    ax_left = axes[0]
    height = 0.27 if per_class_teacher is not None else 0.38
    ax_left.barh(y - height / 2, base_vals, height=height, label="Piso (ResNet-18)", color="#4c72b0")
    ax_left.barh(y + height / 2, meth_vals, height=height, label="Aluno Trilha C", color="#55a868")
    if per_class_teacher is not None:
        teach_vals = np.array([per_class_teacher.get(c, 0.0) for c in classes])
        ax_left.barh(
            y + 1.5 * height,
            teach_vals,
            height=height,
            label="Professor BioCLIP",
            color="#dd8452",
        )
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(classes, fontsize=8.5)
    ax_left.invert_yaxis()
    ax_left.set_xlim(0.0, 1.05)
    ax_left.set_xlabel("F1 Score")
    ax_left.set_title("F1 Absoluto por Espécie")
    ax_left.grid(True, axis="x", alpha=0.3)
    ax_left.legend(loc="lower right")

    ax_right = axes[1]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    ax_right.barh(y, deltas, color=colors, height=0.6)
    ax_right.axvline(0.0, color="black", linewidth=1.0)
    ax_right.set_xlabel("Ganho ΔF1 (Trilha C − Piso)")
    ax_right.set_title("Ganho por Classe (Orçamento 5)")
    ax_right.grid(True, axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def overlay_heatmap(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Sobrepõe mapa de saliência normalizado `[0, 1]` a uma imagem RGB uint8."""
    img = np.asarray(img_rgb, dtype=np.float32)
    if float(img.max()) > 1.0:
        img = img / 255.0

    hm = np.asarray(heatmap, dtype=np.float32)
    hm_min, hm_max = float(hm.min()), float(hm.max())
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm, dtype=np.float32)

    cmap = plt.get_cmap(colormap)
    hm_color = cmap(hm)[..., :3]
    blended = (1.0 - alpha) * img + alpha * hm_color
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def plot_saliency_gallery(
    cases: List[Dict],
    title: str = "Galeria de Saliência (Grad-CAM)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Renderiza grade imagem original vs. Grad-CAM para a Parte 3."""
    n_cases = len(cases)
    if n_cases == 0:
        raise ValueError("Lista de casos para galeria de saliência está vazia.")

    fig, axes = plt.subplots(n_cases, 2, figsize=(10.5, 3.6 * n_cases))
    if n_cases == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, case in enumerate(cases):
        img_rgb = case["image_rgb"]
        heatmap = case["heatmap"]
        overlay = overlay_heatmap(img_rgb, heatmap)

        true_lbl = case.get("true_label", "?")
        pred_lbl = case.get("pred_label", "?")
        conf = float(case.get("confidence", 0.0))
        img_id = case.get("image_id", f"caso_{idx}")
        note = case.get("note", "")

        ax_orig, ax_cam = axes[idx, 0], axes[idx, 1]
        ax_orig.imshow(img_rgb)
        ax_orig.set_title(f"[{img_id}] Real: {true_lbl}", fontsize=9.5)
        ax_orig.axis("off")

        ax_cam.imshow(overlay)
        subtitle = f"Pred: {pred_lbl} ({conf * 100:.1f}%)"
        if note:
            subtitle += f"\n{note}"
        ax_cam.set_title(subtitle, fontsize=9.0)
        ax_cam.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
