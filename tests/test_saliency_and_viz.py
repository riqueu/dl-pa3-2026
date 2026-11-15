"""Testes unitários de `src/saliency.py` e `src/visualization.py`."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.dataset import DEFAULT_CLASSES, ENA24Dataset, get_transforms
from src.models import build_student_backbone
from src.saliency import (
    GradCAM,
    occlusion_sensitivity,
    select_failure_and_success_cases,
)
from src.visualization import (
    overlay_heatmap,
    plot_budget_curve,
    plot_confusion_matrix,
    plot_per_class_gains,
    plot_saliency_gallery,
    plot_training_curves,
)


def test_gradcam_and_occlusion_on_resnet18():
    model = build_student_backbone(arch="resnet18", num_classes=21, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    cam = GradCAM(model)
    heatmap = cam.generate(x)
    cam.close()

    assert heatmap.shape == (224, 224)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0 + 1e-6

    occ = occlusion_sensitivity(model, x, patch_size=56, stride=56)
    assert occ.shape == (224, 224)
    assert float(occ.min()) >= 0.0
    assert float(occ.max()) <= 1.0 + 1e-6


def test_select_failure_and_success_cases_and_gallery_plot(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for i in range(12):
        img_id = f"sample_{i}"
        color = (i * 20, 100, 150) if i % 2 == 0 else (110, 110, 110)
        Image.new("RGB", (160, 120), color=color).save(img_dir / f"{img_id}.jpg")
        rows.append({"image_id": img_id, "label": DEFAULT_CLASSES[i % 4]})

    csv_path = tmp_path / "test_public.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ds = ENA24Dataset(
        image_dir=str(img_dir),
        csv_path=str(csv_path),
        transform=get_transforms(224, is_train=False),
    )
    model = build_student_backbone(arch="resnet18", num_classes=21, pretrained=False)

    gallery = select_failure_and_success_cases(
        model=model,
        dataset=ds,
        class_names=DEFAULT_CLASSES,
        device="cpu",
        n_errors=2,
        n_successes=1,
    )
    cases = gallery["errors"] + gallery["successes"]
    assert len(cases) >= 1
    for c in cases:
        assert c["heatmap"].shape == (120, 160)
        assert c["image_rgb"].shape == (120, 160, 3)

    out_png = tmp_path / "gallery.png"
    fig = plot_saliency_gallery(cases, save_path=str(out_png))
    plt.close(fig)
    assert os.path.isfile(out_png)


def test_visualization_plots_save_outputs(tmp_path):
    history = {
        "train_loss": [2.5, 1.5, 0.8, 0.3],
        "val_loss": [2.6, 1.9, 2.0, 2.4],
        "train_f1": [0.1, 0.4, 0.7, 0.95],
        "val_f1": [0.1, 0.35, 0.33, 0.30],
    }
    p1 = tmp_path / "curves.png"
    fig1 = plot_training_curves(history, overfit_epoch=2, save_path=str(p1))
    plt.close(fig1)
    assert os.path.isfile(p1)

    cm = np.eye(21, dtype=np.int64) * 5
    p2 = tmp_path / "cm.png"
    fig2 = plot_confusion_matrix(cm, DEFAULT_CLASSES, save_path=str(p2))
    plt.close(fig2)
    assert os.path.isfile(p2)

    df_budget = pd.DataFrame([
        {"metodo": "piso", "orcamento": 1, "seed": 42, "f1_macro": 0.15},
        {"metodo": "piso", "orcamento": 5, "seed": 42, "f1_macro": 0.35},
        {"metodo": "piso", "orcamento": 5, "seed": 123, "f1_macro": 0.37},
        {"metodo": "piso", "orcamento": 20, "seed": 42, "f1_macro": 0.50},
        {"metodo": "trilha_c", "orcamento": 1, "seed": 42, "f1_macro": 0.52},
        {"metodo": "trilha_c", "orcamento": 5, "seed": 42, "f1_macro": 0.61},
        {"metodo": "trilha_c", "orcamento": 5, "seed": 123, "f1_macro": 0.63},
        {"metodo": "trilha_c", "orcamento": 20, "seed": 42, "f1_macro": 0.67},
    ])
    p3 = tmp_path / "budget.png"
    fig3 = plot_budget_curve(df_budget, ceiling_f1=0.686, teacher_f1=0.58, save_path=str(p3))
    plt.close(fig3)
    assert os.path.isfile(p3)

    base = {c: 0.3 for c in DEFAULT_CLASSES}
    meth = {c: 0.5 for c in DEFAULT_CLASSES}
    p4 = tmp_path / "gains.png"
    fig4 = plot_per_class_gains(base, meth, save_path=str(p4))
    plt.close(fig4)
    assert os.path.isfile(p4)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    hm = np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
    ov = overlay_heatmap(img, hm)
    assert ov.shape == (64, 64, 3) and ov.dtype == np.uint8
