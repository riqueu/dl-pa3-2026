"""Testes unitários e de integração para predict.py, train.py, evaluate.py e curva com orcamento=0."""

import os
import pandas as pd
from PIL import Image

from predict import _normalize_cli_args, predict_directory
from src.dataset import DEFAULT_CLASSES
from src.models import SmallCNN, save_student_checkpoint
from src.visualization import plot_budget_curve


def test_normalize_single_and_en_dash_cli_args():
    raw = ["-imagens", "/tmp/imgs", "–saida", "/tmp/out.csv"]
    norm = _normalize_cli_args(raw)
    assert norm == ["--imagens", "/tmp/imgs", "--saida", "/tmp/out.csv"]


def test_predict_directory_cpu_offline_and_schema(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name in ("img_b", "img_a"):
        Image.new("RGB", (64, 64), color=(80, 120, 60)).save(img_dir / f"{name}.jpg")

    ckpt_path = tmp_path / "student.pth"
    model = SmallCNN(num_classes=len(DEFAULT_CLASSES))
    save_student_checkpoint(
        model,
        str(ckpt_path),
        arch="small_cnn",
        num_classes=len(DEFAULT_CLASSES),
        extra={"class_names": list(DEFAULT_CLASSES)},
    )

    out_csv = tmp_path / "preds.csv"
    df = predict_directory(
        imagens=str(img_dir),
        saida=str(out_csv),
        checkpoint=str(ckpt_path),
        batch_size=2,
    )
    assert out_csv.is_file()
    loaded = pd.read_csv(out_csv)
    assert list(loaded.columns) == ["image_id", "label"]
    assert list(loaded["image_id"]) == ["img_a", "img_b"]
    assert set(loaded["label"]).issubset(set(DEFAULT_CLASSES))
    assert len(df) == 2


def test_plot_budget_curve_with_zero_shot_row(tmp_path):
    df = pd.DataFrame(
        [
            {"metodo": "professor_bioclip_zero_shot", "orcamento": 0, "seed": 0, "f1_macro": 0.45, "acuracia": 0.50},
            {"metodo": "piso_resnet18", "orcamento": 1, "seed": 42, "f1_macro": 0.12, "acuracia": 0.14},
            {"metodo": "piso_resnet18", "orcamento": 5, "seed": 42, "f1_macro": 0.30, "acuracia": 0.35},
            {"metodo": "piso_resnet18", "orcamento": 5, "seed": 43, "f1_macro": 0.32, "acuracia": 0.36},
            {"metodo": "trilha_c_aluno", "orcamento": 1, "seed": 42, "f1_macro": 0.35, "acuracia": 0.38},
            {"metodo": "trilha_c_aluno", "orcamento": 5, "seed": 42, "f1_macro": 0.50, "acuracia": 0.53},
            {"metodo": "trilha_c_aluno", "orcamento": 20, "seed": 42, "f1_macro": 0.57, "acuracia": 0.60},
        ]
    )
    out_png = tmp_path / "curva.png"
    fig = plot_budget_curve(df, save_path=str(out_png))
    assert os.path.isfile(out_png)
    import matplotlib.pyplot as plt
    plt.close(fig)
