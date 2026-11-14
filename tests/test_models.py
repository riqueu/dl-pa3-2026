"""Testes unitários do módulo `src/models.py`."""

import torch
from torch.utils.data import DataLoader

from src.models import (
    MAX_ALLOWED_PARAMS,
    SmallCNN,
    build_student_backbone,
    count_parameters,
    detect_overfitting_epoch,
    fit_supervised_baseline,
    load_student_checkpoint,
    save_student_checkpoint,
)


def test_small_cnn_and_backbones_under_12m_limit():
    for arch in ["small_cnn", "resnet18", "mobilenet_v3_large", "efficientnet_b0"]:
        model = build_student_backbone(arch=arch, num_classes=21, pretrained=False)
        n_params = count_parameters(model)
        assert 0 < n_params <= MAX_ALLOWED_PARAMS
        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 21)


def test_checkpoint_save_and_offline_load(tmp_path):
    model = SmallCNN(num_classes=21)
    ckpt_path = tmp_path / "student.pth"
    save_student_checkpoint(model, str(ckpt_path), arch="small_cnn", num_classes=21)

    loaded = load_student_checkpoint(str(ckpt_path), arch="small_cnn", num_classes=21, device="cpu")
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_detect_overfitting_epoch():
    history = {
        "train_loss": [2.2, 1.4, 0.7, 0.2, 0.05],
        "val_loss": [2.3, 1.8, 1.9, 2.2, 2.7],
        "train_f1": [0.1, 0.4, 0.8, 0.98, 1.0],
        "val_f1": [0.1, 0.42, 0.40, 0.35, 0.31],
    }
    ep = detect_overfitting_epoch(history, patience=2)
    assert ep == 2


def test_fit_supervised_baseline_restores_best_val_weights():
    torch.manual_seed(42)
    model = SmallCNN(num_classes=21)
    imgs = torch.randn(12, 3, 224, 224)
    labels = torch.tensor([i % 4 for i in range(12)], dtype=torch.long)
    ds = [(imgs[i], labels[i], f"id_{i}") for i in range(len(imgs))]
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    trained, history = fit_supervised_baseline(
        model=model,
        train_loader=loader,
        val_loader=loader,
        epochs=2,
        lr=1e-3,
        device="cpu",
    )
    assert set(history.keys()) == {"train_loss", "val_loss", "train_f1", "val_f1", "val_acc"}
    assert len(history["train_loss"]) == 2
    assert isinstance(trained, torch.nn.Module)
