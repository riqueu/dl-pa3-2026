"""Módulo de arquiteturas do aluno (Parte 0 e modelo final) e treino supervisionado.

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-baseline
"""

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.metrics import compute_macro_f1_and_accuracy


MAX_ALLOWED_PARAMS: int = 12_000_000


def count_parameters(model: nn.Module) -> int:
    """Conta parâmetros conforme a regra oficial do edital (<= 12.000.000)."""
    return sum(p.numel() for p in model.parameters())


class SmallCNN(nn.Module):
    """CNN leve treinada do zero para a Parte 0 (a) — O Piso sem pré-treino."""

    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_student_backbone(
    arch: str = "resnet18",
    num_classes: int = 21,
    pretrained: bool = True,
) -> nn.Module:
    """Constrói a arquitetura do aluno garantindo limite <= 12M parâmetros.

    Quando `pretrained=False`, não realiza download de pesos (necessário para
    `predict.py` rodar offline em CPU).
    """
    name = arch.lower().strip()
    if name == "small_cnn":
        model: nn.Module = SmallCNN(num_classes=num_classes)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name in {"mobilenet_v3_large", "mobilenetv3"}:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(
            f"Arquitetura desconhecida '{arch}'. "
            "Use 'small_cnn', 'resnet18', 'mobilenet_v3_large' ou 'efficientnet_b0'."
        )

    n_params = count_parameters(model)
    if n_params > MAX_ALLOWED_PARAMS:
        raise ValueError(
            f"Arquitetura '{arch}' possui {n_params:,} parâmetros, "
            f"acima do limite de {MAX_ALLOWED_PARAMS:,}."
        )
    return model


def save_student_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    arch: str = "resnet18",
    num_classes: int = 21,
    extra: Optional[Dict] = None,
) -> None:
    """Salva o checkpoint no contrato padronizado do PA3."""
    parent = os.path.dirname(checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload: Dict = {
        "model_state_dict": model.state_dict(),
        "arch": arch,
        "num_classes": int(num_classes),
        "n_params": count_parameters(model),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)


def load_student_checkpoint(
    checkpoint_path: str,
    arch: str = "resnet18",
    num_classes: int = 21,
    device: str = "cpu",
) -> nn.Module:
    """Carrega um checkpoint do aluno sem acesso à internet (`pretrained=False`)."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        ckpt_arch = str(payload.get("arch", arch))
        ckpt_classes = int(payload.get("num_classes", num_classes))
        state_dict = payload["model_state_dict"]
    else:
        ckpt_arch = arch
        ckpt_classes = num_classes
        state_dict = payload

    model = build_student_backbone(
        arch=ckpt_arch, num_classes=ckpt_classes, pretrained=False
    )
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    return model


def detect_overfitting_epoch(
    history: Dict[str, List[float]],
    patience: int = 2,
) -> int:
    """Identifica a época (1-indexada) em que o modelo começa a decorar o treino.

    Critério: primeira época `t` em que `val_loss` sobe por `patience` épocas
    consecutivas enquanto `train_loss` segue caindo. Caso não ocorra subida
    monotônica suficiente, retorna a época do menor `val_loss`.
    """
    val_losses = history.get("val_loss", [])
    train_losses = history.get("train_loss", [])
    if not val_losses:
        val_f1s = history.get("val_f1", [])
        if not val_f1s:
            return 1
        return int(np.argmax(val_f1s)) + 1

    n = len(val_losses)
    if n <= patience:
        return int(np.argmin(val_losses)) + 1

    for idx in range(n - patience):
        val_rising = all(
            val_losses[idx + step + 1] > val_losses[idx + step]
            for step in range(patience)
        )
        train_falling = (
            train_losses[idx + patience] < train_losses[idx]
            if len(train_losses) == n
            else True
        )
        if val_rising and train_falling:
            return idx + 1

    return int(np.argmin(val_losses)) + 1


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["image"], batch["label"], batch["image_id"]
    return batch[0], batch[1], batch[2]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    num_classes: int = 21,
    criterion: Optional[nn.Module] = None,
) -> Dict:
    """Avalia um modelo PyTorch em um `DataLoader` retornando loss, métricas e logits."""
    target_device = torch.device(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    logits_chunks: List[np.ndarray] = []
    image_ids: List[str] = []

    for batch in loader:
        images, labels, batch_ids = _unpack_batch(batch)
        images = images.to(target_device)
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=target_device)

        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, labels_t)
            total_loss += float(loss.item()) * images.size(0)

        total_samples += images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels_t.cpu().tolist())
        logits_chunks.append(logits.cpu().numpy())
        image_ids.extend([str(i) for i in batch_ids])

    metrics = compute_macro_f1_and_accuracy(y_true, y_pred, num_classes=num_classes)
    avg_loss = (total_loss / total_samples) if (criterion is not None and total_samples > 0) else 0.0

    return {
        "loss": float(avg_loss),
        "f1_macro": float(metrics["f1_macro"]),
        "accuracy": float(metrics["accuracy"]),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, num_classes)),
        "image_ids": image_ids,
    }


def _build_optimizer_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
) -> List[Dict]:
    if isinstance(model, SmallCNN) or backbone_lr_mult == 1.0:
        return [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc.") or name.startswith("classifier."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    if not head_params or not backbone_params:
        return [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]

    return [
        {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr, "weight_decay": weight_decay},
    ]


def fit_supervised_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
    backbone_lr_mult: float = 0.25,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Treina o modelo supervisionado (Piso) selecionando a melhor época no `val`.

    Nunca consulta `test_public`. Restaura ao final o `state_dict` de maior
    `val_f1` (desempatando por menor `val_loss`).
    """
    target_device = torch.device(device)
    model = model.to(target_device)

    weight_tensor = class_weights.to(target_device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    param_groups = _build_optimizer_param_groups(
        model, lr=lr, weight_decay=weight_decay, backbone_lr_mult=backbone_lr_mult
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )

    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            num_classes = model.fc.out_features
        elif hasattr(model, "classifier") and isinstance(model.classifier[-1], nn.Linear):
            num_classes = model.classifier[-1].out_features
        else:
            num_classes = 21

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "val_acc": [],
    }

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        train_true: List[int] = []
        train_pred: List[int] = []

        for batch in train_loader:
            images, labels, _ = _unpack_batch(batch)
            images = images.to(target_device)
            labels_t = torch.as_tensor(labels, dtype=torch.long, device=target_device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            total_train += images.size(0)
            train_true.extend(labels_t.cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        scheduler.step()

        train_loss = running_loss / max(1, total_train)
        train_metrics = compute_macro_f1_and_accuracy(
            train_true, train_pred, num_classes=num_classes
        )
        val_res = evaluate_model(
            model, val_loader, device=device, num_classes=num_classes, criterion=criterion
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_res["loss"]))
        history["train_f1"].append(float(train_metrics["f1_macro"]))
        history["val_f1"].append(float(val_res["f1_macro"]))
        history["val_acc"].append(float(val_res["accuracy"]))

        improved = val_res["f1_macro"] > best_val_f1 or (
            np.isclose(val_res["f1_macro"], best_val_f1)
            and val_res["loss"] < best_val_loss
        )
        if improved:
            best_val_f1 = float(val_res["f1_macro"])
            best_val_loss = float(val_res["loss"])
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model, history
