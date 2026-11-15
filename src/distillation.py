"""Destilação professor -> aluno (Trilha C).

Responsável: Membro 2 (Isaias)
Branch: feature/teacher-distillation

O aluno (ResNet-18 ImageNet, <= 12M parâmetros) é treinado com:
  * CE nos rótulos humanos de `rotulos_K.csv`;
  * KL/CE suave com temperatura nos pseudo-rótulos filtrados do pool `train`.
O balanceamento por classe entra pela amostragem (`sample_weights`). A melhor
época é escolhida pelo F1 macro no `val`; o `test_public` não é tocado aqui.

Uso:
    python -m src.distillation --budget 5 --seed 42 --checkpoint checkpoints/student_K5_s42.pth
"""

import argparse
import copy
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.dataset import ENA24Dataset, get_transforms, load_class_mapping
from src.metrics import compute_macro_f1_and_accuracy
from src.models import (
    _build_optimizer_param_groups,
    build_student_backbone,
    count_parameters,
    evaluate_model,
    save_student_checkpoint,
)
from src.pseudo_labels import PSEUDO_DIR, make_pseudo_labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DistillationLoss(nn.Module):
    """`lambda_h * CE(humanos) + T^2 * CE_suave(pseudo, z/T)`, média no batch.

    A CE suave `-sum q log softmax(z/T)` difere da KL(q || p) só pela entropia de
    q (constante), então os gradientes são os mesmos da KL.
    """

    def __init__(self, temperature: float = 2.0, human_lambda: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.human_lambda = float(human_lambda)

    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        hard_targets: torch.Tensor,
        is_human: torch.Tensor,
    ) -> torch.Tensor:
        is_human = is_human.bool()
        total = logits.new_zeros(())
        if is_human.any():
            ce = F.cross_entropy(logits[is_human], hard_targets[is_human], reduction="sum")
            total = total + self.human_lambda * ce
        pseudo = ~is_human
        if pseudo.any():
            t = self.temperature
            log_p = F.log_softmax(logits[pseudo] / t, dim=1)
            soft_ce = -(soft_targets[pseudo] * log_p).sum(dim=1).sum()
            total = total + (t * t) * soft_ce
        return total / logits.size(0)


class DistillDataset(Dataset):
    """Pool `train` filtrado com rótulo suave, rótulo duro e flag de humano."""

    def __init__(self, image_dir: str, pl: Dict[str, np.ndarray], transform):
        keep = np.where(pl["keep_mask"])[0]
        self.base = ENA24Dataset(
            image_dir, image_ids=[str(i) for i in pl["image_ids"][keep]], transform=transform
        )
        self.soft = torch.from_numpy(pl["soft_labels"][keep].astype(np.float32))
        self.hard = torch.from_numpy(pl["hard_labels"][keep].astype(np.int64))
        self.is_human = torch.from_numpy(pl["is_human_labeled"][keep].astype(bool))
        self.weights = pl["sample_weights"][keep].astype(np.float64)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, _, _ = self.base[idx]
        return image, self.soft[idx], self.hard[idx], self.is_human[idx]


def fit_student(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: DistillationLoss,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    backbone_lr_mult: float = 0.1,
    device: str = "cpu",
    num_classes: int = 21,
    log_fn=print,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Treina o aluno e restaura o estado de melhor F1 macro no `val`."""
    dev = torch.device(device)
    model = model.to(dev)
    optimizer = torch.optim.AdamW(
        _build_optimizer_param_groups(model, lr, weight_decay, backbone_lr_mult)
    )
    steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=steps,
        pct_start=0.1,
    )
    val_ce = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_f1_human_pseudo": [], "val_loss": [], "val_f1": [], "val_acc": []
    }
    best_f1, best_loss, best_epoch = -1.0, float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n_seen, y_t, y_p = 0.0, 0, [], []
        for images, soft, hard, is_h in train_loader:
            images, soft, hard, is_h = images.to(dev), soft.to(dev), hard.to(dev), is_h.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, soft, hard, is_h)
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss += float(loss.item()) * images.size(0)
            n_seen += images.size(0)
            y_t.extend(hard.cpu().tolist())
            y_p.extend(logits.argmax(1).detach().cpu().tolist())

        val = evaluate_model(model, val_loader, device=device, num_classes=num_classes, criterion=val_ce)
        tr = compute_macro_f1_and_accuracy(y_t, y_p, num_classes)
        history["train_loss"].append(run_loss / max(1, n_seen))
        history["train_f1_human_pseudo"].append(tr["f1_macro"])
        history["val_loss"].append(val["loss"])
        history["val_f1"].append(val["f1_macro"])
        history["val_acc"].append(val["accuracy"])

        if val["f1_macro"] > best_f1 or (np.isclose(val["f1_macro"], best_f1) and val["loss"] < best_loss):
            best_f1, best_loss, best_epoch = val["f1_macro"], val["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
        log_fn(
            f"[aluno] época {epoch}/{epochs} loss {history['train_loss'][-1]:.4f} "
            f"val_f1 {val['f1_macro']:.4f} val_acc {val['accuracy']:.4f} ({time.time() - t0:.0f}s)"
        )

    history["best_epoch"] = [best_epoch]
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def train_distilled_student(
    data_dir: str = "data/pa3_dados",
    budget: int = 5,
    seed: int = 42,
    checkpoint: Optional[str] = None,
    arch: str = "resnet18",
    epochs: int = 10,
    epoch_size: Optional[int] = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    backbone_lr_mult: float = 0.1,
    kd_temperature: float = 2.0,
    human_lambda: float = 1.0,
    teacher_temperature: float = 1.0,
    conf_threshold: float = 0.5,
    min_per_class: int = 20,
    num_workers: int = 6,
    device: Optional[str] = None,
    pseudo_dir: str = PSEUDO_DIR,
) -> Tuple[nn.Module, Dict]:
    """Pipeline completo da Trilha C para um orçamento K e uma seed."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    class_names, _, _ = load_class_mapping(data_dir)
    n_cls = len(class_names)

    pl = make_pseudo_labels(
        data_dir,
        budget,
        out_dir=pseudo_dir,
        temperature=teacher_temperature,
        conf_threshold=conf_threshold,
        min_per_class=min_per_class,
    )

    train_ds = DistillDataset(
        os.path.join(data_dir, "images", "train"), pl, get_transforms(224, is_train=True)
    )
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        train_ds.weights, num_samples=epoch_size or len(train_ds), replacement=True, generator=gen
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        drop_last=True, persistent_workers=num_workers > 0,
    )
    val_ds = ENA24Dataset(
        os.path.join(data_dir, "images", "val"),
        csv_path=os.path.join(data_dir, "labels", "val.csv"),
        transform=get_transforms(224, is_train=False),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=num_workers)

    model = build_student_backbone(arch, n_cls, pretrained=True)
    print(
        f"[aluno] {arch}: {count_parameters(model):,} parâmetros | treino {len(train_ds)} "
        f"({int(train_ds.is_human.sum())} humanos) | {device}"
    )
    criterion = DistillationLoss(kd_temperature, human_lambda)
    model, history = fit_student(
        model, train_loader, val_loader, criterion, epochs=epochs, lr=lr,
        backbone_lr_mult=backbone_lr_mult, device=device, num_classes=n_cls,
    )

    meta = {
        "method": "trilha_c_distill",
        "budget": budget,
        "seed": seed,
        "best_val_f1": max(history["val_f1"]),
        "best_epoch": history["best_epoch"][0],
        "hparams": {
            "epochs": epochs, "epoch_size": epoch_size, "batch_size": batch_size, "lr": lr,
            "backbone_lr_mult": backbone_lr_mult, "kd_temperature": kd_temperature,
            "human_lambda": human_lambda, "teacher_temperature": teacher_temperature,
            "conf_threshold": conf_threshold, "min_per_class": min_per_class,
        },
    }
    if checkpoint:
        save_student_checkpoint(model, checkpoint, arch=arch, num_classes=n_cls, extra={"meta": meta})
        with open(os.path.splitext(checkpoint)[0] + "_history.json", "w") as f:
            json.dump({"meta": meta, "history": history}, f, indent=2)
        print(f"[aluno] checkpoint salvo em {checkpoint}")
    return model, {"meta": meta, "history": history}


def main() -> None:
    p = argparse.ArgumentParser(description="Treina o aluno destilado (Trilha C).")
    p.add_argument("--data-dir", default="data/pa3_dados")
    p.add_argument("--budget", type=int, default=5, choices=[1, 5, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--epoch-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--human-lambda", type=float, default=1.0)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--min-per-class", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    ckpt = a.checkpoint or f"checkpoints/student_K{a.budget}_s{a.seed}.pth"
    train_distilled_student(
        a.data_dir, a.budget, a.seed, ckpt, a.arch, a.epochs, a.epoch_size, a.batch_size, a.lr,
        a.backbone_lr_mult, a.kd_temperature, a.human_lambda, a.teacher_temperature,
        a.conf_threshold, a.min_per_class, a.num_workers, a.device,
    )


if __name__ == "__main__":
    main()
