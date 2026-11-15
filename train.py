"""Pipeline unificado de treinamento do PA3 (Piso, Teacher Cache e Trilha C).

Exemplos:
    python train.py --mode teacher_cache
    python train.py --mode scratch_cnn --budget 5 --seed 42 --epochs 35
    python train.py --mode piso --budget 5 --seed 42 --epochs 35
    python train.py --mode distill --budget 5 --seed 42 --epochs 15 --freeze none
"""

import argparse
import json
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.dataset import ENA24Dataset, get_transforms, load_class_mapping
from src.distillation import FREEZE_PREFIXES, set_seed, train_distilled_student
from src.models import (
    build_student_backbone,
    count_parameters,
    detect_overfitting_epoch,
    evaluate_model,
    fit_supervised_baseline,
    save_student_checkpoint,
)
from src.teacher import CACHE_DIR, build_teacher_cache


def resolve_default_data_dir() -> str:
    """Detecta automaticamente `data` ou `data/pa3_dados`."""
    return "data" if os.path.isdir("data/images") else "data/pa3_dados"


def train_supervised_piso(
    data_dir: str,
    mode: str,
    budget: int,
    seed: int,
    checkpoint: Optional[str] = None,
    arch: str = "resnet18",
    epochs: int = 35,
    batch_size: int = 32,
    lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    backbone_lr_mult: float = 0.25,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Dict:
    """Treina Piso supervisionado (`scratch_cnn` ou `piso` ResNet-18) usando apenas `rotulos_{budget}.csv`."""
    set_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pin = dev.startswith("cuda")

    eff_arch = "small_cnn" if mode == "scratch_cnn" else arch
    eff_lr = lr if lr is not None else (1e-3 if eff_arch == "small_cnn" else 3e-4)
    if checkpoint is None:
        prefix = "piso_small_cnn" if eff_arch == "small_cnn" else "piso"
        checkpoint = f"checkpoints/{prefix}_K{budget}_s{seed}.pth"

    class_names, _, _ = load_class_mapping(data_dir)
    n_cls = len(class_names)

    train_ds = ENA24Dataset(
        os.path.join(data_dir, "images", "train"),
        csv_path=os.path.join(data_dir, "labels", f"rotulos_{budget}.csv"),
        transform=get_transforms(224, is_train=True),
    )
    val_ds = ENA24Dataset(
        os.path.join(data_dir, "images", "val"),
        csv_path=os.path.join(data_dir, "labels", "val.csv"),
        transform=get_transforms(224, is_train=False),
    )

    eff_bs = min(batch_size, len(train_ds))
    drop_last = len(train_ds) > eff_bs and (len(train_ds) % eff_bs == 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=eff_bs,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    model = build_student_backbone(eff_arch, num_classes=n_cls, pretrained=(eff_arch != "small_cnn"))
    print(
        f"[{mode}] arch={eff_arch} ({count_parameters(model):,} params) | K={budget} ({len(train_ds)} imgs) | "
        f"seed={seed} | epochs={epochs} | lr={eff_lr} | device={dev}"
    )

    model, history = fit_supervised_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=eff_lr,
        weight_decay=weight_decay,
        device=dev,
        class_weights=None,
        backbone_lr_mult=backbone_lr_mult,
    )

    val_res = evaluate_model(model, val_loader, device=dev, num_classes=n_cls)
    overfit_epoch = detect_overfitting_epoch(history, patience=2)
    best_epoch = int(max(range(len(history["val_f1"])), key=lambda i: (history["val_f1"][i], -history["val_loss"][i])) + 1)

    meta = {
        "method": f"piso_{eff_arch}",
        "budget": budget,
        "seed": seed,
        "arch": eff_arch,
        "n_train": len(train_ds),
        "best_val_f1": float(val_res["f1_macro"]),
        "best_val_acc": float(val_res["accuracy"]),
        "best_epoch": best_epoch,
        "overfit_epoch": int(overfit_epoch),
        "hparams": {
            "epochs": epochs,
            "batch_size": eff_bs,
            "lr": eff_lr,
            "weight_decay": weight_decay,
            "backbone_lr_mult": backbone_lr_mult,
        },
    }

    save_student_checkpoint(
        model,
        checkpoint,
        arch=eff_arch,
        num_classes=n_cls,
        extra={"meta": meta, "class_names": class_names},
    )
    hist_path = os.path.splitext(checkpoint)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump({"meta": meta, "history": history}, f, indent=2)

    print(
        f"[{mode}] val_f1={meta['best_val_f1']:.4f} | val_acc={meta['best_val_acc']:.4f} | "
        f"best_epoch={best_epoch} | overfit_epoch={overfit_epoch} | salvo em {checkpoint}"
    )
    return {"meta": meta, "history": history, "checkpoint": checkpoint}


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento unificado PA3 (Piso + Trilha C).")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["teacher_cache", "scratch_cnn", "piso", "distill"],
        help="Etapa/método de treinamento.",
    )
    parser.add_argument("--data-dir", default=resolve_default_data_dir())
    parser.add_argument("--budget", type=int, default=5, choices=[1, 5, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")

    # Parâmetros específicos da Trilha C (destilação)
    parser.add_argument("--epoch-size", type=int, default=3072)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--human-lambda", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--freeze", default="none", choices=list(FREEZE_PREFIXES))
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    args = parser.parse_args()

    if args.mode == "teacher_cache":
        build_teacher_cache(
            data_dir=args.data_dir,
            splits=("train", "test_public"),
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size or 64,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
        return

    if args.mode in ("scratch_cnn", "piso"):
        train_supervised_piso(
            data_dir=args.data_dir,
            mode=args.mode,
            budget=args.budget,
            seed=args.seed,
            checkpoint=args.checkpoint,
            arch=args.arch,
            epochs=args.epochs if args.epochs is not None else 35,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            lr=args.lr,
            weight_decay=args.weight_decay,
            backbone_lr_mult=args.backbone_lr_mult if args.backbone_lr_mult is not None else 0.25,
            num_workers=args.num_workers,
            device=args.device,
        )
        return

    train_cache = os.path.join(args.cache_dir, "train_teacher.npz")
    if not os.path.isfile(train_cache):
        build_teacher_cache(
            data_dir=args.data_dir,
            splits=("train", "test_public"),
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=64,
            num_workers=args.num_workers,
        )

    ckpt = args.checkpoint or f"checkpoints/student_K{args.budget}_s{args.seed}.pth"
    train_distilled_student(
        data_dir=args.data_dir,
        budget=args.budget,
        seed=args.seed,
        checkpoint=ckpt,
        arch=args.arch,
        epochs=args.epochs if args.epochs is not None else 15,
        epoch_size=args.epoch_size,
        batch_size=args.batch_size if args.batch_size is not None else 64,
        lr=args.lr if args.lr is not None else 1e-3,
        backbone_lr_mult=args.backbone_lr_mult if args.backbone_lr_mult is not None else 0.1,
        kd_temperature=args.kd_temperature,
        human_lambda=args.human_lambda,
        teacher_temperature=args.teacher_temperature,
        conf_threshold=args.conf_threshold,
        min_per_class=args.min_per_class,
        freeze=args.freeze,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
