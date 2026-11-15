"""Pipeline unificado de avaliação e consolidação de métricas do PA3.

Modos:
    checkpoint  -- avalia um checkpoint em `val` ou `test_public`.
    teacher     -- gera/atualiza diagnósticos do professor em `test_public` (K=1, 5, 20).
    collect_all -- consolida Piso + Trilha C + Professor em `curva_orcamento.csv`,
                   seleciona `checkpoints/best_student.pth` via `val` (sem leakage!)
                   e plota figuras da Parte 2.
"""

import argparse
import glob
import json
import os
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import ENA24Dataset, get_transforms, load_class_mapping
from src.metrics import compute_confusion_matrix, compute_per_class_f1
from src.models import evaluate_model, load_student_checkpoint
from src.pseudo_labels import diagnose_on_test_public
from src.saliency import GradCAM, occlusion_sensitivity, select_failure_and_success_cases
from src.teacher import load_teacher_cache
from src.visualization import (
    plot_budget_curve,
    plot_confusion_matrix,
    plot_per_class_gains,
    plot_saliency_gallery,
    plot_training_curves,
)


def resolve_default_data_dir() -> str:
    return "data" if os.path.isdir("data/images") else "data/pa3_dados"


def _make_loader(data_dir: str, split: str, batch_size: int = 128, num_workers: int = 4) -> Tuple[ENA24Dataset, DataLoader]:
    ds = ENA24Dataset(
        os.path.join(data_dir, "images", split),
        csv_path=os.path.join(data_dir, "labels", f"{split}.csv"),
        transform=get_transforms(224, is_train=False),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds, loader


def eval_single_checkpoint(
    checkpoint: str,
    data_dir: str,
    split: str = "val",
    device: str = "cpu",
    out_json: str | None = None,
) -> Dict:
    class_names, _, _ = load_class_mapping(data_dir)
    _, loader = _make_loader(data_dir, split)
    model = load_student_checkpoint(checkpoint, device=device)
    res = evaluate_model(model, loader, device=device, num_classes=len(class_names))
    per_cls = compute_per_class_f1(res["y_true"], res["y_pred"], class_names)
    cm = compute_confusion_matrix(res["y_true"], res["y_pred"], num_classes=len(class_names))
    report = {
        "checkpoint": checkpoint,
        "split": split,
        "f1_macro": float(res["f1_macro"]),
        "accuracy": float(res["accuracy"]),
        "per_class_f1": per_cls,
        "confusion_matrix": cm.tolist(),
    }
    print(f"[evaluate] {checkpoint} ({split}): F1={res['f1_macro']:.4f} | acc={res['accuracy']:.4f}")
    if out_json:
        parent = os.path.dirname(out_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
    return report


def collect_all_results(
    data_dir: str,
    out_csv: str = "curva_orcamento.csv",
    best_budget: int = 5,
    best_checkpoint_out: str = "checkpoints/best_student.pth",
    device: str | None = None,
) -> pd.DataFrame:
    """Consolida Piso (`SmallCNN` + `ResNet-18`), Trilha C e Professor em `curva_orcamento.csv`."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    names, _, _ = load_class_mapping(data_dir)
    n_cls = len(names)
    _, val_loader = _make_loader(data_dir, "val")
    _, test_loader = _make_loader(data_dir, "test_public")

    rows: List[Dict] = []
    per_class_all: Dict[str, Dict[str, float]] = {}
    val_summary: Dict[str, Dict] = {}

    patterns = [
        ("piso_small_cnn", "checkpoints/piso_small_cnn_K*_s*.pth", "piso_small_cnn"),
        ("piso_resnet18", "checkpoints/piso_K*_s*.pth", "piso_resnet18"),
        ("trilha_c_aluno", "checkpoints/student_K*_s*.pth", "aluno"),
    ]

    for metodo, pattern, pc_prefix in patterns:
        for ckpt in sorted(glob.glob(pattern)):
            match = re.search(r"K(\d+)_s(\d+)", ckpt)
            if not match:
                continue
            k, seed = map(int, match.groups())
            model = load_student_checkpoint(ckpt, device=dev)
            val_res = evaluate_model(model, val_loader, device=dev, num_classes=n_cls)
            test_res = evaluate_model(model, test_loader, device=dev, num_classes=n_cls)

            rows.append(
                {
                    "metodo": metodo,
                    "orcamento": k,
                    "seed": seed,
                    "f1_macro": round(float(test_res["f1_macro"]), 6),
                    "acuracia": round(float(test_res["accuracy"]), 6),
                }
            )
            key = f"{pc_prefix}_K{k}_s{seed}"
            per_class_all[key] = compute_per_class_f1(test_res["y_true"], test_res["y_pred"], names)
            val_summary[ckpt] = {
                "metodo": metodo,
                "orcamento": k,
                "seed": seed,
                "val_f1": float(val_res["f1_macro"]),
                "val_acc": float(val_res["accuracy"]),
                "test_f1": float(test_res["f1_macro"]),
                "test_acc": float(test_res["accuracy"]),
            }
            print(
                f"{metodo:16s} K={k:2d} s={seed}: "
                f"val_F1={val_res['f1_macro']:.4f} | test_F1={test_res['f1_macro']:.4f} acc={test_res['accuracy']:.4f}"
            )

    zs_data = None
    for path in sorted(glob.glob("outputs/teacher_diagnosis/test_public_K*.json")):
        k = int(re.search(r"K(\d+)", path).group(1))
        with open(path) as f:
            rep = json.load(f)
        imp = rep[f"improved_K{k}"]
        rows.append(
            {
                "metodo": "professor_bioclip_melhorado",
                "orcamento": k,
                "seed": 0,
                "f1_macro": round(float(imp["f1_macro"]), 6),
                "acuracia": round(float(imp["accuracy"]), 6),
            }
        )
        per_class_all[f"professor_K{k}"] = imp["per_class_f1"]
        zs_data = rep["zero_shot_names"]
        per_class_all["professor_zero_shot"] = zs_data["per_class_f1"]

    if zs_data is not None:
        rows.append(
            {
                "metodo": "professor_bioclip_zero_shot",
                "orcamento": 0,
                "seed": 0,
                "f1_macro": round(float(zs_data["f1_macro"]), 6),
                "acuracia": round(float(zs_data["accuracy"]), 6),
            }
        )

    df = pd.DataFrame(rows, columns=["metodo", "orcamento", "seed", "f1_macro", "acuracia"])
    df = df.sort_values(["metodo", "orcamento", "seed"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    os.makedirs("outputs", exist_ok=True)
    df_trilha_c = df[df["metodo"].isin(["trilha_c_aluno", "professor_bioclip_melhorado", "professor_bioclip_zero_shot"])]
    df_trilha_c.to_csv("outputs/curva_trilha_c.csv", index=False)

    with open("outputs/per_class_f1_all.json", "w") as f:
        json.dump(per_class_all, f, indent=2)

    trilha_c_pc = {k: v for k, v in per_class_all.items() if k.startswith(("aluno_", "professor_"))}
    with open("outputs/per_class_f1_trilha_c.json", "w") as f:
        json.dump(trilha_c_pc, f, indent=2)

    with open("outputs/val_and_test_summary.json", "w") as f:
        json.dump(val_summary, f, indent=2)

    # Seleciona checkpoints/best_student.pth estritamente por val_f1 no orçamento padrão
    candidates = [
        (info["val_f1"], ckpt)
        for ckpt, info in val_summary.items()
        if info["metodo"] == "trilha_c_aluno" and info["orcamento"] == best_budget
    ]
    if candidates:
        best_val_f1, best_ckpt = max(candidates, key=lambda x: x[0])
        shutil.copy2(best_ckpt, best_checkpoint_out)
        print(
            f"\n[select] {best_checkpoint_out} <- {best_ckpt} "
            f"(selecionado exclusivamente em val: val_F1={best_val_f1:.4f})"
        )

    # Imprime resumo de K=5 média ± std
    print("\n--- Resumo Parte 2 (test_public) ---")
    for m in ["piso_resnet18", "trilha_c_aluno"]:
        sub = df[(df["metodo"] == m) & (df["orcamento"] == 5)]
        if len(sub) > 0:
            std_val = float(sub["f1_macro"].std(ddof=1)) if len(sub) > 1 else 0.0
            print(f"{m:16s} K=5: F1 = {sub['f1_macro'].mean():.4f} ± {std_val:.4f} (n={len(sub)})")

    # Gera figuras da Parte 2
    os.makedirs("outputs/figures", exist_ok=True)
    plot_budget_curve(
        df,
        ceiling_f1=0.686,
        title="Curva de Orçamento — F1 Macro no test_public",
        save_path="outputs/figures/curva_orcamento.png",
    )

    # Comparação por classe no orçamento K=5 (média das seeds de Piso vs Trilha C vs Professor)
    piso_k5_keys = [k for k in per_class_all if k.startswith("piso_resnet18_K5_")]
    aluno_k5_keys = [k for k in per_class_all if k.startswith("aluno_K5_")]
    if piso_k5_keys and aluno_k5_keys:
        mean_piso_pc = {c: float(sum(per_class_all[k][c] for k in piso_k5_keys) / len(piso_k5_keys)) for c in names}
        mean_aluno_pc = {c: float(sum(per_class_all[k][c] for k in aluno_k5_keys) / len(aluno_k5_keys)) for c in names}
        teacher_pc = per_class_all.get("professor_K5")
        plot_per_class_gains(
            mean_piso_pc,
            mean_aluno_pc,
            per_class_teacher=teacher_pc,
            title="F1 por Classe no Orçamento de 5 Rótulos (Média 3 Seeds)",
            save_path="outputs/figures/part2_ganhos_por_classe_K5.png",
        )

    return df


def generate_all_figures(
    data_dir: str,
    best_checkpoint: str = "checkpoints/best_student.pth",
    device: str | None = None,
) -> Dict:
    """Gera figuras das Partes 0, 1, 2 e 3 + metadados de saliência (`outputs/part3_cases.json`)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/figures", exist_ok=True)
    names, label_to_idx, _ = load_class_mapping(data_dir)
    n_cls = len(names)
    _, test_loader = _make_loader(data_dir, "test_public")
    test_ds = test_loader.dataset

    # Parte 0: Curvas de treino SmallCNN e ResNet-18 Piso (K=5, seed=42)
    for hist_path, title, out_fig in [
        (
            "checkpoints/piso_small_cnn_K5_s42_history.json",
            "Parte 0(a): SmallCNN do zero (K=5, s=42)",
            "outputs/figures/part0_small_cnn_K5.png",
        ),
        (
            "checkpoints/piso_K5_s42_history.json",
            "Parte 0(b): Piso ResNet-18 (K=5, s=42)",
            "outputs/figures/part0_resnet18_K5.png",
        ),
    ]:
        if os.path.isfile(hist_path):
            with open(hist_path) as f:
                data = json.load(f)
            meta = data.get("meta", {})
            plot_training_curves(
                data["history"],
                overfit_epoch=meta.get("overfit_epoch"),
                title=title,
                save_path=out_fig,
            )

    piso_k5_ckpt = "checkpoints/piso_K5_s42.pth"
    if os.path.isfile(piso_k5_ckpt):
        piso_model = load_student_checkpoint(piso_k5_ckpt, device=dev)
        piso_res = evaluate_model(piso_model, test_loader, device=dev, num_classes=n_cls)
        cm_piso = compute_confusion_matrix(piso_res["y_true"], piso_res["y_pred"], num_classes=n_cls, normalize=True)
        plot_confusion_matrix(
            cm_piso,
            names,
            title="Parte 0(b): Matriz de Confusão — Piso ResNet-18 (K=5, test_public)",
            normalize=True,
            save_path="outputs/figures/part0_confusion_resnet18_K5.png",
        )

    # Parte 1: Matriz de confusão do Professor Zero-Shot BioCLIP
    tp_cache = os.path.join("outputs", "teacher_cache", "test_public_teacher.npz")
    if os.path.isfile(tp_cache):
        cache = load_teacher_cache("test_public")
        zs_pred = cache["preds"].astype(int).tolist()
        tp_df = pd.read_csv(os.path.join(data_dir, "labels", "test_public.csv"))
        id_to_true = {str(r["image_id"]): label_to_idx[str(r["label"]).strip()] for _, r in tp_df.iterrows()}
        zs_true = [id_to_true[str(i)] for i in cache["image_ids"]]
        cm_zs = compute_confusion_matrix(zs_true, zs_pred, num_classes=n_cls, normalize=True)
        plot_confusion_matrix(
            cm_zs,
            names,
            title="Parte 1: Matriz de Confusão — Professor Zero-Shot (BioCLIP, test_public)",
            normalize=True,
            save_path="outputs/figures/part1_teacher_zeroshot_confusion.png",
        )

    # Parte 3: Galeria Grad-CAM (3 erros + 5 acertos) e checagem por oclusão
    part3_summary: Dict = {}
    if os.path.isfile(best_checkpoint):
        student = load_student_checkpoint(best_checkpoint, device=dev)
        stu_res = evaluate_model(student, test_loader, device=dev, num_classes=n_cls)
        cm_stu = compute_confusion_matrix(stu_res["y_true"], stu_res["y_pred"], num_classes=n_cls, normalize=True)
        plot_confusion_matrix(
            cm_stu,
            names,
            title="Parte 1/3: Matriz de Confusão — Melhor Aluno K=5 (test_public)",
            normalize=True,
            save_path="outputs/figures/part1_student_K5_confusion.png",
        )

        cases = select_failure_and_success_cases(student, test_ds, names, device=dev, n_errors=3, n_successes=5)
        plot_saliency_gallery(
            cases["errors"],
            title="Parte 3: 3 Erros do Aluno (Grad-CAM — test_public)",
            save_path="outputs/figures/part3_erros_gradcam.png",
        )
        plot_saliency_gallery(
            cases["successes"],
            title="Parte 3: 5 Acertos do Aluno (Grad-CAM — test_public)",
            save_path="outputs/figures/part3_acertos_gradcam.png",
        )

        id_to_ds_idx = {str(test_ds.image_ids[i]): i for i in range(len(test_ds))}
        cam = GradCAM(student)
        try:
            def _summarize(group_name: str, items: List[Dict], check_occlusion: bool) -> List[Dict]:
                out_list = []
                for c in items:
                    rec = {
                        "image_id": c["image_id"],
                        "true_label": c["true_label"],
                        "pred_label": c["pred_label"],
                        "confidence": round(float(c["confidence"]), 4),
                        "is_infrared": bool(c["is_infrared"]),
                        "center_border_ratio": round(float(c["center_border_ratio"]), 4),
                        "note": c["note"],
                    }
                    if check_occlusion and c["image_id"] in id_to_ds_idx:
                        ds_idx = id_to_ds_idx[c["image_id"]]
                        tensor = test_ds[ds_idx][0].unsqueeze(0).to(dev)
                        pred_idx = label_to_idx[c["pred_label"]]
                        hm_cam = cam.generate(tensor, target_class=pred_idx)
                        hm_occ = occlusion_sensitivity(student, tensor, target_class=pred_idx, patch_size=32, stride=16)
                        corr = float(np.corrcoef(hm_cam.ravel(), hm_occ.ravel())[0, 1]) if hm_occ.std() > 1e-6 and hm_cam.std() > 1e-6 else 0.0
                        rec["gradcam_occlusion_corr"] = round(corr, 4)
                    out_list.append(rec)
                return out_list

            part3_summary = {
                "checkpoint": best_checkpoint,
                "errors": _summarize("errors", cases["errors"], check_occlusion=True),
                "successes": _summarize("successes", cases["successes"], check_occlusion=True),
            }
        finally:
            cam.close()

        with open("outputs/part3_cases.json", "w") as f:
            json.dump(part3_summary, f, indent=2)
        print("[figures] Figuras e outputs/part3_cases.json gerados em outputs/figures/.")
    return part3_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação e consolidação PA3.")
    parser.add_argument(
        "--mode",
        default="checkpoint",
        choices=["checkpoint", "teacher", "collect_all", "figures"],
    )
    parser.add_argument("--data-dir", default=resolve_default_data_dir())
    parser.add_argument("--checkpoint", default="checkpoints/best_student.pth")
    parser.add_argument("--split", default="val", choices=["val", "test_public"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--best-budget", type=int, default=5)
    parser.add_argument("--out-csv", default="curva_orcamento.csv")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.mode == "checkpoint":
        eval_single_checkpoint(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            split=args.split,
            device=args.device or "cpu",
            out_json=args.out_json,
        )
    elif args.mode == "teacher":
        for k in args.budgets:
            diagnose_on_test_public(args.data_dir, k)
    elif args.mode == "collect_all":
        collect_all_results(
            data_dir=args.data_dir,
            out_csv=args.out_csv,
            best_budget=args.best_budget,
            device=args.device,
        )
    elif args.mode == "figures":
        generate_all_figures(
            data_dir=args.data_dir,
            best_checkpoint=args.checkpoint,
            device=args.device,
        )


if __name__ == "__main__":
    main()
