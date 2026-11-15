"""Script oficial de inferência offline em CPU para entrega do PA3.

Uso (PA3):
    python predict.py --imagens PASTA --saida preds.csv
    python predict.py -imagens PASTA -saida preds.csv
"""

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import DEFAULT_CLASSES, ENA24Dataset, get_transforms, load_class_mapping
from src.models import MAX_ALLOWED_PARAMS, count_parameters, load_student_checkpoint


def _normalize_cli_args(argv: List[str]) -> List[str]:
    """Aceita tanto `--flag` quanto `-flag` / `–flag` (como no PDF do PA3)."""
    single_to_double = {
        "-imagens": "--imagens",
        "–imagens": "--imagens",
        "-saida": "--saida",
        "–saida": "--saida",
        "-checkpoint": "--checkpoint",
        "–checkpoint": "--checkpoint",
        "-data-dir": "--data-dir",
        "-data_dir": "--data-dir",
        "-classes-csv": "--classes-csv",
        "-classes_csv": "--classes-csv",
        "-batch-size": "--batch-size",
        "-batch_size": "--batch-size",
    }
    return [single_to_double.get(arg, arg) for arg in argv]


def _resolve_idx_to_label(
    checkpoint_path: str,
    classes_csv: str | None = None,
    data_dir: str | None = None,
) -> Dict[int, str]:
    """Resolve mapeamento índice -> nome da espécie offline com fallback seguro."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("class_names"), list):
        names = [str(c).strip() for c in payload["class_names"]]
        if names:
            return {idx: name for idx, name in enumerate(names)}

    if classes_csv and os.path.isfile(classes_csv):
        df = pd.read_csv(classes_csv).sort_values("idx").reset_index(drop=True)
        return {int(row["idx"]): str(row["label"]).strip() for _, row in df.iterrows()}

    candidate_dirs = [d for d in [data_dir, "data", "data/pa3_dados"] if d]
    for d in candidate_dirs:
        if os.path.isfile(os.path.join(d, "classes.csv")):
            _, _, idx_to_label = load_class_mapping(d)
            return idx_to_label

    return {idx: name for idx, name in enumerate(DEFAULT_CLASSES)}


@torch.no_grad()
def predict_directory(
    imagens: str,
    saida: str,
    checkpoint: str = "checkpoints/best_student.pth",
    classes_csv: str | None = None,
    data_dir: str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Executa inferência estritamente em CPU e salva `image_id,label`."""
    if not os.path.isdir(imagens):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {imagens}")

    model = load_student_checkpoint(checkpoint, device="cpu")
    model.eval()

    n_params = count_parameters(model)
    assert n_params <= MAX_ALLOWED_PARAMS, (
        f"Modelo excede o limite de {MAX_ALLOWED_PARAMS:,} parâmetros ({n_params:,})."
    )
    print(
        f"[predict] checkpoint={checkpoint} | parametros={n_params:,} "
        f"(limite<={MAX_ALLOWED_PARAMS:,} OK) | device=cpu"
    )

    idx_to_label = _resolve_idx_to_label(checkpoint, classes_csv=classes_csv, data_dir=data_dir)
    dataset = ENA24Dataset(
        image_dir=imagens,
        csv_path=None,
        transform=get_transforms(img_size=224, is_train=False),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    rows = []
    for images, _, image_ids in loader:
        logits = model(images.to("cpu"))
        preds = logits.argmax(dim=1).cpu().tolist()
        for img_id, pred_idx in zip(image_ids, preds):
            rows.append(
                {
                    "image_id": str(img_id),
                    "label": idx_to_label.get(int(pred_idx), DEFAULT_CLASSES[int(pred_idx)]),
                }
            )

    df_out = pd.DataFrame(rows, columns=["image_id", "label"])
    parent = os.path.dirname(saida)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df_out.to_csv(saida, index=False)
    print(f"[predict] {len(df_out)} predições salvas em {saida}")
    return df_out


def main(argv: List[str] | None = None) -> None:
    raw_args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Inferência offline em CPU — PA3.")
    parser.add_argument("--imagens", required=True, help="Pasta contendo as imagens.")
    parser.add_argument("--saida", required=True, help="Caminho do CSV de saída.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_student.pth",
        help="Checkpoint do modelo aluno treinado.",
    )
    parser.add_argument("--classes-csv", default=None, help="Opcional: caminho para classes.csv.")
    parser.add_argument("--data-dir", default=None, help="Opcional: pasta raiz dos dados.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(_normalize_cli_args(raw_args))

    predict_directory(
        imagens=args.imagens,
        saida=args.saida,
        checkpoint=args.checkpoint,
        classes_csv=args.classes_csv,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
