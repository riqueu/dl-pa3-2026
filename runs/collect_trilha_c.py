"""Coleta os números da Trilha C para a curva de orçamento (Parte 2).

Avalia no `test_public` (só reporte) cada checkpoint `checkpoints/student_K*_s*.pth`
e o professor (zero-shot e melhorado por K). Escreve
`outputs/curva_trilha_c.csv` (colunas do edital) e
`outputs/per_class_f1_trilha_c.json`. O piso entra depois, no
`curva_orcamento.csv` final.

Uso: python runs/collect_trilha_c.py
"""

import glob
import json
import os
import re
import sys

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ENA24Dataset, get_transforms, load_class_mapping  # noqa: E402
from src.metrics import compute_per_class_f1  # noqa: E402
from src.models import evaluate_model, load_student_checkpoint  # noqa: E402

DATA = "data/pa3_dados"


def main() -> None:
    names, _, _ = load_class_mapping(DATA)
    test_ds = ENA24Dataset(
        os.path.join(DATA, "images", "test_public"),
        csv_path=os.path.join(DATA, "labels", "test_public.csv"),
        transform=get_transforms(224),
    )
    loader = DataLoader(test_ds, batch_size=128, num_workers=4)

    rows, per_class = [], {}
    for ckpt in sorted(glob.glob("checkpoints/student_K*_s*.pth")):
        k, seed = map(int, re.search(r"K(\d+)_s(\d+)", ckpt).groups())
        r = evaluate_model(load_student_checkpoint(ckpt), loader)
        rows.append({"metodo": "trilha_c_aluno", "orcamento": k, "seed": seed,
                     "f1_macro": r["f1_macro"], "acuracia": r["accuracy"]})
        per_class[f"aluno_K{k}_s{seed}"] = compute_per_class_f1(r["y_true"], r["y_pred"], names)
        print(f"aluno K={k} seed={seed}: F1 {r['f1_macro']:.4f} acc {r['accuracy']:.4f}")

    for path in sorted(glob.glob("outputs/teacher_diagnosis/test_public_K*.json")):
        k = int(re.search(r"K(\d+)", path).group(1))
        rep = json.load(open(path))
        imp = rep[f"improved_K{k}"]
        rows.append({"metodo": "professor_bioclip_melhorado", "orcamento": k, "seed": 0,
                     "f1_macro": imp["f1_macro"], "acuracia": imp["accuracy"]})
        per_class[f"professor_K{k}"] = imp["per_class_f1"]
        zs = rep["zero_shot_names"]
        per_class["professor_zero_shot"] = zs["per_class_f1"]
    if per_class.get("professor_zero_shot"):
        rows.append({"metodo": "professor_bioclip_zero_shot", "orcamento": 0, "seed": 0,
                     "f1_macro": zs["f1_macro"], "acuracia": zs["accuracy"]})

    df = pd.DataFrame(rows).sort_values(["metodo", "orcamento", "seed"])
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/curva_trilha_c.csv", index=False)
    with open("outputs/per_class_f1_trilha_c.json", "w") as f:
        json.dump(per_class, f, indent=2)
    print(df.to_string(index=False))
    k5 = df[(df.metodo == "trilha_c_aluno") & (df.orcamento == 5)]
    if len(k5) > 1:
        print(f"\naluno K=5: F1 {k5.f1_macro.mean():.4f} ± {k5.f1_macro.std(ddof=1):.4f} ({len(k5)} seeds)")


if __name__ == "__main__":
    main()
