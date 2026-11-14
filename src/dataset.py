"""Módulo de carregamento e pré-processamento do dataset ENA24-detection (PA3).

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-baseline
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_CLASSES: List[str] = [
    "american_black_bear",
    "american_crow",
    "bird",
    "bobcat",
    "chicken",
    "coyote",
    "dog",
    "domestic_cat",
    "eastern_chipmunk",
    "eastern_cottontail",
    "eastern_fox_squirrel",
    "eastern_gray_squirrel",
    "grey_fox",
    "horse",
    "northern_raccoon",
    "red_fox",
    "striped_skunk",
    "virginia_opossum",
    "white_tailed_deer",
    "wild_turkey",
    "woodchuck",
]

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_class_mapping(
    data_dir: str = "data",
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Carrega o mapeamento oficial das 21 classes de `classes.csv` (com fallback local).

    Args:
        data_dir: Diretório raiz contendo `classes.csv`.

    Returns:
        `(class_names, label_to_idx, idx_to_label)` ordenados por `idx`.
    """
    classes_path = os.path.join(data_dir, "classes.csv")
    if os.path.isfile(classes_path):
        df = pd.read_csv(classes_path).sort_values("idx").reset_index(drop=True)
        class_names = [str(lbl).strip() for lbl in df["label"].tolist()]
        label_to_idx = {
            str(row["label"]).strip(): int(row["idx"]) for _, row in df.iterrows()
        }
        idx_to_label = {
            int(row["idx"]): str(row["label"]).strip() for _, row in df.iterrows()
        }
        return class_names, label_to_idx, idx_to_label

    class_names = list(DEFAULT_CLASSES)
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_label = {idx: name for idx, name in enumerate(class_names)}
    return class_names, label_to_idx, idx_to_label


def load_split_df(
    csv_path: str,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Lê um CSV de rótulos (`image_id,label`) e adiciona `label_idx`.

    Args:
        csv_path: Caminho do CSV (`rotulos_1/5/20.csv`, `val.csv` ou `test_public.csv`).
        label_to_idx: Mapeamento classe -> índice. Se None, usa fallback padrão.

    Returns:
        DataFrame com colunas `image_id`, `label` e `label_idx`.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Arquivo CSV de rótulos não encontrado: {csv_path}")

    if label_to_idx is None:
        _, label_to_idx, _ = load_class_mapping(
            os.path.dirname(os.path.dirname(csv_path))
        )

    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV {csv_path} deve conter colunas 'image_id' e 'label'.")

    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    unknown = set(df["label"].unique()) - set(label_to_idx.keys())
    if unknown:
        raise ValueError(f"Classes desconhecidas em {csv_path}: {sorted(unknown)}")

    df["label_idx"] = df["label"].map(label_to_idx).astype(int)
    return df


def is_infrared_image(
    img: Union[np.ndarray, Image.Image],
    threshold: float = 5.0,
) -> bool:
    """Heurística para identificar fotos noturnas em infravermelho (monocromáticas).

    Em câmeras trap, fotos noturnas IR apresentam diferença média muito próxima
    de zero entre os canais R, G e B.
    """
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    else:
        raw = np.asarray(img)
        if raw.ndim == 2:
            return True
        arr = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.floating) and float(arr.max()) <= 1.0:
            arr = arr * 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mean_diff = float(np.mean((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0))
    return mean_diff < threshold


def get_transforms(img_size: int = 224, is_train: bool = False) -> transforms.Compose:
    """Pipeline de transformações para o ENA24.

    Na avaliação/inferência usamos `Resize((img_size, img_size))` em vez de
    `CenterCrop`, evitando recortar animais localizados próximos às bordas.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def list_image_ids_in_dir(image_dir: str) -> List[str]:
    """Lista ordenadamente todos os `image_id` (sem extensão) de uma pasta."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")

    ids: List[str] = []
    for entry in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, entry)
        if os.path.isfile(path) and entry.lower().endswith(VALID_IMAGE_EXTENSIONS):
            ids.append(Path(entry).stem)
    return ids


def get_unlabeled_pool_ids(
    data_dir: str = "data",
    exclude_labeled_csv: Optional[str] = None,
) -> List[str]:
    """Lista os `image_id` do pool `train`, opcionalmente removendo os rotulados."""
    train_dir = os.path.join(data_dir, "images", "train")
    all_ids = list_image_ids_in_dir(train_dir)
    if exclude_labeled_csv is None:
        return all_ids

    labeled_df = pd.read_csv(exclude_labeled_csv)
    labeled_set = set(labeled_df["image_id"].astype(str).str.strip().tolist())
    return [img_id for img_id in all_ids if img_id not in labeled_set]


def _resolve_image_path(image_dir: str, image_id: str) -> str:
    """Resolve o caminho completo de uma imagem dado seu `image_id`."""
    direct = os.path.join(image_dir, image_id)
    if os.path.isfile(direct):
        return direct

    for ext in VALID_IMAGE_EXTENSIONS:
        for candidate_ext in (ext, ext.upper()):
            candidate = os.path.join(image_dir, f"{image_id}{candidate_ext}")
            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(
        f"Imagem não encontrada para id '{image_id}' em '{image_dir}'."
    )


class ENA24Dataset(Dataset):
    """Dataset unificado do ENA24 para splits rotulados, pool `train` e inferência."""

    def __init__(
        self,
        image_dir: str,
        csv_path: Optional[str] = None,
        image_ids: Optional[Sequence[str]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
        soft_labels: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        return_dict: bool = False,
    ):
        self.image_dir = image_dir
        self.transform = (
            transform if transform is not None else get_transforms(224, is_train=False)
        )
        self.return_dict = return_dict

        if csv_path is not None:
            df = load_split_df(csv_path, label_to_idx=label_to_idx)
            self.image_ids = df["image_id"].tolist()
            self.labels = df["label_idx"].tolist()
        else:
            self.image_ids = (
                list(image_ids)
                if image_ids is not None
                else list_image_ids_in_dir(image_dir)
            )
            self.labels = [-1] * len(self.image_ids)

        if soft_labels is not None:
            soft_arr = np.asarray(soft_labels, dtype=np.float32)
            if len(soft_arr) != len(self.image_ids):
                raise ValueError(
                    f"soft_labels tem tamanho {len(soft_arr)}, esperado {len(self.image_ids)}."
                )
            self.soft_labels: Optional[np.ndarray] = soft_arr
        else:
            self.soft_labels = None

        if sample_weights is not None:
            weights_arr = np.asarray(sample_weights, dtype=np.float32)
            if len(weights_arr) != len(self.image_ids):
                raise ValueError(
                    f"sample_weights tem tamanho {len(weights_arr)}, esperado {len(self.image_ids)}."
                )
            self.sample_weights: Optional[np.ndarray] = weights_arr
        else:
            self.sample_weights = None

        self.image_paths: List[str] = [
            _resolve_image_path(self.image_dir, img_id) for img_id in self.image_ids
        ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_image_path(self, idx: int) -> str:
        """Retorna o caminho em disco da amostra `idx`."""
        return self.image_paths[idx]

    def load_raw_rgb(self, idx: int) -> np.ndarray:
        """Carrega a imagem original como array RGB uint8 `(H, W, 3)`."""
        with Image.open(self.image_paths[idx]) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            tensor = self.transform(img_rgb)

        label = int(self.labels[idx])
        image_id = self.image_ids[idx]

        if self.return_dict:
            item = {
                "image": tensor,
                "label": label,
                "image_id": image_id,
                "path": img_path,
            }
            if self.soft_labels is not None:
                item["soft_label"] = torch.from_numpy(self.soft_labels[idx])
            if self.sample_weights is not None:
                item["weight"] = float(self.sample_weights[idx])
            return item

        return tensor, label, image_id
