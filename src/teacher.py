"""Professor da Trilha C: BioCLIP zero-shot e cache de embeddings (PA3).

Responsável: Membro 2 (Isaias)
Branch: feature/teacher-distillation

Regras do edital respeitadas aqui:
  * o professor só olha `train` e `test_public` (diagnóstico); o `val` é recusado;
  * o professor nunca entra no `predict.py` (só gera o cache `.npz`).

Uso:
    python -m src.teacher --data-dir data/pa3_dados --splits train test_public
"""

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import (
    is_infrared_image,
    list_image_ids_in_dir,
    load_class_mapping,
    _resolve_image_path,
)

BIOCLIP_ID = "hf-hub:imageomics/bioclip"
CACHE_DIR = os.path.join("outputs", "teacher_cache")
ALLOWED_SPLITS = ("train", "test_public")

# Nome comum e nome científico de cada classe (ordem não importa: indexado por label).
SCIENTIFIC_NAMES: Dict[str, str] = {
    "american_black_bear": "Ursus americanus",
    "american_crow": "Corvus brachyrhynchos",
    "bird": "Aves",
    "bobcat": "Lynx rufus",
    "chicken": "Gallus gallus domesticus",
    "coyote": "Canis latrans",
    "dog": "Canis familiaris",
    "domestic_cat": "Felis catus",
    "eastern_chipmunk": "Tamias striatus",
    "eastern_cottontail": "Sylvilagus floridanus",
    "eastern_fox_squirrel": "Sciurus niger",
    "eastern_gray_squirrel": "Sciurus carolinensis",
    "grey_fox": "Urocyon cinereoargenteus",
    "horse": "Equus caballus",
    "northern_raccoon": "Procyon lotor",
    "red_fox": "Vulpes vulpes",
    "striped_skunk": "Mephitis mephitis",
    "virginia_opossum": "Didelphis virginiana",
    "white_tailed_deer": "Odocoileus virginianus",
    "wild_turkey": "Meleagris gallopavo",
    "woodchuck": "Marmota monax",
}

RICH_TEMPLATES: Sequence[str] = (
    "a photo of {common}.",
    "a camera trap photo of {common}.",
    "a night infrared camera trap photo of {common}.",
    "a photo of {scientific}.",
    "a camera trap photo of {common}, {scientific}.",
)


def common_name(label: str) -> str:
    return label.replace("_", " ")


def build_prompts(class_names: Sequence[str], prompt_set: str = "names") -> List[List[str]]:
    """Prompts por classe.

    * `names`: só o nome da classe (`a photo of american black bear.`). É o
      zero-shot obrigatório do edital ("nomes das classes como prompt").
    * `rich`: ensemble de templates com nome comum + científico e contexto de
      câmera trap / infravermelho.
    """
    prompts: List[List[str]] = []
    for label in class_names:
        common = common_name(label)
        if prompt_set == "names":
            prompts.append([f"a photo of {common}."])
        elif prompt_set == "rich":
            sci = SCIENTIFIC_NAMES.get(label, common)
            prompts.append([t.format(common=common, scientific=sci) for t in RICH_TEMPLATES])
        else:
            raise ValueError(f"prompt_set desconhecido: {prompt_set}")
    return prompts


def load_bioclip(device: str = "cpu"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP_ID)
    tokenizer = open_clip.get_tokenizer(BIOCLIP_ID)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_text(model, tokenizer, prompts: List[List[str]], device: str = "cpu") -> np.ndarray:
    """Embedding de texto por classe (média dos templates, L2-normalizado). `(C, D)`."""
    out = []
    for class_prompts in prompts:
        tokens = tokenizer(class_prompts).to(device)
        emb = model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean = emb.mean(dim=0)
        out.append((mean / mean.norm()).cpu().numpy())
    return np.stack(out).astype(np.float32)


class _ImageFolderSplit(Dataset):
    """Lê imagens de uma pasta e aplica o preprocess do BioCLIP."""

    def __init__(self, image_dir: str, image_ids: Sequence[str], preprocess):
        self.image_dir = image_dir
        self.image_ids = list(image_ids)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        path = _resolve_image_path(self.image_dir, self.image_ids[idx])
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            ir = is_infrared_image(rgb)
            return self.preprocess(rgb), ir


@torch.no_grad()
def encode_images(
    model,
    preprocess,
    image_dir: str,
    image_ids: Sequence[str],
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 4,
):
    loader = DataLoader(
        _ImageFolderSplit(image_dir, image_ids, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    embs, irs = [], []
    for images, ir in tqdm(loader, desc=f"BioCLIP {os.path.basename(image_dir)}"):
        feat = model.encode_image(images.to(device)).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embs.append(feat.cpu().numpy())
        irs.append(np.asarray(ir, dtype=bool))
    return np.concatenate(embs).astype(np.float32), np.concatenate(irs)


def logits_from_embeddings(
    embeddings: np.ndarray, text_embeddings: np.ndarray, logit_scale: float
) -> np.ndarray:
    return (logit_scale * embeddings @ text_embeddings.T).astype(np.float32)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def cache_path(split: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{split}_teacher.npz")


def text_cache_path(cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, "text_embeddings.npz")


def load_teacher_cache(split: str, cache_dir: str = CACHE_DIR) -> Dict[str, np.ndarray]:
    """Carrega o cache do professor (contrato 2.3 do plano)."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"O professor não tem cache para '{split}' (só {ALLOWED_SPLITS}).")
    with np.load(cache_path(split, cache_dir), allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    out["image_ids"] = out["image_ids"].astype(str)
    return out


def load_text_cache(cache_dir: str = CACHE_DIR) -> Dict[str, np.ndarray]:
    with np.load(text_cache_path(cache_dir), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def build_teacher_cache(
    data_dir: str,
    splits: Sequence[str] = ALLOWED_SPLITS,
    cache_dir: str = CACHE_DIR,
    device: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    for split in splits:
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Split '{split}' proibido para o professor. Permitidos: {ALLOWED_SPLITS} "
                "(o val nunca é visto pelo professor)."
            )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cache_dir, exist_ok=True)
    class_names, _, _ = load_class_mapping(data_dir)

    model, preprocess, tokenizer = load_bioclip(device)
    logit_scale = float(model.logit_scale.exp().item())

    text = {
        f"text_{ps}": encode_text(model, tokenizer, build_prompts(class_names, ps), device)
        for ps in ("names", "rich")
    }
    np.savez(text_cache_path(cache_dir), logit_scale=np.float32(logit_scale), **text)

    for split in splits:
        out_path = cache_path(split, cache_dir)
        if os.path.isfile(out_path) and not overwrite:
            print(f"[teacher] cache já existe: {out_path} (use --overwrite)")
            continue
        image_dir = os.path.join(data_dir, "images", split)
        image_ids = list_image_ids_in_dir(image_dir)
        emb, is_ir = encode_images(
            model, preprocess, image_dir, image_ids, device, batch_size, num_workers
        )
        logits = logits_from_embeddings(emb, text["text_names"], logit_scale)
        probs = softmax(logits)
        np.savez(
            out_path,
            image_ids=np.asarray(image_ids),
            embeddings=emb,
            logits=logits,
            probs=probs,
            preds=probs.argmax(axis=1).astype(np.int64),
            is_infrared=is_ir.astype(bool),
        )
        print(f"[teacher] {split}: {len(image_ids)} imagens -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o cache do professor BioCLIP.")
    parser.add_argument("--data-dir", default="data/pa3_dados")
    parser.add_argument("--splits", nargs="+", default=list(ALLOWED_SPLITS))
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_teacher_cache(
        args.data_dir,
        args.splits,
        args.cache_dir,
        args.device,
        args.batch_size,
        args.num_workers,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
