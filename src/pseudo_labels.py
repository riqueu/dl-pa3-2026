"""Diagnóstico do professor e tratamento de ruído dos pseudo-rótulos (Trilha C).

Responsável: Membro 2 (Isaias)
Branch: feature/teacher-distillation

Pipeline:
  1. `diagnose_predictions`: F1/acurácia, matriz de ruído 21x21, dia vs. noite (IR)
     e confusões sistemáticas do professor no `test_public` (só medição).
  2. `ImprovedTeacher`: professor melhorado com os K rótulos humanos. Cada classe
     vira um protótipo que mistura o embedding de texto do BioCLIP com a média dos
     embeddings das imagens humanas, com refinamento transdutivo opcional sobre o
     pool. Hiperparâmetros escolhidos por leave-one-out nos próprios rótulos
     humanos (nunca `val`, nunca `test_public`).
  3. `build_pseudo_labels`: rótulos suaves com temperatura, filtro de confiança
     balanceado por classe (quota mínima para classes raras como `horse`) e pesos
     de amostragem inversamente proporcionais à frequência (contrato 2.4).

Uso:
    python -m src.pseudo_labels --budget 5 --diagnose
"""

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.dataset import load_class_mapping, load_split_df
from src.metrics import (
    compute_confusion_matrix,
    compute_macro_f1_and_accuracy,
    compute_per_class_f1,
)
from src.teacher import CACHE_DIR, load_teacher_cache, load_text_cache, softmax

PSEUDO_DIR = os.path.join("outputs", "pseudo_labels")
DIAG_DIR = os.path.join("outputs", "teacher_diagnosis")


def _l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------- #
# 1. Diagnóstico
# --------------------------------------------------------------------------- #
def diagnose_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    is_infrared: Optional[Sequence[bool]] = None,
    top_k_confusions: int = 10,
) -> Dict:
    """Métricas e matriz de ruído de um conjunto de predições do professor."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n_cls = len(class_names)

    out: Dict = dict(compute_macro_f1_and_accuracy(y_true, y_pred, n_cls))
    out["per_class_f1"] = compute_per_class_f1(y_true, y_pred, class_names)
    cm = compute_confusion_matrix(y_true, y_pred, n_cls)
    out["confusion_matrix"] = cm.tolist()
    out["noise_matrix"] = compute_confusion_matrix(y_true, y_pred, n_cls, normalize=True).tolist()

    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
    out["top_confusions"] = [
        {
            "true": class_names[i],
            "pred": class_names[j],
            "count": int(off[i, j]),
            "rate": float(off[i, j] / max(1, cm[i].sum())),
        }
        for i, j in order[:top_k_confusions]
        if off[i, j] > 0
    ]

    if is_infrared is not None:
        ir = np.asarray(is_infrared, dtype=bool)
        for name, mask in (("day", ~ir), ("night_ir", ir)):
            if mask.sum() > 0:
                m = compute_macro_f1_and_accuracy(y_true[mask], y_pred[mask], n_cls)
                out[name] = {"n": int(mask.sum()), **m}
    return out


# --------------------------------------------------------------------------- #
# 2. Professor melhorado (protótipos texto + imagens humanas)
# --------------------------------------------------------------------------- #
@dataclass
class TeacherConfig:
    prompt_set: str = "names"  # "names" | "rich"
    alpha: float = 0.5  # peso do texto no protótipo (1 = zero-shot puro)
    refine_iters: int = 0  # iterações de refinamento transdutivo no pool
    refine_top_frac: float = 0.5  # fração mais confiante por classe usada no refinamento


class ImprovedTeacher:
    """Classificador por protótipos sobre os embeddings do BioCLIP."""

    def __init__(self, text_embeddings: np.ndarray, logit_scale: float, config: TeacherConfig):
        self.text = _l2(np.asarray(text_embeddings, dtype=np.float32))
        self.logit_scale = float(logit_scale)
        self.config = config
        self.prototypes: Optional[np.ndarray] = None

    def _mix(self, image_means: np.ndarray, has_img: np.ndarray) -> np.ndarray:
        a = self.config.alpha
        proto = self.text.copy()
        proto[has_img] = a * self.text[has_img] + (1.0 - a) * image_means[has_img]
        return _l2(proto)

    @staticmethod
    def _class_means(emb: np.ndarray, labels: np.ndarray, n_cls: int, weights=None):
        means = np.zeros((n_cls, emb.shape[1]), dtype=np.float32)
        has = np.zeros(n_cls, dtype=bool)
        for c in range(n_cls):
            m = labels == c
            if m.any():
                w = None if weights is None else weights[m]
                means[c] = _l2(np.average(emb[m], axis=0, weights=w))
                has[c] = True
        return means, has

    def fit(
        self,
        human_emb: np.ndarray,
        human_labels: np.ndarray,
        pool_emb: Optional[np.ndarray] = None,
    ) -> "ImprovedTeacher":
        n_cls = self.text.shape[0]
        h_means, has = self._class_means(human_emb, human_labels, n_cls)
        self.prototypes = self._mix(h_means, has)

        if pool_emb is not None and self.config.refine_iters > 0:
            for _ in range(self.config.refine_iters):
                probs = softmax(self.logits(pool_emb))
                pred, conf = probs.argmax(1), probs.max(1)
                keep = np.zeros(len(pred), dtype=bool)
                for c in range(n_cls):
                    idx = np.where(pred == c)[0]
                    if len(idx) == 0:
                        continue
                    k = max(1, int(np.ceil(self.config.refine_top_frac * len(idx))))
                    keep[idx[np.argsort(-conf[idx])[:k]]] = True
                p_means, p_has = self._class_means(pool_emb[keep], pred[keep], n_cls)
                # Média das imagens: humanos + pool confiante (humanos com peso igual
                # ao bloco do pool, para não serem afogados).
                img_means = h_means.copy()
                both = has & p_has
                img_means[both] = _l2(h_means[both] + p_means[both])
                only_pool = (~has) & p_has
                img_means[only_pool] = p_means[only_pool]
                self.prototypes = self._mix(img_means, has | p_has)
        return self

    def logits(self, emb: np.ndarray) -> np.ndarray:
        protos = self.text if self.prototypes is None else self.prototypes
        return (self.logit_scale * np.asarray(emb, dtype=np.float32) @ protos.T).astype(np.float32)


def select_teacher_config(
    human_emb: np.ndarray,
    human_labels: np.ndarray,
    text_by_set: Dict[str, np.ndarray],
    logit_scale: float,
    pool_emb: Optional[np.ndarray] = None,
    alphas: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
    refine_iters: Sequence[int] = (0, 1, 2),
    num_classes: int = 21,
) -> Tuple[TeacherConfig, List[Dict]]:
    """Escolhe a configuração do professor por leave-one-out nos rótulos humanos.

    Cada humano é classificado por um professor ajustado sem ele. Só usa dados de
    `train`, então não viola a regra do `val`. Desempate: preferir mais texto
    (alpha maior) e menos refinamento, que são as opções mais conservadoras.
    """
    n = len(human_labels)
    results: List[Dict] = []
    for ps, a, r in itertools.product(text_by_set.keys(), alphas, refine_iters):
        cfg = TeacherConfig(prompt_set=ps, alpha=a, refine_iters=r)
        if a == 1.0 and r == 0:
            # zero-shot puro: não depende dos humanos
            t = ImprovedTeacher(text_by_set[ps], logit_scale, cfg)
            preds = t.logits(human_emb).argmax(1)
        else:
            preds = np.empty(n, dtype=np.int64)
            for i in range(n):
                mask = np.arange(n) != i
                t = ImprovedTeacher(text_by_set[ps], logit_scale, cfg).fit(
                    human_emb[mask], human_labels[mask], pool_emb
                )
                preds[i] = t.logits(human_emb[i : i + 1]).argmax(1)[0]
        m = compute_macro_f1_and_accuracy(human_labels, preds, num_classes)
        results.append({**asdict(cfg), "loo_f1_macro": m["f1_macro"], "loo_accuracy": m["accuracy"]})

    best = max(
        results,
        key=lambda d: (round(d["loo_f1_macro"], 4), d["alpha"], -d["refine_iters"]),
    )
    cfg = TeacherConfig(
        prompt_set=best["prompt_set"], alpha=best["alpha"], refine_iters=best["refine_iters"]
    )
    return cfg, results


# --------------------------------------------------------------------------- #
# 3. Pseudo-rótulos com tratamento de ruído (contrato 2.4)
# --------------------------------------------------------------------------- #
def build_pseudo_labels(
    teacher_logits: np.ndarray,
    human_index: np.ndarray,
    human_labels: np.ndarray,
    temperature: float = 1.0,
    conf_threshold: float = 0.5,
    min_per_class: int = 20,
    max_per_class: Optional[int] = None,
    balance_power: float = 0.5,
    human_weight: float = 5.0,
    num_classes: int = 21,
) -> Dict[str, np.ndarray]:
    """Gera rótulos suaves filtrados e pesos de amostragem alinhados ao pool `train`.

    * Rótulo suave: `softmax(logits / T)`; humanos recebem one-hot.
    * Filtro balanceado: dentro de cada classe prevista, mantém quem tem
      confiança >= `conf_threshold`, mas garante pelo menos `min_per_class`
      (os mais confiantes) para classes raras não sumirem, e no máximo
      `max_per_class` para classes grandes não dominarem.
    * Peso: `conf * (n_medio / n_classe) ** balance_power`, média 1 nos pseudo;
      humanos recebem `human_weight`.
    """
    logits = np.asarray(teacher_logits, dtype=np.float32)
    n = logits.shape[0]
    soft = softmax(logits, temperature)
    hard = soft.argmax(1).astype(np.int64)
    conf = soft.max(1)

    keep = np.zeros(n, dtype=bool)
    for c in range(num_classes):
        idx = np.where(hard == c)[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-conf[idx])]
        n_keep = int((conf[idx] >= conf_threshold).sum())
        n_keep = max(n_keep, min(min_per_class, len(idx)))
        if max_per_class is not None:
            n_keep = min(n_keep, max_per_class)
        keep[idx[:n_keep]] = True

    is_human = np.zeros(n, dtype=bool)
    is_human[human_index] = True
    soft[human_index] = np.eye(num_classes, dtype=np.float32)[human_labels]
    hard[human_index] = human_labels
    conf[human_index] = 1.0
    keep[human_index] = True

    weights = np.zeros(n, dtype=np.float32)
    pseudo = keep & ~is_human
    counts = np.bincount(hard[pseudo], minlength=num_classes).astype(np.float32)
    present = counts > 0
    if pseudo.any():
        mean_count = counts[present].mean()
        balance = np.ones(num_classes, dtype=np.float32)
        balance[present] = (mean_count / counts[present]) ** balance_power
        weights[pseudo] = conf[pseudo] * balance[hard[pseudo]]
        weights[pseudo] /= weights[pseudo].mean()
    weights[is_human] = human_weight

    return {
        "soft_labels": soft.astype(np.float32),
        "hard_labels": hard,
        "keep_mask": keep,
        "sample_weights": weights,
        "is_human_labeled": is_human,
    }


def pseudo_label_quality(
    pl: Dict[str, np.ndarray], class_names: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Resumo por classe: quantos pseudo-rótulos ficaram e peso médio."""
    keep, human, hard, w = pl["keep_mask"], pl["is_human_labeled"], pl["hard_labels"], pl["sample_weights"]
    out = {}
    for c, name in enumerate(class_names):
        m = keep & ~human & (hard == c)
        out[name] = {
            "n_pseudo": int(m.sum()),
            "n_human": int((human & (hard == c)).sum()),
            "mean_weight": float(w[m].mean()) if m.any() else 0.0,
        }
    return out


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def human_labels_for_pool(
    data_dir: str, budget: int, pool_ids: Sequence[str], label_to_idx: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    df = load_split_df(os.path.join(data_dir, "labels", f"rotulos_{budget}.csv"), label_to_idx)
    pos = {img_id: i for i, img_id in enumerate(pool_ids)}
    missing = [i for i in df["image_id"] if i not in pos]
    if missing:
        raise ValueError(f"{len(missing)} ids de rotulos_{budget}.csv fora do cache do train.")
    index = np.asarray([pos[i] for i in df["image_id"]], dtype=np.int64)
    return index, df["label_idx"].to_numpy(dtype=np.int64)


def fit_improved_teacher(
    data_dir: str,
    budget: int,
    cache_dir: str = CACHE_DIR,
    use_refinement: bool = True,
) -> Tuple[ImprovedTeacher, Dict, np.ndarray, np.ndarray, List[Dict]]:
    """Ajusta o professor melhorado para o orçamento K. Retorna também o cache do train."""
    class_names, label_to_idx, _ = load_class_mapping(data_dir)
    train = load_teacher_cache("train", cache_dir)
    text = load_text_cache(cache_dir)
    text_by_set = {k.replace("text_", ""): v for k, v in text.items() if k.startswith("text_")}
    scale = float(text["logit_scale"])

    h_idx, h_lab = human_labels_for_pool(data_dir, budget, train["image_ids"], label_to_idx)
    cfg, grid = select_teacher_config(
        train["embeddings"][h_idx],
        h_lab,
        text_by_set,
        scale,
        pool_emb=train["embeddings"] if use_refinement else None,
        refine_iters=(0, 1, 2) if use_refinement else (0,),
        num_classes=len(class_names),
    )
    teacher = ImprovedTeacher(text_by_set[cfg.prompt_set], scale, cfg).fit(
        train["embeddings"][h_idx], h_lab, train["embeddings"]
    )
    return teacher, train, h_idx, h_lab, grid


def make_pseudo_labels(
    data_dir: str,
    budget: int,
    cache_dir: str = CACHE_DIR,
    out_dir: str = PSEUDO_DIR,
    **pl_kwargs,
) -> Dict[str, np.ndarray]:
    """Professor melhorado + pseudo-rótulos para o orçamento K; salva `.npz`."""
    class_names, _, _ = load_class_mapping(data_dir)
    teacher, train, h_idx, h_lab, grid = fit_improved_teacher(data_dir, budget, cache_dir)
    pl = build_pseudo_labels(
        teacher.logits(train["embeddings"]), h_idx, h_lab, num_classes=len(class_names), **pl_kwargs
    )
    pl["image_ids"] = train["image_ids"]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"pseudo_K{budget}.npz"), **pl)
    with open(os.path.join(out_dir, f"teacher_K{budget}.json"), "w") as f:
        json.dump(
            {
                "config": asdict(teacher.config),
                "loo_grid": grid,
                "pseudo_label_params": pl_kwargs,
                "per_class": pseudo_label_quality(pl, class_names),
            },
            f,
            indent=2,
        )
    return pl


def load_or_make_pseudo_labels(
    data_dir: str,
    budget: int,
    cache_dir: str = CACHE_DIR,
    out_dir: str = PSEUDO_DIR,
    **pl_kwargs,
) -> Dict[str, np.ndarray]:
    """Reaproveita `pseudo_K{K}.npz` se foi gerado com os mesmos parâmetros."""
    npz_path = os.path.join(out_dir, f"pseudo_K{budget}.npz")
    meta_path = os.path.join(out_dir, f"teacher_K{budget}.json")
    if os.path.isfile(npz_path) and os.path.isfile(meta_path):
        with open(meta_path) as f:
            if json.load(f).get("pseudo_label_params") == pl_kwargs:
                with np.load(npz_path, allow_pickle=False) as d:
                    pl = {k: d[k] for k in d.files}
                pl["image_ids"] = pl["image_ids"].astype(str)
                return pl
    return make_pseudo_labels(data_dir, budget, cache_dir, out_dir, **pl_kwargs)


def diagnose_on_test_public(
    data_dir: str, budget: int, cache_dir: str = CACHE_DIR, out_dir: str = DIAG_DIR
) -> Dict[str, Dict]:
    """Mede zero-shot e professor melhorado no `test_public` (apenas reporte)."""
    class_names, label_to_idx, _ = load_class_mapping(data_dir)
    test = load_teacher_cache("test_public", cache_dir)
    df = load_split_df(os.path.join(data_dir, "labels", "test_public.csv"), label_to_idx)
    y_map = dict(zip(df["image_id"], df["label_idx"]))
    y = np.asarray([y_map[i] for i in test["image_ids"]], dtype=np.int64)

    teacher, _, _, _, _ = fit_improved_teacher(data_dir, budget, cache_dir)
    reports = {
        "zero_shot_names": diagnose_predictions(y, test["preds"], class_names, test["is_infrared"]),
        f"improved_K{budget}": diagnose_predictions(
            y, teacher.logits(test["embeddings"]).argmax(1), class_names, test["is_infrared"]
        ),
    }
    reports[f"improved_K{budget}"]["config"] = asdict(teacher.config)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"test_public_K{budget}.json"), "w") as f:
        json.dump(reports, f, indent=2)
    return reports


def main() -> None:
    p = argparse.ArgumentParser(description="Professor melhorado + pseudo-rótulos (Trilha C).")
    p.add_argument("--data-dir", default="data/pa3_dados")
    p.add_argument("--budget", type=int, default=5, choices=[1, 5, 20])
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--diagnose", action="store_true", help="Reporta o professor no test_public.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--min-per-class", type=int, default=20)
    args = p.parse_args()

    if args.diagnose:
        for name, rep in diagnose_on_test_public(args.data_dir, args.budget, args.cache_dir).items():
            print(f"\n== {name}: F1 macro {rep['f1_macro']:.4f} | acc {rep['accuracy']:.4f}")
            for k in ("day", "night_ir"):
                if k in rep:
                    print(f"   {k}: n={rep[k]['n']} F1 {rep[k]['f1_macro']:.4f}")
            for c in rep["top_confusions"][:5]:
                print(f"   {c['true']} -> {c['pred']}: {c['count']} ({c['rate']:.0%})")

    pl = make_pseudo_labels(
        args.data_dir,
        args.budget,
        args.cache_dir,
        temperature=args.temperature,
        conf_threshold=args.conf_threshold,
        min_per_class=args.min_per_class,
    )
    print(
        f"\n[pseudo] K={args.budget}: {int(pl['keep_mask'].sum())} mantidos "
        f"({int(pl['is_human_labeled'].sum())} humanos) de {len(pl['keep_mask'])}"
    )


if __name__ == "__main__":
    main()
