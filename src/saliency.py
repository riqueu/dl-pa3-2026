"""Módulo de interpretabilidade (Grad-CAM e Oclusão) para a Parte 3 do PA3.

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-baseline
"""

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import is_infrared_image


def _find_default_target_layer(model: nn.Module) -> nn.Module:
    """Localiza a última camada Conv2d (preferindo `layer4` na ResNet-18)."""
    search_root = getattr(model, "layer4", model)
    last_conv: Optional[nn.Module] = None
    for module in search_root.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Nenhuma camada nn.Conv2d encontrada para Grad-CAM.")
    return last_conv


class GradCAM:
    """Calcula mapas de saliência Grad-CAM sobre a última camada convolucional."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = (
            target_layer if target_layer is not None else _find_default_target_layer(model)
        )
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Gera mapa Grad-CAM `(H, W)` normalizado em `[0, 1]`."""
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"input_tensor deve ter shape (1, 3, H, W), recebido {tuple(input_tensor.shape)}."
            )

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        x = input_tensor.requires_grad_(True)
        logits = self.model(x)
        cls_idx = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)

        score = logits[0, cls_idx]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks de Grad-CAM não capturaram ativações/gradientes.")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        self._activations = None
        self._gradients = None
        self.model.zero_grad(set_to_none=True)
        return cam_np

    def close(self) -> None:
        """Remove os hooks registrados na camada alvo."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


@torch.no_grad()
def occlusion_sensitivity(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    patch_size: int = 32,
    stride: int = 16,
    batch_size: int = 32,
) -> np.ndarray:
    """Mapa de saliência por oclusão deslizante (queda de probabilidade na classe alvo)."""
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError(
            f"input_tensor deve ter shape (1, 3, H, W), recebido {tuple(input_tensor.shape)}."
        )

    model.eval()
    _, _, h, w = input_tensor.shape
    base_logits = model(input_tensor)
    base_probs = torch.softmax(base_logits, dim=1)
    cls_idx = int(base_probs.argmax(dim=1).item()) if target_class is None else int(target_class)
    base_prob = float(base_probs[0, cls_idx].item())

    coords = [
        (y, min(y + patch_size, h), x, min(x + patch_size, w))
        for y in range(0, h, stride)
        for x in range(0, w, stride)
    ]

    drop_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for start in range(0, len(coords), batch_size):
        chunk = coords[start : start + batch_size]
        occluded = input_tensor.repeat(len(chunk), 1, 1, 1).clone()
        for idx, (y1, y2, x1, x2) in enumerate(chunk):
            occluded[idx, :, y1:y2, x1:x2] = 0.0

        probs = torch.softmax(model(occluded), dim=1)[:, cls_idx].cpu().numpy()
        for (y1, y2, x1, x2), occ_prob in zip(chunk, probs):
            drop = max(0.0, base_prob - float(occ_prob))
            drop_map[y1:y2, x1:x2] += drop
            count_map[y1:y2, x1:x2] += 1.0

    occ_map = np.divide(drop_map, count_map, out=np.zeros_like(drop_map), where=count_map > 0)
    m_min, m_max = float(occ_map.min()), float(occ_map.max())
    if m_max > m_min:
        occ_map = (occ_map - m_min) / (m_max - m_min)
    else:
        occ_map = np.zeros_like(occ_map, dtype=np.float32)
    return occ_map


def _compute_center_vs_border_ratio(heatmap: np.ndarray) -> float:
    """Razão entre ativação média no centro (50% central) e nas bordas."""
    h, w = heatmap.shape
    y1, y2 = h // 4, (3 * h) // 4
    x1, x2 = w // 4, (3 * w) // 4
    center_mean = float(heatmap[y1:y2, x1:x2].mean())

    mask = np.ones_like(heatmap, dtype=bool)
    mask[y1:y2, x1:x2] = False
    border_mean = float(heatmap[mask].mean()) if mask.any() else 0.0
    return center_mean / (border_mean + 1e-6)


def select_failure_and_success_cases(
    model: nn.Module,
    dataset,
    class_names: Sequence[str],
    device: str = "cpu",
    n_errors: int = 3,
    n_successes: int = 5,
) -> Dict[str, List[Dict]]:
    """Seleciona 3 erros e 5 acertos no `test_public` com mapas Grad-CAM (Parte 3)."""
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            if isinstance(item, dict):
                tensor = item["image"]
                true_idx = int(item["label"])
                img_id = str(item["image_id"])
            else:
                tensor, true_idx, img_id = item[0], int(item[1]), str(item[2])

            logits = model(tensor.unsqueeze(0).to(target_device))
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            conf = float(probs[pred_idx].item())

            records.append({
                "idx": idx,
                "image_id": img_id,
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "confidence": conf,
                "is_correct": (pred_idx == true_idx),
            })

    errors_pool = sorted(
        [r for r in records if not r["is_correct"]],
        key=lambda r: r["confidence"],
        reverse=True,
    )
    success_pool = sorted(
        [r for r in records if r["is_correct"]],
        key=lambda r: r["confidence"],
        reverse=True,
    )

    def _pick_diverse(pool: List[Dict], limit: int) -> List[Dict]:
        chosen: List[Dict] = []
        seen_classes = set()
        for rec in pool:
            if len(chosen) >= limit:
                break
            if rec["true_idx"] not in seen_classes:
                chosen.append(rec)
                seen_classes.add(rec["true_idx"])
        for rec in pool:
            if len(chosen) >= limit:
                break
            if rec not in chosen:
                chosen.append(rec)
        return chosen

    selected_errors = _pick_diverse(errors_pool, n_errors)
    selected_successes = _pick_diverse(success_pool, n_successes)

    cam = GradCAM(model)
    try:
        def _enrich(selected: List[Dict]) -> List[Dict]:
            enriched = []
            for rec in selected:
                idx = rec["idx"]
                item = dataset[idx]
                tensor = item["image"] if isinstance(item, dict) else item[0]
                img_rgb = dataset.load_raw_rgb(idx)
                hm_224 = cam.generate(
                    tensor.unsqueeze(0).to(target_device),
                    target_class=rec["pred_idx"],
                )
                hm_orig = cv2.resize(
                    hm_224,
                    (img_rgb.shape[1], img_rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                hm_orig = np.clip(hm_orig, 0.0, 1.0).astype(np.float32)

                is_ir = is_infrared_image(img_rgb)
                focus_ratio = _compute_center_vs_border_ratio(hm_orig)
                note = (
                    f"{'Noite/IR' if is_ir else 'Dia'} | "
                    f"foco centro/borda={focus_ratio:.2f}"
                )

                enriched.append({
                    "image_id": rec["image_id"],
                    "image_rgb": img_rgb,
                    "heatmap": hm_orig,
                    "true_label": str(class_names[rec["true_idx"]]),
                    "pred_label": str(class_names[rec["pred_idx"]]),
                    "confidence": rec["confidence"],
                    "is_infrared": bool(is_ir),
                    "center_border_ratio": float(focus_ratio),
                    "note": note,
                })
            return enriched

        return {
            "errors": _enrich(selected_errors),
            "successes": _enrich(selected_successes),
        }
    finally:
        cam.close()
