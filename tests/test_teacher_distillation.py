"""Testes sintéticos da Trilha C (professor, pseudo-rótulos e destilação)."""

import numpy as np
import pytest
import torch

from src.distillation import DistillationLoss
from src.pseudo_labels import (
    ImprovedTeacher,
    TeacherConfig,
    build_pseudo_labels,
    diagnose_predictions,
    select_teacher_config,
)
from src.teacher import ALLOWED_SPLITS, build_prompts, build_teacher_cache, softmax

C, D = 4, 16
NAMES = ["a_b", "c", "d", "e"]


def _clusters(n_per=30, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(C, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    y = np.repeat(np.arange(C), n_per)
    x = centers[y] + noise * rng.normal(size=(len(y), D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32), y, centers.astype(np.float32)


def test_prompts_names_and_rich():
    p = build_prompts(["american_black_bear", "horse"], "names")
    assert p[0] == ["a photo of american black bear."]
    rich = build_prompts(["horse"], "rich")[0]
    assert any("Equus caballus" in s for s in rich)


def test_teacher_refuses_val(tmp_path):
    assert "val" not in ALLOWED_SPLITS
    with pytest.raises(ValueError):
        build_teacher_cache(str(tmp_path), splits=["val"], cache_dir=str(tmp_path))


def test_diagnose_shapes_and_ir_split():
    y = np.array([0, 0, 1, 1, 2, 3])
    p = np.array([0, 1, 1, 1, 2, 2])
    rep = diagnose_predictions(y, p, NAMES, is_infrared=[0, 0, 0, 1, 1, 1])
    assert np.array(rep["confusion_matrix"]).shape == (C, C)
    assert rep["top_confusions"][0]["count"] == 1
    assert rep["day"]["n"] == 3 and rep["night_ir"]["n"] == 3


def test_improved_teacher_beats_bad_text():
    x, y, centers = _clusters()
    rng = np.random.default_rng(1)
    bad_text = centers + 1.5 * rng.normal(size=centers.shape)  # texto ruim
    zs = ImprovedTeacher(bad_text, 30.0, TeacherConfig(alpha=1.0))
    human = np.concatenate([np.where(y == c)[0][:3] for c in range(C)])
    imp = ImprovedTeacher(bad_text, 30.0, TeacherConfig(alpha=0.25, refine_iters=1)).fit(
        x[human], y[human], x
    )
    acc_zs = (zs.logits(x).argmax(1) == y).mean()
    acc_imp = (imp.logits(x).argmax(1) == y).mean()
    assert acc_imp >= acc_zs and acc_imp > 0.9


def test_select_config_loo_runs():
    x, y, centers = _clusters(n_per=5)
    cfg, grid = select_teacher_config(
        x, y, {"names": centers}, 30.0, pool_emb=None, refine_iters=(0,), num_classes=C
    )
    assert cfg.prompt_set == "names" and len(grid) == 5


def test_pseudo_labels_contract_and_rare_class_kept():
    rng = np.random.default_rng(0)
    n = 200
    logits = rng.normal(size=(n, C)).astype(np.float32)
    logits[:150, 0] += 6  # classe 0 dominante e confiante
    logits[150:155, 3] += 0.5  # classe 3 rara e pouco confiante
    h_idx, h_lab = np.array([0, 160]), np.array([0, 2])
    pl = build_pseudo_labels(logits, h_idx, h_lab, conf_threshold=0.9, min_per_class=3, num_classes=C)

    for k, dt in [("soft_labels", np.float32), ("hard_labels", np.int64), ("keep_mask", bool),
                  ("sample_weights", np.float32), ("is_human_labeled", bool)]:
        assert pl[k].dtype == dt and len(pl[k]) == n
    np.testing.assert_allclose(pl["soft_labels"].sum(1), 1.0, atol=1e-5)
    assert pl["keep_mask"][h_idx].all() and pl["hard_labels"][160] == 2
    kept_pseudo = pl["keep_mask"] & ~pl["is_human_labeled"]
    assert (pl["hard_labels"][kept_pseudo] == 3).sum() >= 1  # rara não sumiu
    # peso por amostra maior para classe rara do que para a dominante
    w = pl["sample_weights"]
    h = pl["hard_labels"]
    assert w[kept_pseudo & (h == 3)].mean() > w[kept_pseudo & (h == 0)].mean()


def test_distillation_loss_matches_ce_and_kl():
    torch.manual_seed(0)
    z = torch.randn(6, C, requires_grad=True)
    hard = torch.tensor([0, 1, 2, 3, 0, 1])
    soft = torch.softmax(torch.randn(6, C), 1)
    is_h = torch.tensor([1, 1, 0, 0, 0, 0])
    loss = DistillationLoss(temperature=2.0, human_lambda=1.0)(z, soft, hard, is_h)
    ce = torch.nn.functional.cross_entropy(z[:2], hard[:2], reduction="sum")
    kd = -(soft[2:] * torch.log_softmax(z[2:] / 2.0, 1)).sum() * 4.0
    assert torch.isclose(loss, (ce + kd) / 6)
    loss.backward()
    assert z.grad is not None


def test_softmax_temperature():
    l = np.array([[2.0, 0.0]])
    assert softmax(l, 10.0)[0, 0] < softmax(l, 1.0)[0, 0]
