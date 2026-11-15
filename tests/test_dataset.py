"""Testes unitários do módulo `src/dataset.py` (sintéticos e dataset real)."""

import os

import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import (
    DEFAULT_CLASSES,
    ENA24Dataset,
    get_transforms,
    get_unlabeled_pool_ids,
    is_infrared_image,
    load_class_mapping,
    load_split_df,
)


def test_default_classes_and_fallback_mapping(tmp_path):
    assert len(DEFAULT_CLASSES) == 21
    assert DEFAULT_CLASSES[0] == "american_black_bear"
    assert DEFAULT_CLASSES[-1] == "woodchuck"

    classes, l2i, i2l = load_class_mapping(str(tmp_path))
    assert classes == DEFAULT_CLASSES
    assert l2i["horse"] == 13
    assert i2l[13] == "horse"


def test_is_infrared_image_color_vs_grayscale():
    gray = np.full((64, 64, 3), 120, dtype=np.uint8)
    assert is_infrared_image(gray) is True

    color = np.zeros((64, 64, 3), dtype=np.uint8)
    color[..., 0] = 220
    color[..., 1] = 80
    color[..., 2] = 30
    assert is_infrared_image(color) is False


def test_dataset_labeled_unlabeled_and_dict_modes(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name, color in [("abc123", (200, 50, 50)), ("def456", (100, 100, 100))]:
        Image.new("RGB", (320, 240), color=color).save(img_dir / f"{name}.jpg")

    csv_path = tmp_path / "split.csv"
    pd.DataFrame([
        {"image_id": "abc123", "label": "horse"},
        {"image_id": "def456", "label": "bobcat"},
    ]).to_csv(csv_path, index=False)

    _, l2i, _ = load_class_mapping(str(tmp_path))
    df = load_split_df(str(csv_path), l2i)
    assert df["label_idx"].tolist() == [13, 3]

    ds_labeled = ENA24Dataset(
        image_dir=str(img_dir),
        csv_path=str(csv_path),
        label_to_idx=l2i,
        transform=get_transforms(224, is_train=False),
    )
    assert len(ds_labeled) == 2
    tensor, lbl, img_id = ds_labeled[0]
    assert tensor.shape == (3, 224, 224)
    assert lbl == 13
    assert img_id == "abc123"
    assert ds_labeled.load_raw_rgb(0).shape == (240, 320, 3)

    soft = np.full((2, 21), 1.0 / 21.0, dtype=np.float32)
    weights = np.array([1.5, 0.8], dtype=np.float32)
    ds_dict = ENA24Dataset(
        image_dir=str(img_dir),
        csv_path=None,
        transform=get_transforms(224, is_train=True),
        soft_labels=soft,
        sample_weights=weights,
        return_dict=True,
    )
    item = ds_dict[1]
    assert item["image"].shape == (3, 224, 224)
    assert item["label"] == -1
    assert item["soft_label"].shape == (21,)
    assert np.isclose(item["weight"], 0.8)


def test_real_ena24_splits_if_present():
    if not os.path.isdir("data/images/train"):
        return

    classes, l2i, _ = load_class_mapping("data")
    assert len(classes) == 21

    assert len(get_unlabeled_pool_ids("data")) == 6505
    assert len(get_unlabeled_pool_ids("data", exclude_labeled_csv="data/labels/rotulos_5.csv")) == 6400

    ds_val = ENA24Dataset("data/images/val", "data/labels/val.csv", label_to_idx=l2i)
    ds_test = ENA24Dataset("data/images/test_public", "data/labels/test_public.csv", label_to_idx=l2i)
    assert len(ds_val) == 202
    assert len(ds_test) == 811
    t, y, img_id = ds_val[0]
    assert t.shape == (3, 224, 224)
    assert 0 <= y < 21
    assert isinstance(img_id, str)
