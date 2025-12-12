from __future__ import annotations

from typing import Tuple, Dict, List

import tensorflow as tf
import pandas as pd

from src.utils.config_utils import TrainConfig
from src.data.ch_utils import ClickHouseClient


def build_label_mapping(class_names: List[str]) -> Dict[str, int]:
    """Mapeia nome de classe -> índice inteiro."""
    return {name: index for index, name in enumerate(class_names)}


def load_splits_from_clickhouse(cfg: TrainConfig) -> Dict[str, pd.DataFrame]:
    """
    Carrega DataFrames de patches a partir do ClickHouse para os splits:
    - train, val, test

    Retorna: dict[split] -> DataFrame com colunas:
      - patient_id, slide_id, image_path, label, dataset_source
    """
    assert cfg.clickhouse is not None
    ch = ClickHouseClient(cfg.clickhouse)
    dfs = ch.load_patches(
        splits=["train", "val", "test"],
        limit_per_split=cfg.clickhouse.limit_per_split if cfg.clickhouse else None,
    )
    return dfs


def make_tf_dataset(
    df: pd.DataFrame,
    label_map: Dict[str, int],
    class_names: List[str],  # mantido por consistência de assinatura
    batch_size: int,
    shuffle: bool,
    patch_size: int,
) -> tf.data.Dataset:
    """
    Constrói um tf.data.Dataset a partir de um DataFrame com colunas:
      - image_path
      - label (string)
    """
    paths = df["image_path"].tolist()
    labels = [label_map[label] for label in df["label"].tolist()]

    ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_fn(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(path)
        # Usamos argumento posicional para evitar problema com stubs antigos
        img = tf.io.decode_png(img_bytes, 3)
        img = tf.image.resize(img, (patch_size, patch_size))
        img = tf.cast(img, tf.float32)

        # Normalização simples; pode ser substituída por algo específico de histo
        img = tf.math.divide(img, 255.0)
        img = tf.math.divide(img - 0.5, 0.25)

        return img, tf.cast(label, tf.int32)

    ds = ds.map(_load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_tf_datasets_from_clickhouse(
    cfg: TrainConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    Cria tf.data.Datasets para train/val/test a partir do ClickHouse.
    """
    dfs = load_splits_from_clickhouse(cfg)
    label_map = build_label_mapping(cfg.class_names)

    train_ds = make_tf_dataset(
        dfs["train"], label_map, cfg.class_names, cfg.batch_size, True, cfg.patch_size
    )
    val_ds = make_tf_dataset(
        dfs["val"], label_map, cfg.class_names, cfg.batch_size, False, cfg.patch_size
    )
    test_ds = make_tf_dataset(
        dfs["test"], label_map, cfg.class_names, cfg.batch_size, False, cfg.patch_size
    )
    return train_ds, val_ds, test_ds, label_map
