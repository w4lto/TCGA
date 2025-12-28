from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd
import tensorflow as tf

from src.data.ch_utils import ClickHouseClient
from src.utils.config_utils import TrainConfig


def _normalize_stage_label(x: str) -> str:
    s = str(x).strip()
    if not s:
        return s
    if s in {"I", "II", "III", "IV"}:
        return s

    upper = s.upper()
    if upper.startswith("STAGE"):
        parts = upper.split()
        roman = parts[1] if len(parts) >= 2 else upper.replace("STAGE", "").strip()

        if roman.startswith("III"):
            return "III"
        if roman.startswith("II"):
            return "II"
        if roman.startswith("IV"):
            return "IV"
        if roman.startswith("I"):
            return "I"

    return s


def _build_label_map(all_labels: Union[pd.Series, List[str], Iterable[str]]) -> Dict[str, int]:
    if isinstance(all_labels, pd.Series):
        raw = [str(x) for x in all_labels.dropna().tolist()]
    else:
        raw = [str(x) for x in list(all_labels) if x is not None]

    labels = [_normalize_stage_label(x) for x in raw if str(x).strip() != ""]
    uniq = sorted(set(labels))

    preferred = ["I", "II", "III", "IV"]
    if uniq and all(u in preferred for u in uniq):
        uniq = [u for u in preferred if u in uniq]

    return {lab: i for i, lab in enumerate(uniq)}


def _get_path_column(df: pd.DataFrame) -> str:
    if "patch_path" in df.columns:
        return "patch_path"
    if "image_path" in df.columns:
        return "image_path"
    raise KeyError(f"DataFrame sem coluna de path. Colunas: {list(df.columns)}")


def _decode_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    Lê PNG/JPG e retorna float32 [H,W,3] em [0,1].

    Importante: `image_size` aqui é int Python (já normalizado fora do grafo),
    e vira um Tensor int32 constante para o resize.
    """
    size2 = tf.constant([image_size, image_size], dtype=tf.int32)  # 1-D int32

    data = tf.io.read_file(path)

    def _decode_png():
        return tf.image.decode_png(data, channels=3)

    def _decode_jpg():
        return tf.image.decode_jpeg(data, channels=3)

    img = tf.cond(
        tf.strings.regex_full_match(tf.strings.lower(path), r".*\.png"),
        _decode_png,
        _decode_jpg,
    )

    img = tf.image.resize(img, size2, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _coerce_int(v: object, name: str) -> int:
    """
    Coerção robusta para configs vindos de YAML:
      - 224
      - 224.0
      - "224"
      - "224.0"
    """
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(round(v))
    if isinstance(v, str):
        s = v.strip()
        try:
            return int(s)
        except ValueError:
            return int(round(float(s)))
    raise TypeError(f"{name} inválido: {v!r} (tipo={type(v)})")


def make_tf_dataset(
    df: pd.DataFrame,
    label_map: Dict[str, int],
    image_size: object,
    batch_size: object,
    shuffle: bool,
    seed: object,
) -> tf.data.Dataset:
    path_col = _get_path_column(df)

    image_size_i = _coerce_int(image_size, "image_size")
    batch_size_i = _coerce_int(batch_size, "batch_size")
    seed_i = _coerce_int(seed, "seed")

    paths = df[path_col].astype(str).tolist()
    labels_str = df["stage_label"].astype(str).tolist()
    labels_str = [_normalize_stage_label(x) for x in labels_str]

    missing = sorted(set(labels_str) - set(label_map.keys()))
    if missing:
        raise RuntimeError(f"Labels não mapeadas: {missing}. label_map={label_map}")

    labels = [label_map[x] for x in labels_str]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(10_000, len(paths)),
            seed=seed_i,
            reshuffle_each_iteration=True,
        )

    def _map_fn(p, y):
        img = _decode_image(p, image_size=image_size_i)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size_i, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_tf_datasets_from_clickhouse(
    cfg: TrainConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    assert cfg.clickhouse is not None

    ch = ClickHouseClient(cfg.clickhouse)

    dfs = ch.load_patches(
        splits=["train", "val", "test"],
        limit_per_split=getattr(cfg.clickhouse, "limit_per_split", None),
    )

    for sp in ["train", "val", "test"]:
        if sp not in dfs or dfs[sp] is None or len(dfs[sp]) == 0:
            raise RuntimeError(f"Split '{sp}' retornou 0 patches do ClickHouse. Verifique ingest/patch_injector.")

    df_train = dfs["train"].copy()
    df_val = dfs["val"].copy()
    df_test = dfs["test"].copy()

    df_train["stage_label"] = df_train["stage_label"].astype(str).map(_normalize_stage_label)
    df_val["stage_label"] = df_val["stage_label"].astype(str).map(_normalize_stage_label)
    df_test["stage_label"] = df_test["stage_label"].astype(str).map(_normalize_stage_label)

    label_map = _build_label_map(df_train["stage_label"])

    train_ds = make_tf_dataset(
        df_train,
        label_map=label_map,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )
    val_ds = make_tf_dataset(
        df_val,
        label_map=label_map,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=False,
        seed=cfg.seed,
    )
    test_ds = make_tf_dataset(
        df_test,
        label_map=label_map,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        shuffle=False,
        seed=cfg.seed,
    )

    return train_ds, val_ds, test_ds, label_map
