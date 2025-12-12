from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow import keras

from src.utils.config_utils import load_config
from src.utils.tf_device_utils import setup_tf_device
from src.data.tf_dataset import (
    make_tf_dataset,
    load_splits_from_clickhouse,
    build_label_mapping,
)
from src.training.metrics_utils import (
    compute_patch_metrics_multiclass,
    aggregate_by_slide,
    compute_slide_metrics_multiclass,
    dump_classification_report,
)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    device_str, device_desc = setup_tf_device(preferred=cfg.device)
    print(f"[eval_tf] TensorFlow device: {device_desc}")

    dfs = load_splits_from_clickhouse(cfg)
    df_test = dfs["test"]

    label_map = build_label_mapping(cfg.class_names)

    with tf.device(device_str):
        test_ds = make_tf_dataset(
            df_test,
            label_map,
            cfg.class_names,
            batch_size=cfg.batch_size,
            shuffle=False,
            patch_size=cfg.patch_size,
        )

        model_path = cfg.output_dir / "models" / "ensemble_tf"
        model = keras.models.load_model(model_path)

        y_proba = model.predict(test_ds, verbose=1)
        y_true = df_test["label"].map(lambda x: label_map[x]).to_numpy()
        slide_ids = df_test["slide_id"].to_numpy()

    patch_metrics = compute_patch_metrics_multiclass(
        y_true=y_true,
        y_proba=y_proba,
        class_names=cfg.class_names,
    )
    y_pred = y_proba.argmax(axis=1)
    cls_report_patch = dump_classification_report(
        y_true,
        y_pred,
        cfg.class_names,
    )

    print("\n=== Métricas PATCH-level (multi-classe) ===")
    for key, value in patch_metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nClassification report (patch-level):")
    print(cls_report_patch)


    y_true_slide_soft, y_proba_slide_soft = aggregate_by_slide(
        y_true=y_true,
        y_proba=y_proba,
        slide_ids=slide_ids,
        method="mean_prob",
    )
    slide_metrics_soft = compute_slide_metrics_multiclass(
        y_true_slide_soft,
        y_proba_slide_soft,
        class_names=cfg.class_names,
    )
    y_pred_slide_soft = y_proba_slide_soft.argmax(axis=1)
    cls_report_slide_soft = dump_classification_report(
        y_true_slide_soft,
        y_pred_slide_soft,
        cfg.class_names,
    )

    # majority vote / hard voting
    y_true_slide_mv, y_proba_slide_mv = aggregate_by_slide(
        y_true=y_true,
        y_proba=y_proba,
        slide_ids=slide_ids,
        method="majority_vote",
    )
    slide_metrics_mv = compute_slide_metrics_multiclass(
        y_true_slide_mv,
        y_proba_slide_mv,
        class_names=cfg.class_names,
    )
    y_pred_slide_mv = y_proba_slide_mv.argmax(axis=1)
    cls_report_slide_mv = dump_classification_report(
        y_true_slide_mv,
        y_pred_slide_mv,
        cfg.class_names,
    )

    print("\n=== Métricas SLIDE-level (soft voting / mean_prob) ===")
    for key, value in slide_metrics_soft.items():
        print(f"{key}: {value:.4f}")
    print("\nClassification report (slide-level, soft voting):")
    print(cls_report_slide_soft)

    print("\n=== Métricas SLIDE-level (majority_vote) ===")
    for key, value in slide_metrics_mv.items():
        print(f"{key}: {value:.4f}")
    print("\nClassification report (slide-level, majority_vote):")
    print(cls_report_slide_mv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
