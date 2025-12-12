from __future__ import annotations

import argparse
import random
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.utils.config_utils import load_config, TrainConfig
from src.utils.logging_utils import setup_logging
from src.utils.mlflow_utils import init_mlflow, start_run, log_metrics
from src.utils.tf_device_utils import setup_tf_device

from src.data.tf_dataset import (
    create_tf_datasets_from_clickhouse,
    load_splits_from_clickhouse,
    build_label_mapping,
    make_tf_dataset,
)
from src.models.kimianet_backbone_tf import build_kimianet_backbone_tf
from src.models.resnet_backbone_tf import build_resnet_backbone_tf
from src.models.ensemble_tf import build_ensemble_model_tf
from src.data.ch_utils import ClickHouseClient
from src.training.metrics_utils import (
    compute_patch_metrics_multiclass,
    aggregate_by_slide,
    compute_slide_metrics_multiclass,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_class_weights_dynamic(cfg: TrainConfig, label_map: Dict[str, int]):
    """
    Calcula pesos de classe dinâmicos a partir da distribuição de patches no split 'train',
    combinando peso inversamente proporcional à frequência com pesos clínicos (stage_clinical_weights).
    """
    assert cfg.clickhouse is not None
    ch = ClickHouseClient(cfg.clickhouse)
    client = ch.get_client()
    df = client.query_df(
        f"SELECT stage_label FROM {cfg.clickhouse.table_patches} WHERE split = 'train'"
    )
    counts = df["stage_label"].value_counts().to_dict()

    num_classes = len(cfg.class_names)
    freqs = np.zeros(num_classes, dtype=float)
    for name, index in label_map.items():
        freqs[index] = counts.get(name, 0)

    freqs[freqs == 0] = 1.0
    inv = 1.0 / freqs
    inv = inv / inv.sum()

    if cfg.stage_clinical_weights:
        boost = np.array(
            [cfg.stage_clinical_weights.get(name, 1.0) for name in cfg.class_names],
            dtype=float,
        )
        weights = inv * boost
    else:
        weights = inv

    weights = weights / weights.sum()
    return {int(i): float(w) for i, w in enumerate(weights)}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    log_dir = cfg.output_dir / "logs"
    logger = setup_logging(log_dir, "train_tf")

    # Configura dispositivo TF + memory growth
    device_str, device_desc = setup_tf_device(preferred=cfg.device)
    logger.info(f"TensorFlow device: {device_desc}")

    # Inicializa MLflow
    init_mlflow(cfg.mlflow_tracking_uri, cfg.experiment_name)
    params: Dict[str, Any] = {
        "task_type": cfg.task_type,
        "class_names": ",".join(cfg.class_names),
        "batch_size": cfg.batch_size,
        "patch_size": cfg.patch_size,
        "stride": cfg.stride,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "dropout": cfg.dropout,
        "device": cfg.device,
        "slide_aggregation": getattr(cfg, "slide_aggregation", "mean_prob"),
    }
    start_run(cfg.run_name, params=params)

    with tf.device(device_str):
        # Datasets
        train_ds, val_ds, test_ds, label_map = create_tf_datasets_from_clickhouse(cfg)

        num_classes = len(cfg.class_names)

        # Backbones individuais
        backbones: List[keras.Model] = []
        backbones.append(
            build_kimianet_backbone_tf(
                num_classes=num_classes,
                weights_path=cfg.backbone_kimianet_weights,
                dropout=cfg.dropout,
            )
        )
        if cfg.use_resnet:
            backbones.append(
                build_resnet_backbone_tf(
                    num_classes=num_classes,
                    dropout=cfg.dropout,
                )
            )

        # Ensemble (média de logits)
        model = build_ensemble_model_tf(backbones, num_classes=num_classes)

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]

        optimizer = keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
        )

        # Class weights dinâmicos
        class_weight = None
        if cfg.dynamic_class_weights:
            class_weight = compute_class_weights_dynamic(cfg, label_map)
            logger.info("class_weight inicial: %s", class_weight)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.num_epochs,
            class_weight=class_weight,
            callbacks=callbacks,
        )

        # Métricas de treino/val da última época
        last_epoch = len(history.history["loss"]) - 1
        metrics_to_log = {
            f"train_{k}": float(history.history[k][last_epoch])
            for k in history.history.keys()
            if not k.startswith("val_")
        }
        metrics_to_log.update(
            {
                f"val_{k[4:]}": float(history.history[k][last_epoch])
                for k in history.history.keys()
                if k.startswith("val_")
            }
        )
        log_metrics(metrics_to_log, step=last_epoch)

        # Avaliação "básica" no tf.data test
        test_metrics = model.evaluate(test_ds, return_dict=True)
        log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
        logger.info("Métricas básicas test_ds: %s", test_metrics)

        # ---------- Métricas avançadas patch/slide via sklearn ----------
        logger.info("Calculando métricas avançadas patch-level e slide-level...")

        dfs = load_splits_from_clickhouse(cfg)
        df_test = dfs["test"]
        label_map = build_label_mapping(cfg.class_names)

        test_ds_pred = make_tf_dataset(
            df_test,
            label_map,
            cfg.class_names,
            batch_size=cfg.batch_size,
            shuffle=False,
            patch_size=cfg.patch_size,
        )

        y_proba = model.predict(test_ds_pred, verbose=0)
        y_true = df_test["label"].map(lambda x: label_map[x]).to_numpy()
        slide_ids = df_test["slide_id"].to_numpy()

        # Patch-level
        patch_metrics = compute_patch_metrics_multiclass(
            y_true=y_true,
            y_proba=y_proba,
            class_names=cfg.class_names,
        )
        logger.info("Patch-level metrics: %s", patch_metrics)
        log_metrics({f"patch_{k}": v for k, v in patch_metrics.items()})

        # Slide-level (soft voting)
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
        logger.info("Slide-level metrics (mean_prob): %s", slide_metrics_soft)
        log_metrics({f"slide_soft_{k}": v for k, v in slide_metrics_soft.items()})

        # Slide-level (majority vote)
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
        logger.info("Slide-level metrics (majority_vote): %s", slide_metrics_mv)
        log_metrics({f"slide_mv_{k}": v for k, v in slide_metrics_mv.items()})

        # Salvar modelo
        out_dir = cfg.output_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "ensemble_tf")
        logger.info("Modelo salvo em ensemble_tf.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
