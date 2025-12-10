import argparse
import random

import numpy as np
import tensorflow as tf

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.mlflow_utils import init_mlflow, start_run, log_metrics
from src.utils.tf_device_utils import setup_tf_device
from src.data.tf_dataset import create_tf_datasets_from_clickhouse
from src.models.kimianet_backbone_tf import build_kimianet_backbone_tf
from src.models.resnet_backbone_tf import build_resnet_backbone_tf
from src.models.ensemble_tf import build_ensemble_model_tf
from src.data.ch_utils import ClickHouseClient


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_class_weights_dynamic(cfg, label_map):
    assert cfg.clickhouse is not None
    ch = ClickHouseClient(cfg.clickhouse)
    client = ch.get_client()
    df = client.query_df("SELECT stage_label FROM tcga_patches WHERE split = 'train'")
    counts = df["stage_label"].value_counts().to_dict()

    num_classes = len(cfg.class_names)
    freqs = np.zeros(num_classes, dtype=float)
    for name, idx in label_map.items():
        freqs[idx] = counts.get(name, 0)

    freqs[freqs == 0] = 1.0
    inv = 1.0 / freqs
    inv = inv / inv.sum()

    if cfg.stage_clinical_weights:
        boost = np.array(
            [cfg.stage_clinical_weights.get(name, 1.0) for name in cfg.class_names],
            dtype=float,
        )
        w = inv * boost
    else:
        w = inv

    w = w / w.sum()
    return {int(i): float(wi) for i, wi in enumerate(w)}


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    log_dir = cfg.output_dir / "logs"
    logger = setup_logging(log_dir, "train_tf")

    # Configura dispositivo TF + memory growth
    device_str, device_desc = setup_tf_device(preferred=cfg.device)
    logger.info(f"TensorFlow device: {device_desc}")

    init_mlflow(cfg.mlflow_tracking_uri, cfg.experiment_name)
    start_run(cfg.run_name, params={k: str(v) for k, v in cfg.__dict__.items()})

    # A partir daqui, garantimos que a memory growth já está configurada
    with tf.device(device_str):
        train_ds, val_ds, test_ds, label_map = create_tf_datasets_from_clickhouse(cfg)

        num_classes = len(cfg.class_names)

        backbones = []
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

        model = build_ensemble_model_tf(backbones, num_classes=num_classes)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
        )

        class_weight = None
        if cfg.dynamic_class_weights:
            class_weight = compute_class_weights_dynamic(cfg, label_map)
            logger.info(f"class_weight inicial: {class_weight}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
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

        test_metrics = model.evaluate(test_ds, return_dict=True)
        log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})

        out_dir = cfg.output_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "ensemble_tf")
        logger.info("Modelo salvo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
