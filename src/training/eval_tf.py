# src/training/eval_tf.py

import argparse

import tensorflow as tf

from src.utils.config import load_config
from src.utils.tf_device_utils import setup_tf_device   # <-- NOVO
from src.data.tf_dataset import create_tf_datasets_from_clickhouse


def main(config_path: str):
    cfg = load_config(config_path)

    device_str, device_desc = setup_tf_device(preferred=cfg.device)
    print(f"[eval_tf] TensorFlow device: {device_desc}")

    with tf.device(device_str):
        _, _, test_ds, _ = create_tf_datasets_from_clickhouse(cfg)

        model_path = cfg.output_dir / "models" / "ensemble_tf"
        model = tf.keras.models.load_model(model_path)

        results = model.evaluate(test_ds, return_dict=True)
        print("Test metrics:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
