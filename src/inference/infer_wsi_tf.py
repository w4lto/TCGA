from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from src.utils.config_utils import load_config
from src.utils.tf_device_utils import setup_tf_device
from src.data.patch_extractor import extract_patches_from_wsi
from src.data.tf_dataset import make_tf_dataset, build_label_mapping


def main(slide_path: str, config_path: str, output_dir: str) -> None:
    cfg = load_config(config_path)

    device_str, device_desc = setup_tf_device(preferred=cfg.device)
    print(f"[infer_wsi_tf] TensorFlow device: {device_desc}")

    slide_path_path = Path(slide_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patches_root = out_dir / "patches_tmp"
    patch_infos = extract_patches_from_wsi(
        slide_path_path,
        patches_root,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        tissue_threshold=0.5,
        top_cellularity_quantile=0.8,
        max_patches=cfg.max_patches_per_wsi,
        patient_id_hint=slide_path_path.stem,
    )

    if not patch_infos:
        print("Nenhum patch válido extraído.")
        return

    rows: List[dict] = [
        {
            "patient_id": p.patient_id,
            "slide_id": p.slide_id,
            "image_path": str(p.image_path),
            # rótulo dummy apenas para compatibilizar com make_tf_dataset
            "label": cfg.class_names[0],
        }
        for p in patch_infos
    ]
    df = pd.DataFrame(rows)

    label_map = build_label_mapping(cfg.class_names)

    with tf.device(device_str):
        ds = make_tf_dataset(
            df,
            label_map,
            cfg.class_names,
            batch_size=cfg.batch_size,
            shuffle=False,
            patch_size=cfg.patch_size,
        )

        model_path = cfg.output_dir / "models" / "ensemble_tf"
        model = keras.models.load_model(model_path)

        all_probas: List[np.ndarray] = []
        for batch_x, _ in ds:
            logits = model(batch_x, training=False)
            probs = tf.nn.softmax(logits, axis=-1)
            all_probas.append(np.asarray(probs))

        all_probas_np = np.concatenate(all_probas, axis=0)
        mean_proba = all_probas_np.mean(axis=0)
        pred_idx = int(mean_proba.argmax())
        pred_label = cfg.class_names[pred_idx]

        print("Probabilidades por estágio (slide-level, mean_prob):")
        for cname, prob in zip(cfg.class_names, mean_proba.tolist()):
            print(f"  {cname}: {prob:.4f}")

        print("Estágio predito (slide-level):", pred_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/infer_wsi_tf")
    args = parser.parse_args()
    main(args.slide_path, args.config, args.output_dir)
