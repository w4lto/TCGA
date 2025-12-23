from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.utils.config_utils import load_config, TrainConfig
from src.utils.logging_utils import logger
from src.utils.mlflow_utils import init_mlflow, start_run, log_metrics
from src.utils.tf_device_utils import setup_tf_device

from src.data.tf_dataset import (
    create_tf_datasets_from_clickhouse
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencia avisos do TensorFlow (CUDA/XLA)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def compute_class_weights_dynamic(cfg: TrainConfig, label_map: Dict[str, int]):
    """
    Calcula pesos de classe dinâmicos baseados na frequência no banco de dados.
    """
    assert cfg.clickhouse is not None
    ch = ClickHouseClient(cfg.clickhouse)
    
    # Query otimizada para contar apenas o necessário
    query = f"SELECT stage_label, count(*) as cnt FROM {cfg.clickhouse.table_patches} WHERE split = 'train' GROUP BY stage_label"
    df = ch.query_df(query)
    
    # Converte para dict {label: count}
    counts = dict(zip(df["stage_label"], df["cnt"]))
    logger.info(f"Base stage count: {counts}")

    num_classes = len(cfg.class_names)
    freqs = np.zeros(num_classes, dtype=float)
    
    # Mapeia contagens para índices inteiros
    for name, index in label_map.items():
        freqs[index] = counts.get(name, 0)

    # Evita divisão por zero
    freqs[freqs == 0] = 1.0 
    
    # Inverso da frequência (balanceamento)
    inv = 1.0 / freqs
    inv = inv / inv.sum()

    # Boost clínico opcional
    if cfg.stage_clinical_weights:
        boost = np.array(
            [cfg.stage_clinical_weights.get(name, 1.0) for name in cfg.class_names],
            dtype=float,
        )
        weights = inv * boost
    else:
        weights = inv

    # Normaliza para que a soma seja 1 (ou escala conforme preferência)
    weights = weights / weights.sum()
    
    return {int(i): float(w) for i, w in enumerate(weights)}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    logger.info(f"Loaded config: {cfg}")
    set_seed(cfg.seed)

    # Configura Hardware
    device_str, device_desc = setup_tf_device()
    logger.info(f"TensorFlow device: {device_desc}")

    # Inicializa MLflow
    tracking_uri = Path(cfg.mlflow_tracking_uri)
    init_mlflow(tracking_uri, cfg.experiment_name)
    
    params: Dict[str, Any] = {
        "task_type": cfg.task_type,
        "class_names": ",".join(cfg.class_names),
        "batch_size": cfg.batch_size,
        "patch_size": cfg.patch_size,
        "model_type": "Ensemble (KimiaNet + ResNet)" if cfg.use_resnet else "KimiaNet",
        "learning_rate": cfg.learning_rate,
        "early_stopping_patience": cfg.early_stopping_patience,
    }
    start_run(cfg.run_name, params=params)

    with tf.device(device_str):
        train_ds, val_ds, test_ds, label_map = create_tf_datasets_from_clickhouse(cfg)
        num_classes = len(cfg.class_names)

        backbones: List[keras.Model] = []
        
        backbone_path = Path(cfg.backbone_kimianet_weights)
        kimianet = build_kimianet_backbone_tf(
            num_classes=num_classes,
            weights_path=backbone_path,
            dropout=cfg.dropout,
        )
        backbones.append(kimianet)

        if cfg.use_resnet:
            resnet = build_resnet_backbone_tf(
                num_classes=num_classes,
                dropout=cfg.dropout,
            )
            backbones.append(resnet)

        # Ensemble
        model = build_ensemble_model_tf(backbones, num_classes=num_classes)

        # ATENÇÃO: Se o Ensemble faz a média de Softmax (probabilidades), from_logits deve ser False.
        # Se faz média de outputs lineares, deve ser True. Assumindo Probabilidades aqui:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        model.summary(print_fn=logger.info)

        class_weight = None
        if cfg.dynamic_class_weights:
            class_weight = compute_class_weights_dynamic(cfg, label_map)
            logger.info(f"Class weights calculados: {class_weight}")

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=Path("logs"), update_freq='batch')
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True, # Isso restaura os pesos da MELHOR época
                verbose=1
            )
        ]

        logger.info("Iniciando treinamento...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.num_epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )

        # Como restore_best_weights=True, o modelo atual tem os pesos da melhor época.
        # Porém, history contém todas as épocas até a paciência acabar.
        # Precisamos achar o índice da melhor época baseada na métrica monitorada (val_loss).
        
        val_loss_history = history.history["val_loss"]
        best_epoch_idx = val_loss_history.index(min(val_loss_history))
        logger.info(f"Melhor época identificada: {best_epoch_idx + 1}")

        metrics_to_log = {}
        # Extrai métricas do histórico no índice correto
        for k, v_list in history.history.items():
            metric_type = "val" if k.startswith("val_") else "train"
            clean_k = k.replace("val_", "")
            metrics_to_log[f"{metric_type}_{clean_k}"] = float(v_list[best_epoch_idx])
        
        log_metrics(metrics_to_log, step=best_epoch_idx)

        # Avaliação no Test Set (Básico)
        logger.info("Avaliando test_ds (métricas básicas)...")
        test_metrics = model.evaluate(test_ds, return_dict=True)
        log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})

        # Métricas Avançadas e Slide-Level
        logger.info("Gerando predições para métricas avançadas...")
        
        # Predições (Isso respeita a ordem do dataset)
        y_proba = model.predict(test_ds, verbose=1)
        
        # Extração de Labels Verdadeiros do Dataset
        # O dataset test_ds retorna (imagem, label). Precisamos extrair apenas os labels.
        # ATENÇÃO: O test_ds NÃO deve estar com shuffle=True para garantir alinhamento
        logger.info("Extraindo labels verdadeiros do dataset...")
        y_true_batches = []
        for _, batch_labels in test_ds:
            y_true_batches.append(batch_labels.numpy())
        
        y_true = np.concatenate(y_true_batches, axis=0)
        
        if len(y_true) != len(y_proba):
            raise RuntimeError(f"Desalinhamento! y_true tem {len(y_true)} amostras, mas y_proba tem {len(y_proba)}.")
        
        # Patch-level Metrics
        logger.info("Calculando métricas Patch-Level...")
        patch_metrics = compute_patch_metrics_multiclass(
            y_true=y_true,
            y_proba=y_proba,
            class_names=cfg.class_names,
        )
        log_metrics({f"patch_{k}": v for k, v in patch_metrics.items()})

        # Recuperação de Slide IDs para Métricas de Slide
        # tf.data.Dataset perde o slide_id. Precisamos buscar do banco.
        if cfg.clickhouse:
            logger.info("Consultando metadados de teste para agregação por slide...")
            ch = ClickHouseClient(cfg.clickhouse)
            # É crucial que esta query retorne os dados NA MESMA ORDEM que o create_tf_datasets gerou
            # Se create_tf_datasets usa ordem aleatória no teste, isso aqui vai falhar.
            # Idealmente, o dataset de teste deve ter shuffle=False e uma ordenação determinística.
            df_test_meta = ch.query_df(
                f"SELECT slide_id, stage_label FROM {cfg.clickhouse.table_patches} WHERE split = 'test'"
                # f" ORDER BY patch_id" # Recomendado se houver coluna de ID único
            )
            
            if len(df_test_meta) == len(y_true):
                slide_ids = df_test_meta["slide_id"].to_numpy()
                
                # Slide-level (Mean Prob)
                y_true_slide, y_proba_slide = aggregate_by_slide(
                    y_true=y_true,
                    y_proba=y_proba,
                    slide_ids=slide_ids,
                    method="mean_prob",
                )
                slide_metrics = compute_slide_metrics_multiclass(
                    y_true_slide, y_proba_slide, cfg.class_names
                )
                log_metrics({f"slide_soft_{k}": v for k, v in slide_metrics.items()})
                logger.info("Métricas de slide calculadas com sucesso.")
            else:
                logger.warning(
                    f"Tamanho do dataset de teste ({len(y_true)}) difere do metadata do banco ({len(df_test_meta)}). "
                    "Pulando métricas de slide para evitar desalinhamento."
                )
        
        # Salvar Modelo
        out_dir = cfg.output_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "ensemble_model.keras" # Formato .keras é preferido no TF > 2.10
        model.save(save_path)
        logger.info(f"Modelo salvo em: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)