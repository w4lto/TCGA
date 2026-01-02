from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ClickHouseConfig:
    host: str = "clickhouse"
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "ml_histopath"
    table_slides: str = "tcga_slides"
    table_patches: str = "tcga_patches"
    limit_per_split: int = None
    


@dataclass
class TrainConfig:
    #Run
    run_name: str

    # Log
    log_level: str
    
    stage_clinical_weights:dict[str,float] = None
    
    # Reprodutibilidade
    seed: int = 42

    # Splits
    val_split: float = 0.1
    test_split: float = 0.1

    # Dataset / patches
    patch_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    prefetch: int = 2

    # Treino
    num_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    dynamic_class_weights = True
    backbone_kimianet_weights:str = "models/KimiaNetKerasWeights.h5"
    class_names:list[str] = field(default_factory=["I", "II", "III", "IV"])
    stride: int = 256
    dropout: float = 0.3
    use_resnet:bool = True

    # Task
    num_classes: int = 4  # I/II/III/IV (ou 2 no binário)
    label_mode: str = "stage"  # "stage" | "binary"
    task_type: str = "stage_multiclass"

    # Agregação por slide (PATCH -> SLIDE)
    # - "mean_prob": média das probabilidades por classe e argmax
    # - "majority_vote": maioria do argmax por patch (com desempate por mean_prob)
    slide_aggregation: str = "majority_vote"

    # Paths / outputs
    output_dir: str = "/app/experiments"
    mlflow_tracking_uri: str = "file:/app/mlruns"
    experiment_name: str = "tcga_brca"

    # ClickHouse
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)
    
    data_root: str = "data"
    
    image_size: int = 256
    
    training_strategy: str = "progressive"


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def _filter_kwargs(dc_type: Any, raw: Dict[str, Any], ctx: str) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dc_type)}
    extras = sorted([k for k in raw.keys() if k not in allowed])
    if extras:
        logger.warning("Config: chaves ignoradas para %s: %s", ctx, ", ".join(extras))
    return {k: v for k, v in raw.items() if k in allowed}


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _expand_env(raw)

    ch_raw = raw.get("clickhouse", {}) or {}
    ch_kwargs = _filter_kwargs(ClickHouseConfig, ch_raw, ctx="ClickHouseConfig")
    ch_cfg = ClickHouseConfig(**ch_kwargs)

    # TrainConfig (top-level)
    train_kwargs = _filter_kwargs(TrainConfig, raw, ctx="TrainConfig")
    train_kwargs["clickhouse"] = ch_cfg
    cfg = TrainConfig(**train_kwargs)

    # Validação leve
    valid_aggs = {"mean_prob", "majority_vote"}
    if cfg.slide_aggregation not in valid_aggs:
        raise ValueError(
            f"slide_aggregation inválido: {cfg.slide_aggregation}. Use um de: {sorted(valid_aggs)}"
        )

    valid_label_modes = {"stage", "binary"}
    if cfg.label_mode not in valid_label_modes:
        raise ValueError(
            f"label_mode inválido: {cfg.label_mode}. Use um de: {sorted(valid_label_modes)}"
        )

    return cfg
