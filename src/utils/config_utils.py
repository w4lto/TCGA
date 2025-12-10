from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml


@dataclass
class ClickHouseConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    table_slides: str
    table_patches: str
    limit_per_split: Optional[int] = None


@dataclass
class TrainConfig:
    # Dados
    data_root: Path

    # Tarefa
    task_type: str
    class_names: List[str]
    positive_classes: Optional[List[str]] = None

    # Pesos clínicos para estágio
    stage_clinical_weights: Optional[Dict[str, float]] = None
    dynamic_class_weights: bool = False

    # Patches / DataLoader
    patch_size: int = 256
    stride: int = 256
    batch_size: int = 16
    num_workers: int = 4
    max_patches_per_wsi: Optional[int] = None

    # Splits
    val_split: float = 0.1
    test_split: float = 0.1
    num_folds: int = 1
    fold_index: int = 0

    # Modelo
    backbone_kimianet_weights: Optional[Path] = None
    use_resnet: bool = True
    dropout: float = 0.3

    # Treino
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10

    # MLOps
    experiment_name: str = "breast_histopathology_tf"
    run_name: str = "kimianet_resnet_ensemble_tf"
    mlflow_tracking_uri: Path = Path("mlruns")
    output_dir: Path = Path("experiments")

    # Misc
    seed: int = 42
    device: str = "gpu"  # "gpu" ou "cpu"

    # ClickHouse
    clickhouse: ClickHouseConfig | None = None


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    with path.open("r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    raw["data_root"] = Path(raw["data_root"])

    ch_raw = raw.pop("clickhouse", None)
    ch_cfg = None
    if ch_raw is not None:
        ch_cfg = ClickHouseConfig(**ch_raw)
    raw["clickhouse"] = ch_cfg

    if raw.get("backbone_kimianet_weights"):
        raw["backbone_kimianet_weights"] = Path(raw["backbone_kimianet_weights"])

    raw["mlflow_tracking_uri"] = Path(raw.get("mlflow_tracking_uri", "mlruns"))
    raw["output_dir"] = Path(raw.get("output_dir", "experiments"))

    return TrainConfig(**raw)
