from pathlib import Path
from typing import Dict, Any

import mlflow
import mlflow.tensorflow  # type: ignore


def init_mlflow(tracking_uri: Path, experiment_name: str) -> None:
    tracking_uri = tracking_uri.resolve()
    tracking_uri.mkdir(parents=True, exist_ok=True)
    mlflow_utils.set_tracking_uri(tracking_uri.as_uri())
    mlflow_utils.set_experiment(experiment_name)


def start_run(run_name: str, params: Dict[str, Any]):
    run = mlflow_utils.start_run(run_name=run_name)
    mlflow_utils.log_params(params)
    return run


def log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
    mlflow_utils.log_metrics(metrics, step=step)
