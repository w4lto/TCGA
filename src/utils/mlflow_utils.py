from pathlib import Path
from typing import Dict, Any

import mlflow


def init_mlflow(tracking_uri: Path, experiment_name: str) -> None:
    tracking_uri = tracking_uri.resolve()
    tracking_uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri.as_uri())
    mlflow.set_experiment(experiment_name)


def start_run(run_name: str, params: Dict[str, Any]):
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    return run


def log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)
