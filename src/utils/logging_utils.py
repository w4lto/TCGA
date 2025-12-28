import logging
from pathlib import Path
import os
from src.utils.config_utils import load_config

CONFIG_FILE_PATH = [
    "/app/config.yaml",
    "../../config.yaml"
]

def resolve_config_path() -> Path:
    env_path = os.getenv("APP_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"APP_CONFIG set but config.yaml file not found")
    for c in CONFIG_FILE_PATH:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError("Config.yaml file not found")

def setup_logging(log_dir: Path, name: str = "pipeline") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    cfg = load_config(str(resolve_config_path()))
    
    logger.setLevel(cfg.log_level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

logger = setup_logging(Path("logs"), "train_tf")