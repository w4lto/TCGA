# src/utils/tf_device_utils.py

from __future__ import annotations

from typing import Tuple, List

import tensorflow as tf


def setup_tf_device(preferred: str = "gpu") -> Tuple[str, str]:
    """
    Configura o dispositivo do TensorFlow com:
    - memory growth ativado em todas as GPUs visíveis
    - fallback para CPU quando não há GPU
    Retorna:
        (device_str, descricao_humana)
        ex: ("/GPU:0", "GPU(s) detectadas: NVIDIA GPU 0, logical_gpus=1")
    """
    # Lista GPUs físicas
    gpus: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices("GPU")

    if gpus:
        # Ativa memory growth em todas as GPUs
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Se já estiver configurado ou não suportado, ignoramos
                pass

        logical_gpus = tf.config.list_logical_devices("GPU")
        device_str = "/GPU:0"

        gpu_names = [getattr(g, "name", "GPU") for g in gpus]
        desc = (
            f"GPU(s) detectadas: {', '.join(gpu_names)}; "
            f"logical_gpus={len(logical_gpus)}; usando {device_str}"
        )
        return device_str, desc

    # Sem GPU -> fallback CPU
    cpus: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices("CPU")
    device_str = "/CPU:0"
    cpu_names = [getattr(c, "name", "CPU") for c in cpus]
    desc = f"Nenhuma GPU detectada. CPU(s): {', '.join(cpu_names)}; usando {device_str}"
    return device_str, desc
