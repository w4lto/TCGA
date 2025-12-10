from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models


def build_kimianet_backbone_tf(
    num_classes: int,
    weights_path: Optional[Path] = None,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """
    Aproximação de KimiaNet usando DenseNet121 do Keras.
    Se weights_path for fornecido, espera-se um .h5 compatível.
    """
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=None,  # ou "imagenet" se quiser inicializar de ImageNet
        input_shape=(None, None, 3),
        pooling="avg",
    )

    inputs = tf.keras.Input(shape=(None, None, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes)(x)  # logits

    model = models.Model(inputs=inputs, outputs=outputs, name="kimianet_tf")

    if weights_path is not None and weights_path.exists():
        model.load_weights(str(weights_path))

    return model
