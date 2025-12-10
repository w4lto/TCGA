import tensorflow as tf
from tensorflow.keras import layers, models


def build_resnet_backbone_tf(
    num_classes: int,
    dropout: float = 0.3,
) -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, 3),
        pooling="avg",
    )

    inputs = tf.keras.Input(shape=(None, None, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes)(x)  # logits
    return models.Model(inputs=inputs, outputs=outputs, name="resnet_tf")
