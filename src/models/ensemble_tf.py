import tensorflow as tf
from tensorflow.keras import models


def build_ensemble_model_tf(
    backbones: list[tf.keras.Model],
    num_classes: int,
    combine: str = "mean_logits", 
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, None, 3))
    logits_list = [m(inputs, training=False) for m in backbones]
    stacked = tf.stack(logits_list, axis=0)

    if combine == "mean_logits":
        logits = tf.reduce_mean(stacked, axis=0)
    else:
        probs = tf.nn.softmax(stacked, axis=-1)
        probs_mean = tf.reduce_mean(probs, axis=0)
        logits = tf.math.log(probs_mean + 1e-8)

    return models.Model(inputs=inputs, outputs=logits, name="ensemble_tf")
