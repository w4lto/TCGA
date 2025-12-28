from __future__ import annotations

from typing import List, Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_ensemble_model_tf(
    backbones: Sequence[keras.Model],
    num_classes: int,
    strategy: str = "mean_logits",
    name: str = "ensemble",
) -> keras.Model:
    """
    Constrói um ensemble Keras-Functional compatível com Keras 3.

    Estratégias:
      - mean_logits: média dos logits (recomendado como baseline)
      - mean_probs: média das probabilidades (softmax/sigmoid e depois média)
      - stacking_mlp: concatena logits e aplica um pequeno MLP (gancho para stacking)

    Observação:
      - Todos os backbones devem retornar logits no shape [B, num_classes].
      - Para binário (num_classes=1), o modelo retorna logits [B, 1].
    """
    if not backbones:
        raise ValueError("backbones vazio")

    # garante que todos são models Keras
    for m in backbones:
        if not isinstance(m, keras.Model):
            raise TypeError(f"backbone não é keras.Model: {type(m)}")

    inp = keras.Input(shape=(None, None, 3), name="image")  # H,W,3 (redimensionado no dataset)

    logits_list: List[tf.Tensor] = []
    for i, m in enumerate(backbones):
        # Força nome estável
        x = m(inp)
        # garante rank 2 (B,C)
        x = layers.Lambda(lambda t: t, name=f"{m.name}_logits")(x)
        logits_list.append(x)

    if strategy == "mean_logits":
        # Média diretamente em logits via Keras layers (sem tf.stack)
        if len(logits_list) == 1:
            out = logits_list[0]
        else:
            out = layers.Average(name="mean_logits")(logits_list)

        model = keras.Model(inputs=inp, outputs=out, name=name)
        return model

    if strategy == "mean_probs":
        # Converte para prob e faz média; retorno é prob (não logits).
        probs_list: List[tf.Tensor] = []
        for j, lg in enumerate(logits_list):
            if num_classes == 1:
                p = layers.Activation("sigmoid", name=f"sigmoid_{j}")(lg)
            else:
                p = layers.Activation("softmax", name=f"softmax_{j}")(lg)
            probs_list.append(p)

        if len(probs_list) == 1:
            out = probs_list[0]
        else:
            out = layers.Average(name="mean_probs")(probs_list)

        model = keras.Model(inputs=inp, outputs=out, name=name)
        return model

    if strategy == "stacking_mlp":
        # Concatena logits e aplica MLP pequeno.
        if len(logits_list) == 1:
            concat = logits_list[0]
        else:
            concat = layers.Concatenate(name="concat_logits")(logits_list)

        x = layers.Dense(128, activation="relu", name="stack_dense_1")(concat)
        x = layers.Dropout(0.2, name="stack_dropout")(x)

        if num_classes == 1:
            out = layers.Dense(1, name="stack_logits")(x)
        else:
            out = layers.Dense(num_classes, name="stack_logits")(x)

        model = keras.Model(inputs=inp, outputs=out, name=name)
        return model

    raise ValueError(f"strategy inválida: {strategy}")
