import tensorflow as tf

from src.models.kimianet_backbone_tf import build_kimianet_backbone_tf
from src.models.resnet_backbone_tf import build_resnet_backbone_tf
from src.models.ensemble_tf import build_ensemble_model_tf


def test_tf_models_forward():
    x = tf.random.normal((2, 224, 224, 3))
    k = build_kimianet_backbone_tf(num_classes=4, weights_path=None, dropout=0.0)
    r = build_resnet_backbone_tf(num_classes=4, dropout=0.0)
    e = build_ensemble_model_tf([k, r], num_classes=4)

    out_k = k(x, training=False)
    out_r = r(x, training=False)
    out_e = e(x, training=False)

    assert out_k.shape == (2, 4)
    assert out_r.shape == (2, 4)
    assert out_e.shape == (2, 4)
