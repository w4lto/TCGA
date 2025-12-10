import pandas as pd

from src.data.tf_dataset import make_tf_dataset, build_label_mapping


def test_tf_dataset_basic(tmp_path):
    from PIL import Image
    import numpy as np
    import tensorflow as tf

    img_path = tmp_path / "img.png"
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img_path)

    df = pd.DataFrame(
        [
            dict(
                patient_id="P1",
                slide_id="S1",
                image_path=str(img_path),
                label="I",
                dataset_source="SYN",
            )
        ]
    )

    label_map = build_label_mapping(["I", "II", "III", "IV"])
    ds = make_tf_dataset(
        df,
        label_map,
        ["I", "II", "III", "IV"],
        batch_size=1,
        shuffle=False,
        patch_size=32,
    )
    x, y = next(iter(ds))
    assert x.shape[1:] == (32, 32, 3)
    assert int(y.numpy()[0]) == 0
