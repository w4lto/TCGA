from pathlib import Path

import numpy as np

from src.data.patch_extractor import segment_tissue, compute_cellularity_score


def test_segment_tissue_shape():
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    mask = segment_tissue(img)
    assert mask.shape == img.shape[:2]


def test_cellularity_range():
    img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    score = compute_cellularity_score(img)
    assert 0.0 <= score <= 1.0
