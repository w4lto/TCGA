from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import openslide  # type: ignore
import cv2
from skimage.color import rgb2hed  # type: ignore
from skimage.filters import threshold_otsu  # type: ignore
from skimage.morphology import remove_small_objects  # type: ignore


@dataclass
class PatchInfo:
    slide_id: str
    patient_id: str
    x: int
    y: int
    level: int
    image_path: Path
    tissue_score: float
    cellularity: float


def segment_tissue(rgb: np.ndarray) -> np.ndarray:
    """Segmenta tecido (1) vs fundo (0) em thumbnail RGB simples."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    thr = threshold_otsu(v)
    mask = v < thr
    mask = remove_small_objects(mask, min_size=64)
    return mask.astype(np.uint8)


def compute_cellularity_score(patch_rgb: np.ndarray) -> float:
    """Score simples de celularidade via canal H (hematoxilina) em HED."""
    hed = rgb2hed(patch_rgb)
    h_channel = hed[:, :, 0]
    thr = threshold_otsu(h_channel)
    nuclei_mask = h_channel < thr
    return float(nuclei_mask.mean())


def extract_patches_from_wsi(
    wsi_path: Path,
    output_dir: Path,
    patch_size: int = 256,
    stride: int = 256,
    tissue_threshold: float = 0.5,
    top_cellularity_quantile: float = 0.8,
    max_patches: int | None = None,
    patient_id_hint: str | None = None,
) -> List[PatchInfo]:
    slide = openslide.OpenSlide(str(wsi_path))
    level = 0
    w, h = slide.level_dimensions[level]

    thumb = np.array(slide.get_thumbnail((w // 16, h // 16)).convert("RGB"))
    tissue_mask_thumb = segment_tissue(thumb)
    scale_x = w / thumb.shape[1]
    scale_y = h / thumb.shape[0]

    patch_infos: List[PatchInfo] = []
    patch_scores: List[float] = []
    temp_patches: List[np.ndarray] = []

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            tx = int(x / scale_x)
            ty = int(y / scale_y)
            tw = int(patch_size / scale_x)
            th = int(patch_size / scale_y)

            tissue_region = tissue_mask_thumb[ty : ty + th, tx : tx + tw]
            tissue_score = float(tissue_region.mean())
            if tissue_score < tissue_threshold:
                continue

            patch = slide.read_region(
                (x, y), level, (patch_size, patch_size)
            ).convert("RGB")
            patch_np = np.array(patch)
            score = compute_cellularity_score(patch_np)

            temp_patches.append(patch_np)
            patch_scores.append(score)
            slide_id = wsi_path.stem
            patch_infos.append(
                PatchInfo(
                    slide_id=slide_id,
                    patient_id=patient_id_hint or slide_id,
                    x=x,
                    y=y,
                    level=level,
                    image_path=output_dir,  # atualizado depois
                    tissue_score=tissue_score,
                    cellularity=score,
                )
            )

    slide.close()

    if not patch_scores:
        return []

    scores = np.array(patch_scores)
    cutoff = np.quantile(scores, top_cellularity_quantile)
    indices = np.where(scores >= cutoff)[0].tolist()

    if max_patches is not None and len(indices) > max_patches:
        indices = indices[:max_patches]

    output_dir.mkdir(parents=True, exist_ok=True)
    final_infos: List[PatchInfo] = []
    for i, idx in enumerate(indices):
        patch_np = temp_patches[idx]
        info = patch_infos[idx]
        out_path = output_dir / f"{info.slide_id}_x{info.x}_y{info.y}_{i}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))
        final_infos.append(
            PatchInfo(
                slide_id=info.slide_id,
                patient_id=info.patient_id,
                x=info.x,
                y=info.y,
                level=info.level,
                image_path=out_path,
                tissue_score=info.tissue_score,
                cellularity=info.cellularity,
            )
        )

    return final_infos
