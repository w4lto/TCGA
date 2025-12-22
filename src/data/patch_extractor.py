from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openslide  # type: ignore
import cv2
from skimage.color import rgb2hed  # type: ignore
from skimage.filters import threshold_otsu  # type: ignore
from skimage.morphology import remove_small_objects  # type: ignore


@dataclass(frozen=True)
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
    """
    Segmenta tecido (1) vs fundo (0) em thumbnail RGB simples.

    Heurística:
      - converte para HSV
      - usa canal V + Otsu
      - remove pequenos objetos
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("segment_tissue espera RGB [H,W,3].")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]

    # Guard-rail: se thumbnail for quase uniforme, Otsu pode ser instável
    if float(v.std()) < 1.0:
        # tudo tecido? não — aqui assumimos “quase branco” -> sem tecido
        return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    thr = threshold_otsu(v)
    mask = v < thr
    mask = remove_small_objects(mask, min_size=64)
    return mask.astype(np.uint8)


def compute_cellularity_score(patch_rgb: np.ndarray) -> float:
    """
    Score simples de celularidade via canal H (hematoxilina) em HED.

    Retorna proporção de "núcleos" estimados no patch.
    """
    if patch_rgb.ndim != 3 or patch_rgb.shape[2] != 3:
        return 0.0

    # Guard-rail: patch muito branco / sem variação
    if float(patch_rgb.std()) < 2.0:
        return 0.0

    hed = rgb2hed(patch_rgb)
    h_channel = hed[:, :, 0].astype(np.float32)

    # Guard-rail: se canal H for quase uniforme, Otsu é ruim
    if float(h_channel.std()) < 1e-3:
        return 0.0

    thr = threshold_otsu(h_channel)
    nuclei_mask = h_channel < thr
    return float(nuclei_mask.mean())


def _clamp_slice(a: int, b: int, max_b: int) -> Tuple[int, int]:
    a2 = max(0, a)
    b2 = min(max_b, b)
    if b2 <= a2:
        return 0, 0
    return a2, b2


def _tissue_score_from_thumb(
    tissue_mask_thumb: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    scale_x: float,
    scale_y: float,
) -> float:
    """
    Calcula tissue_score aproximado no thumbnail para a ROI do patch.
    """
    tx = int(x / scale_x)
    ty = int(y / scale_y)
    tw = max(1, int(patch_size / scale_x))
    th = max(1, int(patch_size / scale_y))

    h_thumb, w_thumb = tissue_mask_thumb.shape[:2]

    x0, x1 = _clamp_slice(tx, tx + tw, w_thumb)
    y0, y1 = _clamp_slice(ty, ty + th, h_thumb)
    if x1 == 0 and y1 == 0:
        return 0.0

    region = tissue_mask_thumb[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def extract_patches_from_wsi(
    slide_path: Path,
    out_dir: Path,
    patch_size: int = 256,
    stride: int = 256,
    tissue_threshold: float = 0.5,
    top_cellularity_quantile: float = 0.8,
    max_patches: Optional[int] = None,
    patient_id_hint: Optional[str] = None,
    level: int = 0,
    thumbnail_downsample: int = 16,
) -> List[PatchInfo]:
    """
    Extrai patches de um WSI e salva como PNG.
    """
    slide_path = Path(slide_path)
    if not slide_path.exists():
        raise RuntimeError(f"WSI não existe: {slide_path}")

    slide_id = slide_path.stem
    patient_id = patient_id_hint or slide_id

    slide = openslide.OpenSlide(str(slide_path))
    try:
        if level < 0 or level >= slide.level_count:
            raise ValueError(f"level inválido {level}. level_count={slide.level_count}")

        w, h = slide.level_dimensions[level]
        if w < patch_size or h < patch_size:
            return []

        # thumbnail para segmentação (a partir do level 0 "visual")
        thumb_w = max(1, w // thumbnail_downsample)
        thumb_h = max(1, h // thumbnail_downsample)

        thumb = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
        tissue_mask_thumb = segment_tissue(thumb)

        scale_x = w / float(thumb.shape[1])
        scale_y = h / float(thumb.shape[0])

        # Guardamos apenas metadados e score para seleção.
        # candidates: (cellularity, tissue_score, x, y)
        candidates: List[Tuple[float, float, int, int]] = []

        # 1o passe: coletar scores
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                tissue_score = _tissue_score_from_thumb(
                    tissue_mask_thumb=tissue_mask_thumb,
                    x=x,
                    y=y,
                    patch_size=patch_size,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                if tissue_score < tissue_threshold:
                    continue

                patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
                patch_np = np.array(patch)

                cellularity = compute_cellularity_score(patch_np)
                candidates.append((cellularity, tissue_score, x, y))

        if not candidates:
            return []

        # Seleção
        candidates.sort(key=lambda t: t[0], reverse=True)  # por cellularity desc

        if max_patches is not None and max_patches > 0:
            selected = candidates[: int(max_patches)]
        else:
            # Quantile cutoff
            scores = np.array([c[0] for c in candidates], dtype=np.float32)
            cutoff = float(np.quantile(scores, float(top_cellularity_quantile)))
            selected = [c for c in candidates if c[0] >= cutoff]

        if not selected:
            return []

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2o passe: salvar apenas os selecionados
        final_infos: List[PatchInfo] = []
        for i, (cellularity, tissue_score, x, y) in enumerate(selected):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            out_path = out_dir / f"{slide_id}_x{x}_y{y}_{i}.png"
            ok = cv2.imwrite(str(out_path), cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Falha ao salvar patch: {out_path}")

            final_infos.append(
                PatchInfo(
                    slide_id=slide_id,
                    patient_id=patient_id,
                    x=int(x),
                    y=int(y),
                    level=int(level),
                    image_path=out_path,
                    tissue_score=float(tissue_score),
                    cellularity=float(cellularity),
                )
            )

        return final_infos

    finally:
        slide.close()
