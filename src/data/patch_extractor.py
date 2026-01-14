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
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("segment_tissue espera RGB [H,W,3].")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]

    if float(v.std()) < 1.0:
        return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    thr = threshold_otsu(v)
    mask = v < thr
    mask = remove_small_objects(mask, min_size=64)
    return mask.astype(np.uint8)


def compute_cellularity_score(patch_rgb: np.ndarray) -> float:

    if patch_rgb.ndim != 3 or patch_rgb.shape[2] != 3:
        return 0.0

    if float(patch_rgb.std()) < 2.0:
        return 0.0

    hed = rgb2hed(patch_rgb)
    h_channel = hed[:, :, 0].astype(np.float32)

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


def _compute_spatial_diversity_score(
    candidate_coords: List[Tuple[int, int]],
    new_x: int,
    new_y: int,
    min_distance: int
) -> float:

    if not candidate_coords:
        return 1.0
    
    min_dist_found = float('inf')
    for ex_x, ex_y in candidate_coords:
        dist = np.sqrt((new_x - ex_x)**2 + (new_y - ex_y)**2)
        min_dist_found = min(min_dist_found, dist)
    
    if min_dist_found >= 2 * min_distance:
        return 1.0
    elif min_dist_found < min_distance:
        return 0.0
    else:
        return (min_dist_found - min_distance) / min_distance


def _select_diverse_patches(
    candidates: List[Tuple[float, float, int, int]],
    num_patches: int,
    patch_size: int,
    min_distance_multiplier: float = 2.0
) -> List[Tuple[float, float, int, int]]:

    if len(candidates) <= num_patches:
        return candidates
    
    candidates_sorted = sorted(candidates, key=lambda t: t[0], reverse=True)
    
    min_distance = patch_size * min_distance_multiplier
    
    selected = []
    selected_coords = []
    
    best = candidates_sorted[0]
    selected.append(best)
    selected_coords.append((best[2], best[3]))  # (x, y)
    
    remaining = candidates_sorted[1:]
    
    while len(selected) < num_patches and remaining:
        scored_candidates = []
        
        for candidate in remaining:
            cellularity, tissue_score, x, y = candidate
            
            max_cellularity = candidates_sorted[0][0]
            quality_score = cellularity / max_cellularity if max_cellularity > 0 else 0.0
            
            diversity_score = _compute_spatial_diversity_score(
                selected_coords, x, y, int(min_distance)
            )

            combined_score = 0.6 * quality_score + 0.4 * diversity_score
            
            scored_candidates.append((combined_score, candidate))
        
        # Selecionar melhor score combinado
        if scored_candidates:
            scored_candidates.sort(key=lambda t: t[0], reverse=True)
            _, next_patch = scored_candidates[0]
            
            selected.append(next_patch)
            selected_coords.append((next_patch[2], next_patch[3]))
            
            # Remover selecionado dos restantes
            remaining = [c for c in remaining if c != next_patch]
        else:
            break
    
    return selected


def extract_patches_from_wsi(
    slide_path: Path,
    out_dir: Path,
    patch_size: int = 256,
    stride: int = 256,
    tissue_threshold: float = 0.5,
    top_cellularity_quantile: float = 0.8,
    max_patches: Optional[int] = 3,
    patient_id_hint: Optional[str] = None,
    level: int = 0,
    thumbnail_downsample: int = 16,
    ensure_spatial_diversity: bool = True,
    min_distance_multiplier: float = 2.0,
) -> List[PatchInfo]:
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

        thumb_w = max(1, w // thumbnail_downsample)
        thumb_h = max(1, h // thumbnail_downsample)

        thumb = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
        tissue_mask_thumb = segment_tissue(thumb)

        scale_x = w / float(thumb.shape[1])
        scale_y = h / float(thumb.shape[0])


        candidates: List[Tuple[float, float, int, int]] = []

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
            print(f"AVISO: Nenhum patch com tissue suficiente em {slide_id}")
            return []

        scores = np.array([c[0] for c in candidates], dtype=np.float32)
        cutoff = float(np.quantile(scores, float(top_cellularity_quantile)))
        high_quality_candidates = [c for c in candidates if c[0] >= cutoff]
        
        if not high_quality_candidates:
            high_quality_candidates = candidates
        
        print(f"Slide {slide_id}: {len(candidates)} candidatos totais, "
              f"{len(high_quality_candidates)} de alta qualidade (top {int((1-top_cellularity_quantile)*100)}%)")

        if ensure_spatial_diversity and max_patches and max_patches > 1:
            selected = _select_diverse_patches(
                candidates=high_quality_candidates,
                num_patches=max_patches,
                patch_size=patch_size,
                min_distance_multiplier=min_distance_multiplier
            )
            print(f"Slide {slide_id}: {len(selected)} patches selecionados com diversidade espacial")
        else:
            high_quality_candidates.sort(key=lambda t: t[0], reverse=True)
            if max_patches is not None and max_patches > 0:
                selected = high_quality_candidates[:int(max_patches)]
            else:
                selected = high_quality_candidates
            print(f"Slide {slide_id}: {len(selected)} patches selecionados (sem diversidade)")

        if not selected:
            return []

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        final_infos: List[PatchInfo] = []
        for i, (cellularity, tissue_score, x, y) in enumerate(selected):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            out_path = out_dir / f"{slide_id}_patch{i+1}_x{x}_y{y}_cell{int(cellularity*100)}.png"
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
            
            print(f"  Patch {i+1}/{len(selected)}: "
                  f"cellularity={cellularity:.3f}, tissue={tissue_score:.3f}, "
                  f"coords=({x}, {y})")

        return final_infos

    finally:
        slide.close()


def extract_patches_batch(
    slide_paths: List[Path],
    out_dir: Path,
    patches_per_slide: int = 3,
    **kwargs
) -> dict:
    stats = {
        'total_slides': len(slide_paths),
        'successful_slides': 0,
        'total_patches': 0,
        'cellularities': [],
        'failed_slides': []
    }
    
    for slide_path in slide_paths:
        try:
            patches = extract_patches_from_wsi(
                slide_path=slide_path,
                out_dir=out_dir,
                max_patches=patches_per_slide,
                ensure_spatial_diversity=True,
                **kwargs
            )
            
            if patches:
                stats['successful_slides'] += 1
                stats['total_patches'] += len(patches)
                stats['cellularities'].extend([p.cellularity for p in patches])
            else:
                stats['failed_slides'].append(str(slide_path))
                
        except Exception as e:
            print(f"ERRO ao processar {slide_path}: {e}")
            stats['failed_slides'].append(str(slide_path))
    
    if stats['cellularities']:
        stats['avg_cellularity'] = float(np.mean(stats['cellularities']))
    else:
        stats['avg_cellularity'] = 0.0
    
    return stats