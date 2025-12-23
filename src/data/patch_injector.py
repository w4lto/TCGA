from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

from src.utils.config_utils import load_config, TrainConfig
from src.data.ch_utils import ClickHouseClient
from src.data.patch_extractor import extract_patches_from_wsi


def _full_table_name(cfg: TrainConfig, table: str) -> str:
    db = cfg.clickhouse.database
    if "." in table:
        return table
    return f"{db}.{table}"


def _ensure_patches_table(cfg: TrainConfig, client) -> str:
    """
    Garante que a tabela de patches existe. Mantém schema simples.
    """
    full_table = _full_table_name(cfg, cfg.clickhouse.table_patches)

    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {full_table} (
          patient_id String,
          slide_id String,
          patch_id String,
          patch_path String,
          x Int32,
          y Int32,
          level Int32,
          patch_size Int32,
          tissue_score Float32,
          cellularity Float32,
          stage_label String,
          dataset_source String,
          split String
        )
        ENGINE = MergeTree
        ORDER BY (dataset_source, stage_label, patient_id, slide_id, patch_id)
        """
    )
    return full_table


def _insert_batch(client, full_table: str, rows: Sequence[Tuple]) -> None:
    if not rows:
        return

    client.insert(
        full_table,
        rows,
        column_names=[
            "patient_id",
            "slide_id",
            "patch_id",
            "patch_path",
            "x",
            "y",
            "level",
            "patch_size",
            "tissue_score",
            "cellularity",
            "stage_label",
            "dataset_source",
            "split",
        ],
    )


def main(
    config_path: str,
    out_root: Optional[str],
    patch_size: Optional[int],
    stride: Optional[int],
    max_patches_per_wsi: Optional[int],
    tissue_threshold: float,
    top_cellularity_quantile: float,
    limit_slides: Optional[int],
    truncate_patches: bool,
) -> None:
    cfg = load_config(config_path)

    patch_size = int(patch_size or cfg.patch_size)
    stride = int(stride or patch_size)  # default: não-overlap
    max_patches_per_wsi = int(max_patches_per_wsi or 500)  # default para sanity check

    if out_root:
        patches_root = Path(out_root)
    else:
        patches_root = Path("/app/data/processed/patches_tcga")

    patches_root.mkdir(parents=True, exist_ok=True)

    ch_client = ClickHouseClient(cfg.clickhouse).client

    slides_table = _full_table_name(cfg, cfg.clickhouse.table_slides)
    patches_table = _ensure_patches_table(cfg, ch_client)

    if truncate_patches:
        ch_client.command(f"TRUNCATE TABLE {patches_table}")

    q = f"""
    SELECT patient_id, slide_id, image_path, stage_label, split
    FROM {slides_table}
    ORDER BY stage_label, patient_id, slide_id
    """
    if limit_slides and limit_slides > 0:
        q += f"\nLIMIT {int(limit_slides)}"

    df_slides = ch_client.query_df(q)

    if df_slides.empty:
        raise RuntimeError(
            f"Nenhum slide encontrado em {slides_table}. "
            f"Você já rodou o ingestion_tcga?"
        )

    batch_rows: List[Tuple] = []

    for _, row in df_slides.iterrows():
        slide_path = Path(row["image_path"])
        patient_id = str(row["patient_id"])
        slide_id = str(row["slide_id"])
        stage_label = str(row["stage_label"])
        split = str(row["split"])

        if not slide_path.exists():
            # falha cedo para evitar gerar metadado inválido
            raise RuntimeError(f"WSI não encontrado no filesystem do container: {slide_path}")

        out_dir = patches_root / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        patch_infos = extract_patches_from_wsi(
            slide_path=slide_path,
            out_dir=out_dir,
            patch_size=patch_size,
            stride=stride,
            tissue_threshold=tissue_threshold,
            top_cellularity_quantile=top_cellularity_quantile,
            max_patches=max_patches_per_wsi,
            patient_id_hint=patient_id,
        )

        for i, info in enumerate(patch_infos):
            patch_id = f"{slide_id}_x{info.x}_y{info.y}_{i}"
            batch_rows.append(
                (
                    patient_id,
                    slide_id,
                    patch_id,
                    str(Path(info.image_path).resolve()),
                    int(info.x),
                    int(info.y),
                    int(info.level),
                    int(patch_size),
                    float(info.tissue_score),
                    float(info.cellularity),
                    stage_label,
                    "TCGA_BRCA_PATCH",
                    split,
                )
            )

        if len(batch_rows) >= 10_000:
            _insert_batch(ch_client, patches_table, batch_rows)
            batch_rows.clear()

        print(
            f"[patch_injector] slide={slide_id} patient={patient_id} "
            f"stage={stage_label} split={split} patches={len(patch_infos)}"
        )

    if batch_rows:
        _insert_batch(ch_client, patches_table, batch_rows)

    print(f"[patch_injector] Patches inseridos em {patches_table}")
    print(f"[patch_injector] Patches salvos em: {patches_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--out_root", type=str, default=None, help="Diretório raiz para salvar patches (default: /app/data/processed/patches_tcga)")
    parser.add_argument("--patch_size", type=int, default=None, help="Override do patch_size (default: cfg.patch_size)")
    parser.add_argument("--stride", type=int, default=None, help="Stride (default: patch_size)")
    parser.add_argument("--max_patches_per_wsi", type=int, default=None, help="Máximo de patches por WSI (default: 500)")

    parser.add_argument("--tissue_threshold", type=float, default=0.5)
    parser.add_argument("--top_cellularity_quantile", type=float, default=0.8)

    parser.add_argument("--limit_slides", type=int, default=None, help="Limita número de slides para validação rápida")
    parser.add_argument("--truncate_patches", action="store_true", help="TRUNCATE na tabela de patches antes de inserir")

    args = parser.parse_args()

    main(
        config_path=args.config,
        out_root=args.out_root,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches_per_wsi=args.max_patches_per_wsi,
        tissue_threshold=args.tissue_threshold,
        top_cellularity_quantile=args.top_cellularity_quantile,
        limit_slides=args.limit_slides,
        truncate_patches=args.truncate_patches,
    )
