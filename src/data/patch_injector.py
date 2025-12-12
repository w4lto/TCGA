from pathlib import Path
import argparse

from src.utils.config_utils import load_config
from src.data.ch_utils import ClickHouseClient
from src.data.patch_extractor import extract_patches_from_wsi


def main(config_path: str):
    cfg = load_config(config_path)
    ch_cfg = cfg.clickhouse
    assert ch_cfg is not None

    ch = ClickHouseClient(ch_cfg)
    client = ch.get_client()

    df_slides = client.query_df(
        f"""
        SELECT patient_id, slide_id, wsi_path, stage_label, split
        FROM {ch_cfg.table_slides}
        """
    )

    patches_root = cfg.data_root / "processed" / "patches_tcga"
    patch_size = cfg.patch_size
    stride = cfg.stride
    max_patches = cfg.max_patches_per_wsi

    batch_rows = []

    for _, row in df_slides.iterrows():
        slide_path = Path(row["wsi_path"])
        patient_id = row["patient_id"]
        slide_id = row["slide_id"]
        stage_label = row["stage_label"]
        split = row["split"]

        out_dir = patches_root / slide_id
        patch_infos = extract_patches_from_wsi(
            slide_path,
            out_dir,
            patch_size=patch_size,
            stride=stride,
            tissue_threshold=0.5,
            top_cellularity_quantile=0.8,
            max_patches=max_patches,
            patient_id_hint=patient_id,
        )

        for i, info in enumerate(patch_infos):
            patch_id = f"{slide_id}_x{info.x}_y{info.y}_{i}"
            batch_rows.append(
                (
                    patient_id,
                    slide_id,
                    patch_id,
                    str(info.image_path.resolve()),
                    info.x,
                    info.y,
                    info.level,
                    patch_size,
                    info.tissue_score,
                    info.cellularity,
                    stage_label,
                    "TCGA_BRCA_PATCH",
                    split,
                )
            )

        if len(batch_rows) >= 10_000:
            ch.insert_patches(batch_rows)
            batch_rows.clear()

    if batch_rows:
        ch.insert_patches(batch_rows)

    print("Patches inseridos em tcga_patches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
