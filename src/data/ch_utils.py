from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import clickhouse_connect
import pandas as pd

from src.utils.config import ClickHouseConfig


@dataclass
class ClickHouseClient:
    cfg: ClickHouseConfig

    def get_client(self):
        return clickhouse_connect.get_client(
            host=self.cfg.host,
            port=self.cfg.port,
            username=self.cfg.username,
            password=self.cfg.password,
            database=self.cfg.database,
        )

    def create_tables(self):
        client = self.get_client()

        client.command(
            """
            CREATE TABLE IF NOT EXISTS tcga_slides
            (
                patient_id     String,
                slide_id       String,
                wsi_path       String,
                stage_label    LowCardinality(String),
                ajcc_raw       String,
                dataset_source LowCardinality(String),
                split          LowCardinality(String),
                created_at     DateTime DEFAULT now()
            )
            ENGINE = MergeTree
            ORDER BY (patient_id, slide_id)
            """
        )

        client.command(
            """
            CREATE TABLE IF NOT EXISTS tcga_patches
            (
                patient_id     String,
                slide_id       String,
                patch_id       String,
                patch_path     String,
                x              UInt32,
                y              UInt32,
                level          UInt8,
                patch_size     UInt16,
                tissue_score   Float32,
                cellularity    Float32,
                stage_label    LowCardinality(String),
                dataset_source LowCardinality(String),
                split          LowCardinality(String),
                created_at     DateTime DEFAULT now()
            )
            ENGINE = MergeTree
            ORDER BY (patient_id, slide_id, patch_id)
            """
        )

    def insert_slides(self, rows: List[tuple]):
        if not rows:
            return
        client = self.get_client()
        client.insert(
            "tcga_slides",
            rows,
            column_names=[
                "patient_id",
                "slide_id",
                "wsi_path",
                "stage_label",
                "ajcc_raw",
                "dataset_source",
                "split",
            ],
        )

    def insert_patches(self, rows: List[tuple]):
        if not rows:
            return
        client = self.get_client()
        client.insert(
            "tcga_patches",
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

    def load_patches(
        self,
        splits: List[str],
        limit_per_split: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        client = self.get_client()
        result: Dict[str, pd.DataFrame] = {}
        for split in splits:
            q = f"""
                SELECT
                    patient_id,
                    slide_id,
                    patch_path,
                    stage_label,
                    dataset_source
                FROM {self.cfg.table_patches}
                WHERE split = '{split}'
            """
            if limit_per_split is not None:
                q += f" LIMIT {limit_per_split}"
            df = client.query_df(q)
            df = df.rename(
                columns={"patch_path": "image_path", "stage_label": "label"}
            )
            result[split] = df
        return result
