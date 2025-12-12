from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.config_utils import load_config, TrainConfig
from src.data.ch_utils import ClickHouseClient

def _normalize_stage(raw_stage: str) -> str | None:
    """
    Normaliza o estágio clínico vindo do TCGA para I/II/III/IV.
    Exemplo de valores típicos: 'stage iia', 'Stage III', 'stage iv', etc.
    """
    if not isinstance(raw_stage, str):
        return None

    s = raw_stage.strip().lower()
    if "stage i" in s and "ii" not in s and "iii" not in s and "iv" not in s:
        return "I"
    if "stage ii" in s and "iii" not in s and "iv" not in s:
        return "II"
    if "stage iii" in s and "iv" not in s:
        return "III"
    if "stage iv" in s:
        return "IV"

    # fallback: tenta bater exatamente
    if s in {"i", "ii", "iii", "iv"}:
        return s.upper()

    return None


def _load_clinical_table(clinical_path: Path) -> pd.DataFrame:
    """
    Lê o TSV clínico do TCGA e retorna DataFrame com:
      - patient_id
      - stage_label (I/II/III/IV)
    """
    df = pd.read_csv(clinical_path, sep="\t", dtype=str)

    # Exemplos de colunas que costumam trazer o estágio:
    #   - "ajcc_pathologic_tumor_stage"
    #   - "tumor_stage"
    stage_col_candidates = [
        "ajcc_pathologic_tumor_stage",
        "tumor_stage",
        "pathologic_stage",
    ]
    stage_col = None
    for col in stage_col_candidates:
        if col in df.columns:
            stage_col = col
            break

    if stage_col is None:
        raise RuntimeError(
            f"Não encontrei coluna de estágio em {clinical_path}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    # Usamos 'case_submitter_id' ou 'submitter_id' como patient_id
    patient_col_candidates = ["case_submitter_id", "submitter_id", "patient_id"]
    patient_col = None
    for col in patient_col_candidates:
        if col in df.columns:
            patient_col = col
            break

    if patient_col is None:
        raise RuntimeError(
            f"Não encontrei coluna de paciente em {clinical_path}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["patient_id"] = df[patient_col].astype(str)
    out["raw_stage"] = df[stage_col].astype(str)

    out["stage_label"] = out["raw_stage"].apply(_normalize_stage)
    out = out.dropna(subset=["stage_label"]).reset_index(drop=True)

    return out[["patient_id", "stage_label"]]


def _discover_wsi_files(wsi_root: Path) -> pd.DataFrame:
    """
    Descobre arquivos .svs em wsi_root e retorna DataFrame com:
      - slide_id: nome do arquivo sem extensão
      - patient_id: inferido a partir do prefixo (até o primeiro '_', por exemplo)
      - image_path: caminho absoluto
    """
    paths: List[Path] = sorted(wsi_root.glob("*.svs"))
    rows: List[Dict[str, str]] = []

    for p in paths:
        slide_id = p.stem
        # Estratégia comum no TCGA: case_id = primeiros 12 caracteres
        # ex: TCGA-XX-YYYY-01Z-... -> patient_id = TCGA-XX-YYYY
        patient_id = slide_id[:12]
        rows.append(
            {
                "slide_id": slide_id,
                "patient_id": patient_id,
                "image_path": str(p.resolve()),
            }
        )

    return pd.DataFrame(rows)


def _make_splits(
    df_slides: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_slides_per_stage: int | None = None,
) -> pd.DataFrame:
    """
    Cria colunas de split (train/val/test) estratificadas por stage_label.
    Opcionalmente limita o número de slides por estágio (para base inicial pequena).
    """
    rng = np.random.default_rng(seed)
    df = df_slides.copy()

    all_rows: List[pd.DataFrame] = []

    for stage_label, group in df.groupby("stage_label"):
        g = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        if max_slides_per_stage is not None:
            g = g.head(max_slides_per_stage).copy()

        n = len(g)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            # Garante pelo menos 1 em train, se possível
            n_train = max(1, n_train)
            n_val = max(0, n_val)
            n_test = max(0, n - n_train - n_val)

        splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * (n - n_train - n_val)
        )
        rng.shuffle(splits)

        g["split"] = splits
        all_rows.append(g)

    return pd.concat(all_rows, axis=0).reset_index(drop=True)

def ingest_tcga_stage_to_clickhouse(cfg: TrainConfig) -> None:
    """
    Monta base inicial de slides TCGA-BRCA em ClickHouse:
      - tabela tcga_slides
    """
    assert cfg.clickhouse is not None, "Configuração de ClickHouse não encontrada"

    data_root = Path(cfg.data_root)
    wsi_root = data_root / "raw" / "tcga_brca" / "wsi"
    clinical_path = data_root / "raw" / "tcga_brca" / "clinical" / "tcga_brca_clinical.tsv"

    if not wsi_root.exists():
        raise RuntimeError(f"Diretório de WSI não encontrado: {wsi_root}")

    if not clinical_path.exists():
        raise RuntimeError(f"Arquivo clínico não encontrado: {clinical_path}")

    print(f"Lendo tabela clínica: {clinical_path}")
    df_clin = _load_clinical_table(clinical_path)

    print(f"Descobrindo arquivos WSI em: {wsi_root}")
    df_wsi = _discover_wsi_files(wsi_root)

    df = df_wsi.merge(df_clin, on="patient_id", how="inner")
    print(f"Slides com estágio válido: {len(df)}")

    if len(df) == 0:
        raise RuntimeError("Nenhum slide com estágio clínico válido encontrado.")

    # Cria splits estratificados
    max_per_stage = getattr(cfg, "tcga_max_slides_per_stage", None)
    df = _make_splits(
        df,
        val_ratio=cfg.val_split,
        test_ratio=cfg.test_split,
        seed=cfg.seed,
        max_slides_per_stage=max_per_stage,
    )

    df["dataset_source"] = "TCGA-BRCA"

    print("Exemplo de linhas:")
    print(df.head())

    ch = ClickHouseClient(cfg.clickhouse)
    client = ch.get_client()

    print("Limpando tabela tcga_slides (se existir)...")
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {cfg.clickhouse.table_slides} (
            patient_id String,
            slide_id String,
            image_path String,
            stage_label String,
            dataset_source String,
            split String
        )
        ENGINE = MergeTree
        ORDER BY (dataset_source, stage_label, patient_id, slide_id)
        """
    )
    client.command(f"TRUNCATE TABLE {cfg.clickhouse.table_slides}")

    print("Inserindo slides na tabela tcga_slides...")
    records: List[Tuple[str, str, str, str, str, str]] = []
    for row in df.itertuples(index=False):
        records.append(
            (
                row.patient_id,
                row.slide_id,
                row.image_path,
                row.stage_label,
                row.dataset_source,
                row.split,
            )
        )

    client.insert(
        cfg.clickhouse.table_slides,
        records,
        column_names=[
            "patient_id",
            "slide_id",
            "image_path",
            "stage_label",
            "dataset_source",
            "split",
        ],
    )

    print(f"Inseridos {len(records)} registros em {cfg.clickhouse.table_slides}.")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    ingest_tcga_stage_to_clickhouse(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
