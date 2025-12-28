from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.ch_utils import ClickHouseClient
from src.utils.config_utils import TrainConfig, load_config


TCGA_PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def normalize_stage(raw_stage: str) -> Optional[str]:
    """
    Converte o valor de estágio do clinical para macro-estágios I/II/III/IV.
    Exemplos comuns:
      - 'stage iia' -> II
      - 'Stage III' -> III
      - 'stage iv'  -> IV
    """
    if not isinstance(raw_stage, str):
        return None

    s = raw_stage.strip().lower()
    if not s or s in {"nan", "none"}:
        return None

    # Ordem importa: iv -> iii -> ii -> i
    if "stage iv" in s or s == "iv":
        return "IV"
    if "stage iii" in s or s == "iii":
        return "III"
    if "stage ii" in s or s == "ii":
        return "II"
    if "stage i" in s or s == "i":
        return "I"

    return None


def load_clinical_table(clinical_tsv: Path) -> pd.DataFrame:
    """
    Espera TSV do GDC/TCGA contendo algo como:
      - case_submitter_id (ex: TCGA-A2-A1G0)
      - tumor_stage / ajcc_pathologic_tumor_stage / etc.

    Retorna DF com:
      - patient_id
      - stage_label (I/II/III/IV)
    """
    df = pd.read_csv(clinical_tsv, sep="\t", dtype=str)

    patient_candidates = ["case_submitter_id", "submitter_id", "patient_id", "case_id"]
    stage_candidates = [
        "ajcc_pathologic_tumor_stage",
        "tumor_stage",
        "pathologic_stage",
        "ajcc_pathologic_stage",
        "ajcc_clinical_stage",
        "clinical_stage",
        "stage_label"
    ]

    patient_col = next((c for c in patient_candidates if c in df.columns), None)
    stage_col = next((c for c in stage_candidates if c in df.columns), None)

    if patient_col is None:
        raise RuntimeError(
            f"Não encontrei coluna de paciente/caso no TSV. Colunas: {df.columns.tolist()}"
        )
    if stage_col is None:
        raise RuntimeError(
            f"Não encontrei coluna de estágio no TSV. Colunas: {df.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["patient_id"] = df[patient_col].astype(str).str.strip()
    out["raw_stage"] = df[stage_col].astype(str)

    out["stage_label"] = out["raw_stage"].apply(normalize_stage)
    out = out.dropna(subset=["stage_label"]).reset_index(drop=True)

    return out[["patient_id", "stage_label"]]


def extract_patient_id_from_filename(filename: str) -> Optional[str]:
    """
    Extrai 'TCGA-XX-YYYY' do início do filename.
    Ex:
      TCGA-A2-A1G0-01Z-00-DX1.XXXX.svs -> TCGA-A2-A1G0
    """
    m = TCGA_PATIENT_RE.match(filename)
    if not m:
        return None
    return m.group(1).upper()


def discover_svs_files(wsi_root: Path) -> pd.DataFrame:
    """
    Descobre .svs recursivamente a partir do wsi_root.
    Seu layout é:
      /run/media/.../tcga/<uuid>/<filename>.svs
    """
    paths = sorted(wsi_root.rglob("*.svs"))
    rows: List[Dict[str, str]] = []

    for p in paths:
        slide_filename = p.name
        patient_id = extract_patient_id_from_filename(slide_filename)
        if patient_id is None:
            # Ignora arquivos inesperados
            continue

        slide_id = p.stem  # sem ".svs"
        rows.append(
            {
                "patient_id": patient_id,
                "slide_id": slide_id,
                "image_path": str(p.resolve()),
            }
        )

    return pd.DataFrame(rows)


def make_splits(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_slides_per_stage: Optional[int] = None,
) -> pd.DataFrame:
    """
    Cria split estratificado por stage_label.
    """
    rng = np.random.default_rng(seed)
    out_parts: List[pd.DataFrame] = []

    for stage, g in df.groupby("stage_label"):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        if max_slides_per_stage is not None:
            g = g.head(max_slides_per_stage).copy()

        n = len(g)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            rest = n - 1
            n_val = min(n_val, rest)
            n_test = max(0, rest - n_val)

        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        rng.shuffle(splits)
        g["split"] = splits[:n]

        out_parts.append(g)

    return pd.concat(out_parts, axis=0).reset_index(drop=True)

def ingest_to_clickhouse(
    cfg: TrainConfig,
    wsi_root: Path,
    clinical_tsv: Path,
    max_slides_per_stage: Optional[int],
    truncate: bool,
) -> None:
    ch = ClickHouseClient(cfg.clickhouse).client
    val_split = float(cfg.val_split)
    test_split = float(cfg.test_split)
    seed = int(cfg.seed)
    db = cfg.clickhouse.database
    table = cfg.clickhouse.table_slides

    full_table = f"{db}.{table}"

    df_clin = load_clinical_table(clinical_tsv)
    df_wsi = discover_svs_files(wsi_root)

    print(f"[ingest] clinical rows: {len(df_clin)}")
    print(f"[ingest] svs encontrados: {len(df_wsi)}")

    df = df_wsi.merge(df_clin, on="patient_id", how="inner")
    print(f"[ingest] após join (wsi x clinical): {len(df)}")

    if df.empty:
        clin_sample = df_clin["patient_id"].head(10).tolist()
        wsi_sample = df_wsi["patient_id"].head(10).tolist()
        raise RuntimeError(
            "Nenhum slide casou com o clinical.\n"
            f"Exemplos clinical patient_id: {clin_sample}\n"
            f"Exemplos WSI patient_id: {wsi_sample}\n"
        )

    df = make_splits(
        df,
        val_ratio=val_split,
        test_ratio=test_split,
        seed=seed,
        max_slides_per_stage=max_slides_per_stage,
    )

    df["dataset_source"] = "TCGA-BRCA"

    # cria tabela se não existir
    ch.command(
        f"""
        CREATE TABLE IF NOT EXISTS {full_table} (
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

    if truncate:
        print(f"[ingest] TRUNCATE {full_table}")
        ch.command(f"TRUNCATE TABLE {full_table}")

    records: List[Tuple[str, str, str, str, str, str]] = [
        (
            r.patient_id,
            r.slide_id,
            r.image_path,
            r.stage_label,
            r.dataset_source,
            r.split,
        )
        for r in df.itertuples(index=False)
    ]

    ch.insert(
        full_table,
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

    print(f"[ingest] inseridos {len(records)} slides em {full_table}.")
    print("[ingest] distribuição por estágio/split:")
    print(
        df.groupby(["stage_label", "split"])
        .size()
        .reset_index(name="n")
        .sort_values(["stage_label", "split"])
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument(
        "--wsi_root",
        type=str,
        required=True,
        help="Diretório root onde estão os UUIDs do GDC (ex.: /run/media/.../tcga).",
    )
    parser.add_argument(
        "--clinical_tsv",
        type=str,
        required=True,
        help="Caminho para o TSV clínico (com patient_id + tumor_stage).",
    )
    parser.add_argument(
        "--max_slides_per_stage",
        type=int,
        default=None,
        help="Limita número de slides por estágio (sanity check).",
    )
    parser.add_argument(
        "--no_truncate",
        action="store_true",
        help="Não truncar tabela antes de inserir.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    wsi_root = Path(args.wsi_root).expanduser()
    clinical_tsv = Path(args.clinical_tsv).expanduser()

    if not wsi_root.exists():
        raise RuntimeError(f"wsi_root não existe: {wsi_root}")
    if not clinical_tsv.exists():
        raise RuntimeError(f"clinical_tsv não existe: {clinical_tsv}")

    ingest_to_clickhouse(
        cfg=cfg,
        wsi_root=wsi_root,
        clinical_tsv=clinical_tsv,
        max_slides_per_stage=args.max_slides_per_stage,
        truncate=not args.no_truncate,
    )


if __name__ == "__main__":
    main()
