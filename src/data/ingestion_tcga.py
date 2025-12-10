from pathlib import Path
import re
import argparse

import pandas as pd

from src.utils.config import load_config
from src.data.ch_utils import ClickHouseClient


def normalize_stage(stage_str):
    if not isinstance(stage_str, str):
        return None
    s = stage_str.upper().strip()
    m = re.search(r"STAGE\s+([IVX]+)", s)
    if not m:
        return None
    roman = m.group(1)

    if roman.startswith("IV"):
        return "IV"
    if roman.startswith("III"):
        return "III"
    if roman.startswith("II") and roman != "III":
        return "II"
    if roman.startswith("I") and roman not in ("II", "III", "IV"):
        return "I"
    return None


def main(config_path: str):
    cfg = load_config(config_path)
    ch_cfg = cfg.clickhouse
    assert ch_cfg is not None, "Config clickhouse obrigatória"

    wsi_root = cfg.data_root / "raw" / "tcga_brca_wsi"
    clinical_tsv = cfg.data_root / "raw" / "tcga_brca_clinical.tsv"

    clin = pd.read_csv(clinical_tsv, sep="\t")
    if "ajcc_pathologic_stage" not in clin.columns:
        raise ValueError("ajcc_pathologic_stage não encontrado em clinical.tsv")
    if "case_submitter_id" not in clin.columns:
        raise ValueError("case_submitter_id não encontrado em clinical.tsv")

    clin["stage_norm"] = clin["ajcc_pathologic_stage"].apply(normalize_stage)
    clin = clin.dropna(subset=["stage_norm"])
    stage_by_case = dict(zip(clin["case_submitter_id"], clin["stage_norm"]))

    rows = []
    for wsi_path in wsi_root.rglob("*.svs"):
        m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-.*", wsi_path.stem)
        if not m:
            continue
        patient_id = m.group(1)
        stage = stage_by_case.get(patient_id)
        if stage is None:
            continue

        slide_id = wsi_path.stem
        rows.append(
            (
                patient_id,
                slide_id,
                str(wsi_path.resolve()),
                stage,
                "",  # ajcc_raw (pode ser preenchido com texto original)
                "TCGA_BRCA_WSI",
                "train",
            )
        )

    ch = ClickHouseClient(ch_cfg)
    ch.create_tables()
    ch.insert_slides(rows)
    print(f"Inseridos {len(rows)} slides em tcga_slides")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
