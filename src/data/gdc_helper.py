from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Ex.: "Stage IA", "stage iiic", "Stage IIB"
STAGE_RE = re.compile(r"stage\s+([iv]+)", re.IGNORECASE)


def normalize_stage(raw: Optional[str]) -> Optional[str]:
    """
    Normaliza estágio clínico/Patológico do TCGA para macro-estágios I/II/III/IV.
    Aceita strings como "Stage IA", "Stage IIIC", "stage iib".
    """
    if not raw:
        return None

    s = str(raw).strip().lower()
    if not s or s in {"nan", "none", "not reported"}:
        return None

    m = STAGE_RE.search(s)
    if not m:
        # Às vezes vem "i", "ii", etc.
        if s in {"i", "ii", "iii", "iv"}:
            return s.upper()
        return None

    roman = m.group(1).lower()

    # Macro estágio é o prefixo do romano:
    # ia -> i, iiic -> iii, etc.
    if roman.startswith("iv"):
        return "IV"
    if roman.startswith("iii"):
        return "III"
    if roman.startswith("ii"):
        return "II"
    if roman.startswith("i"):
        return "I"
    return None


def iter_file_ids_from_gdc_download_root(root: Path) -> List[str]:
    """
    Layout típico do gdc-client:
      root/<file_id>/<filename>.svs
    Então o nome do diretório é o file_id.
    """
    file_ids: List[str] = []
    for p in root.iterdir():
        if p.is_dir() and UUID_RE.match(p.name):
            if any(p.glob("*.svs")):
                file_ids.append(p.name)
    return sorted(set(file_ids))


def chunked(xs: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


def gdc_post_json(url: str, payload: dict, timeout_s: int = 60, retries: int = 5) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Falha ao chamar GDC API após {retries} tentativas: {last_err}")


def fetch_files_with_cases(file_ids: List[str]) -> List[Dict]:
    """
    Consulta /files no GDC e expande cases + diagnoses.
    """
    endpoint = "https://api.gdc.cancer.gov/files"

    # Campos relevantes (inclui o que de fato aparece no seu retorno)
    fields = [
        "file_id",
        "file_name",
        "cases.submitter_id",
        "cases.diagnoses.diagnosis_is_primary_disease",
        "cases.diagnoses.ajcc_pathologic_stage",  # <-- ESTE é o seu caso
        "cases.diagnoses.ajcc_pathologic_tumor_stage",
        "cases.diagnoses.tumor_stage",
        "cases.diagnoses.ajcc_clinical_stage",
        "cases.diagnoses.ajcc_clinical_tumor_stage",
    ]

    hits: List[Dict] = []

    for batch in chunked(file_ids, 200):
        filters = {
            "op": "in",
            "content": {"field": "file_id", "value": batch},
        }

        payload = {
            "filters": filters,
            "format": "JSON",
            "size": len(batch),
            "fields": ",".join(fields),
            "expand": "cases,cases.diagnoses",
        }

        data = gdc_post_json(endpoint, payload)
        hits.extend(data.get("data", {}).get("hits", []))

    return hits


def _pick_stage_from_diagnoses(diagnoses: List[Dict]) -> Optional[str]:
    """
    Preferência:
    1) diagnosis_is_primary_disease == True com ajcc_pathologic_stage
    2) qualquer diagnosis com ajcc_pathologic_stage
    3) fallback para outros campos
    """
    if not diagnoses:
        return None

    def stage_from(d: Dict) -> Optional[str]:
        return (
            d.get("ajcc_pathologic_stage")  # o seu exemplo
            or d.get("ajcc_clinical_stage")
            or d.get("ajcc_pathologic_tumor_stage")
            or d.get("tumor_stage")
            or d.get("ajcc_clinical_tumor_stage")
        )

    primaries = [d for d in diagnoses if d.get("diagnosis_is_primary_disease") is True]
    for d in primaries:
        st = stage_from(d)
        if st:
            return st

    for d in diagnoses:
        st = stage_from(d)
        if st:
            return st

    return None


def extract_patient_and_stage(hit: Dict) -> Tuple[Optional[str], Optional[str]]:
    cases = hit.get("cases") or []
    if not cases:
        return None, None

    case = cases[0]
    patient_id = case.get("submitter_id")  # ex: TCGA-BH-A18H

    diagnoses = case.get("diagnoses") or []
    stage_raw = _pick_stage_from_diagnoses(diagnoses)

    return patient_id, normalize_stage(stage_raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gdc_root",
        required=True,
        type=str,
        help="Root do download do GDC (diretórios UUID com .svs dentro).",
    )
    ap.add_argument(
        "--out_tsv",
        required=True,
        type=str,
        help="TSV de saída com patient_id e stage_label.",
    )
    ap.add_argument(
        "--keep_unknown",
        action="store_true",
        help="Mantém pacientes sem estágio como stage_label=UNKNOWN.",
    )
    args = ap.parse_args()

    root = Path(args.gdc_root)
    if not root.exists():
        raise RuntimeError(f"gdc_root não existe: {root}")

    file_ids = iter_file_ids_from_gdc_download_root(root)
    if not file_ids:
        raise RuntimeError(f"Nenhum diretório UUID com .svs encontrado em: {root}")

    print(f"[gdc] file_ids encontrados: {len(file_ids)}")

    hits = fetch_files_with_cases(file_ids)
    print(f"[gdc] hits retornados: {len(hits)}")

    rows: List[Tuple[str, str]] = []
    ignored = 0

    for h in hits:
        patient_id, stage = extract_patient_and_stage(h)
        if not patient_id:
            ignored += 1
            continue

        if stage is None:
            if args.keep_unknown:
                rows.append((patient_id, "UNKNOWN"))
            else:
                ignored += 1
            continue

        rows.append((patient_id, stage))

    # Dedup por patient_id: mantém o primeiro (suficiente para agora).
    best: Dict[str, str] = {}
    for pid, st in rows:
        best.setdefault(pid, st)

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("patient_id\tstage_label\n")
        for pid, st in sorted(best.items()):
            f.write(f"{pid}\t{st}\n")

    print(f"[gdc] TSV gerado: {out_path} (patients={len(best)}, ignorados={ignored})")


if __name__ == "__main__":
    main()
