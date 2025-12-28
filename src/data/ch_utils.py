from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import clickhouse_connect

logger = logging.getLogger(__name__)


class ClickHouseClient:
    """
    Wrapper fino para clickhouse-connect.

    Convenção do projeto:
      - self.client: instância do clickhouse_connect.driver.client.Client
      - self.conn / self.ch: aliases por compatibilidade

    Helpers:
      - query_df(sql) -> pandas.DataFrame
      - load_patches(...) -> dict split->DataFrame
      - load_slides(...) -> pandas.DataFrame
    """

    def __init__(self, cfg: Any):
        # Aceita ClickHouseConfig (dataclass) ou dict-like, para robustez.
        if is_dataclass(cfg):
            c = asdict(cfg)
        elif isinstance(cfg, dict):
            c = cfg
        else:
            # último fallback: tenta acessar atributos
            c = {
                "host": getattr(cfg, "host", "clickhouse"),
                "port": getattr(cfg, "port", 8123),
                "username": getattr(cfg, "username", "default"),
                "password": getattr(cfg, "password", ""),
                "database": getattr(cfg, "database", "default"),
                "table_slides": getattr(cfg, "table_slides", "tcga_slides"),
                "table_patches": getattr(cfg, "table_patches", "tcga_patches"),
            }

        host = c.get("host", "clickhouse")
        port = int(c.get("port", 8123))
        username = c.get("username", "default")
        password = c.get("password", "")
        database = c.get("database", "default")

        # nomes de tabelas (podem vir como "db.table" ou apenas "table")
        self.database = database
        self.table_slides = c.get("table_slides", "tcga_slides")
        self.table_patches = c.get("table_patches", "tcga_patches")

        logger.info("Conectando ao ClickHouse host=%s port=%s user=%s db=%s", host, port, username, database)

        self.client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )

        # aliases (compat)
        self.conn = self.client
        self.ch = self.client

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def command(self, sql: str) -> None:
        self.client.command(sql)

    def query(self, sql: str):
        return self.client.query(sql)

    def query_df(self, sql: str):
        # clickhouse-connect provê query_df
        return self.client.query_df(sql)

    def insert(
        self,
        table: str,
        records: Sequence[Sequence[Any]],
        column_names: Optional[List[str]] = None,
    ) -> None:
        self.client.insert(table, records, column_names=column_names)

    def _full_table(self, table: str) -> str:
        if "." in table:
            return table
        return f"{self.database}.{table}"

    def load_slides(
        self,
        splits: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        limit_per_split: Optional[int] = None,
    ):
        """
        Carrega slides (metadados) do ClickHouse.
        """
        cols = columns or ["patient_id", "slide_id", "image_path", "stage_label", "split", "dataset_source"]
        table = self._full_table(self.table_slides)

        if not splits:
            q = f"SELECT {', '.join(cols)} FROM {table}"
            return self.query_df(q)

        # Se há splits, concatena via UNION ALL para permitir LIMIT por split
        queries: List[str] = []
        for sp in splits:
            q = f"SELECT {', '.join(cols)} FROM {table} WHERE split = '{sp}'"
            if limit_per_split:
                q += f" LIMIT {int(limit_per_split)}"
            queries.append(q)

        return self.query_df(" UNION ALL ".join(queries))

    def load_patches(
        self,
        splits: List[str],
        limit_per_split: Optional[int] = None,
        columns: Optional[List[str]] = None,
        where_extra: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Carrega patches para os splits informados.

        Retorna dict split -> DataFrame

        Observação:
          - usa `patch_path` como caminho da imagem (o patch_injector grava isso)
          - `label` será `stage_label` por padrão (classificação por estágio I-IV),
            mas você pode trocar no tf_dataset para binário ou outro mapeamento.
        """
        cols = columns or [
            "patient_id",
            "slide_id",
            "patch_id",
            "patch_path",
            "stage_label",
            "split",
            "dataset_source",
            "tissue_score",
            "cellularity",
            "patch_size",
            "x",
            "y",
            "level",
        ]

        table = self._full_table(self.table_patches)

        out: Dict[str, Any] = {}
        for sp in splits:
            filters = [f"split = '{sp}'"]
            if where_extra:
                filters.append(f"({where_extra})")
            where = " AND ".join(filters)

            q = f"SELECT {', '.join(cols)} FROM {table} WHERE {where}"
            if limit_per_split:
                q += f" LIMIT {int(limit_per_split)}"

            out[sp] = self.query_df(q)

        return out
