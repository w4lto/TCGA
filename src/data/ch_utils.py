from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from src.utils.config_utils import ClickHouseConfig

import clickhouse_connect

logger = logging.getLogger(__name__)

class ClickHouseClient:
    """
    Wrapper fino para clickhouse-connect.

    Convenção do projeto:
      - self.client: instância do clickhouse_connect.driver.client.Client
      - self.conn / self.ch: aliases por compatibilidade

    Métodos:
      - command(sql)
      - query(sql) -> result
      - insert(table, records, column_names=...)
    """

    def __init__(self, cfg: ClickHouseConfig ):
        host = cfg.host
        port = cfg.port
        username = cfg.username
        password = cfg.password
        database = cfg.database

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
            # clickhouse-connect não exige close sempre, mas mantém simetria
            self.client.close()
        except Exception:
            pass

    def command(self, sql: str) -> None:
        self.client.command(sql)

    def query(self, sql: str):
        return self.client.query(sql)

    def insert(
        self,
        table: str,
        records: Sequence[Sequence[Any]],
        column_names: Optional[List[str]] = None,
    ) -> None:
        self.client.insert(table, records, column_names=column_names)
