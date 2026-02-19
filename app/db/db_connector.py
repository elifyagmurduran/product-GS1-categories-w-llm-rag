"""Azure SQL Database connector using Azure AD Service Principal authentication.

Environment Variables:
    AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_CLIENT_ID, AZURE_SQL_CLIENT_SECRET
    Optional: AZURE_SQL_TENANT_ID, AZURE_SQL_TIMEOUT, AZURE_SQL_DRIVER
"""
from __future__ import annotations

import os
import urllib.parse
from contextlib import contextmanager
from typing import Any, Iterable

import pandas as pd
import pyodbc
from azure.identity import ClientSecretCredential
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from config.exceptions import PipelineError
from utils.logging import get_logger

logger = get_logger(__name__)


class DBConnector:
    """Manage Azure SQL connections via a lazily-created SQLAlchemy engine."""

    def __init__(self) -> None:
        load_dotenv()
        self.server = os.getenv("AZURE_SQL_SERVER")
        self.database = os.getenv("AZURE_SQL_DATABASE")
        self.client_id = os.getenv("AZURE_SQL_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_SQL_CLIENT_SECRET")
        self.timeout = int(os.getenv("AZURE_SQL_TIMEOUT", "30"))
        self._engine: Engine | None = None

    # ------------------------ Internal helpers ------------------------
    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _resolve_table(self, schema: str | None, table: str | None) -> tuple[str, str]:
        resolved_schema = schema or os.getenv("AZURE_SQL_SCHEMA", "playground")
        resolved_table = table or os.getenv("AZURE_SQL_TABLE", "promo_bronze")
        return resolved_schema, resolved_table

    def _validate_env_vars(self) -> None:
        required = [
            "AZURE_SQL_SERVER",
            "AZURE_SQL_DATABASE",
            "AZURE_SQL_CLIENT_ID",
            "AZURE_SQL_CLIENT_SECRET",
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise PipelineError(f"Missing required Azure SQL env vars: {', '.join(missing)}")

    def _choose_driver(self) -> str:
        requested = os.getenv("AZURE_SQL_DRIVER")
        installed = pyodbc.drivers()
        logger.debug("Installed ODBC drivers: %s", installed)
        if requested:
            if requested in installed:
                return requested
            logger.warning("Requested driver '%s' not found, using auto-detect", requested)
        for preferred in ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"]:
            if preferred in installed:
                return preferred
        for drv in installed:
            if drv == "SQL Server":
                raise PipelineError(
                    "Legacy ODBC driver 'SQL Server' detected but Azure AD requires a modern driver. "
                    "Install: winget install -e --id Microsoft.ODBCDriverForSQLServer.18"
                )
        raise PipelineError(
            "No suitable SQL Server ODBC driver. "
            "Install: winget install -e --id Microsoft.ODBCDriverForSQLServer.18"
        )

    def _build_odbc_string(self) -> str:
        self._validate_env_vars()
        driver = self._choose_driver()
        self.selected_driver = driver  # type: ignore[attr-defined]
        tenant_id = os.getenv("AZURE_SQL_TENANT_ID")
        parts = [
            f"Driver={{{driver}}}",
            f"Server=tcp:{self.server},1433",
            f"Database={self.database}",
            "Authentication=ActiveDirectoryServicePrincipal",
            f"UID={self.client_id}",
            f"PWD={self.client_secret}",
            "Encrypt=yes",
            "TrustServerCertificate=no",
            f"Connection Timeout={self.timeout}",
        ]
        if tenant_id:
            parts.append(f"Authority Id={tenant_id}")
        return ";".join(parts)

    def _create_engine(self) -> Engine:
        try:
            odbc_str = self._build_odbc_string()
        except Exception as exc:
            raise PipelineError(f"Failed building ODBC string: {exc}") from exc

        tenant_id = os.getenv("AZURE_SQL_TENANT_ID")
        if tenant_id and self.client_id and self.client_secret:
            try:
                cred = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                )
                token = cred.get_token("https://database.windows.net/.default")
                if not token.token:
                    raise RuntimeError("Received empty token")
                logger.debug("Azure AD credentials validated")
            except Exception as exc:
                raise PipelineError(
                    "Azure AD credential validation failed: "
                    f"{exc}. Check: client secret, expiration, permissions, tenant ID."
                ) from exc

        params = urllib.parse.quote_plus(odbc_str)
        return create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}",
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def connect_and_verify(self, schema: str | None = None, table: str | None = None) -> None:
        """Verify connection by fetching one row. Raises PipelineError on failure."""
        schema, table = self._resolve_table(schema, table)
        try:
            preview = self.fetch_table(table=table, schema=schema, top=1)
            logger.info(
                "Connected to [%s].[%s] (%d columns) via %s",
                schema,
                table,
                len(preview.columns),
                getattr(self, "selected_driver", "?"),
            )
            logger.debug("Columns: %s", ", ".join(preview.columns))
        except Exception as exc:
            raise PipelineError(f"Connection verification failed: {exc}") from exc

    def preview_table(
        self, table: str | None = None, schema: str | None = None, top: int = 5
    ) -> pd.DataFrame:
        """Return up to 'top' rows for a quick glance."""
        schema, table = self._resolve_table(schema, table)
        df = self.fetch_table(table=table, schema=schema, top=top)
        logger.info("Preview %s.%s: %d rows", schema, table, len(df))
        return df

    # ---------------- Query / execution helpers ----------------
    @contextmanager
    def get_connection(self) -> Iterable[Connection]:
        """Context-managed connection (commit on success, rollback on exception)."""
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def query_to_dataframe(self, query_text: str) -> pd.DataFrame:
        """Execute a read-only SQL query and return a DataFrame."""
        try:
            return pd.read_sql_query(text(query_text), self.engine)
        except SQLAlchemyError:
            raise

    def execute_query(self, query_text: str) -> Any:
        """Execute a SQL statement. Returns fetched rows or rowcount."""
        with self.get_connection() as conn:
            result = conn.execute(text(query_text))
            if result.returns_rows:
                return result.fetchall()
            return result.rowcount

    def get_table_schema(self, table: str, schema: str | None = None) -> pd.DataFrame:
        """Return INFORMATION_SCHEMA metadata for a table."""
        schema = schema or os.getenv("AZURE_SQL_SCHEMA", "dbo")
        q = f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
        ORDER BY ORDINAL_POSITION;
        """
        return self.query_to_dataframe(q)

    def fetch_table(
        self,
        table: str | None = None,
        schema: str | None = None,
        top: int | None = None,
    ) -> pd.DataFrame:
        """Fetch table rows with optional TOP limit.

        If no explicit top provided, apply a default cap of 500 rows to avoid
        accidentally loading huge tables into memory. Set AZURE_SQL_TOP_LIMIT
        environment variable or pass top argument explicitly to override.
        Pass top=None explicitly to fetch all rows (use cautiously).
        """
        schema, table = self._resolve_table(schema, table)
        if top is None:
            top_env = os.getenv("AZURE_SQL_TOP_LIMIT")
            if top_env and top_env.isdigit():
                parsed = int(top_env)
                top = parsed if parsed > 0 else None
            else:
                top = 500
        prefix = f"TOP {top} " if top else ""
        sql = f"SELECT {prefix}* FROM [{schema}].[{table}]"
        return self.query_to_dataframe(sql)

    # -------------------- GS1 Classification Methods -------------------- #

    def count_unclassified_gs1_rows(
        self,
        table: str | None = None,
        schema: str | None = None,
    ) -> int:
        """Count rows where gs1_segment IS NULL (unprocessed for GS1)."""
        schema, table = self._resolve_table(schema, table)
        sql = f"""
        SELECT COUNT(*) as cnt FROM [{schema}].[{table}]
        WHERE [gs1_segment] IS NULL
        """
        result = self.query_to_dataframe(sql)
        return int(result["cnt"].iloc[0])

    def fetch_unclassified_gs1_batch(
        self,
        batch_size: int,
        table: str | None = None,
        schema: str | None = None,
        primary_key: str = "id",
    ) -> pd.DataFrame:
        """Fetch a batch of rows where gs1_segment IS NULL.

        Includes the embedding_context column needed for RAG search.
        Always fetches TOP N (offset=0) since processed rows get updated.
        """
        schema, table = self._resolve_table(schema, table)
        sql = f"""
        SELECT TOP {batch_size} * FROM [{schema}].[{table}]
        WHERE [gs1_segment] IS NULL
        ORDER BY [{primary_key}]
        """
        df = self.query_to_dataframe(sql)
        logger.debug("Fetched %d unclassified GS1 rows", len(df))
        return df

    def update_gs1_classifications(
        self,
        updates: list[dict],
        table: str | None = None,
        schema: str | None = None,
        primary_key: str = "id",
    ) -> int:
        """Write GS1 classifications to DB.

        Each update dict must contain:
            id, gs1_segment, gs1_family, gs1_class,
            gs1_brick, gs1_attribute, gs1_attribute_value

        SAFETY: Only updates rows WHERE gs1_segment IS NULL.
        """
        schema, table = self._resolve_table(schema, table)
        if not updates:
            return 0

        updated_count = 0
        with self.get_connection() as conn:
            for update in updates:
                pk_value = update.get("row_id") or update.get(primary_key)
                if pk_value is None:
                    logger.warning("Skipping GS1 update with no PK: %s", update)
                    continue

                sql = f"""
                UPDATE [{schema}].[{table}]
                SET [gs1_segment]         = :seg,
                    [gs1_family]          = :fam,
                    [gs1_class]           = :cls,
                    [gs1_brick]           = :brk,
                    [gs1_attribute]       = :attr,
                    [gs1_attribute_value] = :attrv
                WHERE [{primary_key}] = :pk_val
                  AND [gs1_segment] IS NULL
                """

                result = conn.execute(
                    text(sql),
                    {
                        "seg": update.get("gs1_segment", "NONE"),
                        "fam": update.get("gs1_family", "NONE"),
                        "cls": update.get("gs1_class", "NONE"),
                        "brk": update.get("gs1_brick", "NONE"),
                        "attr": update.get("gs1_attribute", "NONE"),
                        "attrv": update.get("gs1_attribute_value", "NONE"),
                        "pk_val": pk_value,
                    },
                )
                if result.rowcount > 0:
                    updated_count += result.rowcount

        logger.info(
            "GS1 updated %d/%d rows in [%s].[%s]",
            updated_count,
            len(updates),
            schema,
            table,
        )
        return updated_count
