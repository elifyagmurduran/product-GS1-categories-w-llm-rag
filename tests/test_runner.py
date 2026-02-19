"""
Test Mode Runner - Local JSON-based classification pipeline.

This replicates the original pipeline behavior:
- Downloads data from DB to local JSON
- Runs classification on local DataFrame
- Outputs results to local JSON
- NEVER writes back to the database

Usage:
    python tests/test_runner.py

Configuration via .env:
    TEST_ROW_LIMIT: Number of rows to fetch (default: 100)
    LLM_BATCH_SIZE: Batch size for LLM calls (default: 10)
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Resolve project root (../..)
# This ensures we work correctly regardless of where the script is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add app directory to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "app"))

try:
    import pandas as pd
except ImportError as e:
    print(f"\n[ERROR] Failed to import pandas: {e}")
    print(f"Please check your Python environment (Python {sys.version.split()[0]}).\n")
    sys.exit(1)

from dotenv import load_dotenv

from config.constants import (
    DEPARTMENTS,
    DEPARTMENT_CONTEXT,
    DEFAULT_CLASSIFICATION_CONTEXT,
    TARGET_DEPARTMENT_COL,
    get_int_env,
)
from config.exceptions import PipelineError
from db.db_connector import DBConnector
from helpers.data_operations import (
    JsonManager,
    validate_classification_output,
)
from services import AzureClient, Batcher, Parser, PromptBuilder, run_classification
from utils.logging import get_logger, init_logging
from utils.console import console


# Test mode data directories (located in project root's data folder)
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test_run"
TEST_INPUT_JSON = TEST_DATA_DIR / "input.json"
TEST_OUTPUT_JSON = TEST_DATA_DIR / "output.json"


@dataclass
class TestConfig:
    """Configuration for test mode."""

    schema: str
    table: str
    row_limit: int
    batch_size: int

    @classmethod
    def from_env(cls) -> "TestConfig":
        return cls(
            schema=os.getenv("AZURE_SQL_SCHEMA", "playground"),
            table=os.getenv("AZURE_SQL_TABLE", "promo_bronze"),
            row_limit=get_int_env("TEST_ROW_LIMIT", 100) or 100,
            batch_size=get_int_env("LLM_BATCH_SIZE", 10) or 10,
        )


# Setup logging
init_logging("test")
logger = get_logger(__name__)


def export_data(connector: DBConnector, cfg: TestConfig) -> tuple[int, int, float]:
    """Export raw data from SQL table into JSON. Returns (rows, cols, elapsed)."""
    TEST_INPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Fetch data directly as DataFrame
    df = connector.fetch_table(table=cfg.table, schema=cfg.schema, top=cfg.row_limit)

    # Save to JSON (Test Mode specific logic)
    df.to_json(TEST_INPUT_JSON, orient="records", indent=2, force_ascii=False)

    rows, cols = df.shape
    elapsed = time.time() - start_time
    logger.info(
        "Exported to %s: %d rows, %d cols in %.1fs",
        TEST_INPUT_JSON,
        rows,
        cols,
        elapsed,
    )
    return rows, cols, elapsed


def load_json_data(json_file: Path) -> pd.DataFrame:
    """Load JSON into DataFrame and add target column if needed."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    logger.debug("Loaded %d rows from %s", len(data), json_file)
    df = pd.DataFrame(data)

    # Add target column if it doesn't exist
    if TARGET_DEPARTMENT_COL not in df.columns:
        df[TARGET_DEPARTMENT_COL] = None

    logger.info("Loaded %d rows with target column '%s'", len(df), TARGET_DEPARTMENT_COL)
    return df


def classify_data(df: pd.DataFrame, cfg: TestConfig) -> float:
    """Run department classification and save output. Returns elapsed time."""
    start_time = time.time()
    jm = JsonManager()

    def _on_interrupt(signum, frame):
        try:
            jm.write(TEST_OUTPUT_JSON, df)
            logger.info("Partial data saved on interrupt to %s", TEST_OUTPUT_JSON)
        except Exception as e:
            logger.warning("Failed to save on interrupt: %s", e)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _on_interrupt)

    client = AzureClient.from_env()
    if client is None:
        raise PipelineError("Azure OpenAI not configured (missing env vars)")

    logger.info("Azure client initialized: deployment=%s", client.deployment)

    batcher = Batcher(client=client, parser=Parser(), builder=PromptBuilder())

    run_classification(
        batcher,
        df,
        context=DEFAULT_CLASSIFICATION_CONTEXT,
        departments=DEPARTMENTS,
        department_context=DEPARTMENT_CONTEXT,
        target_col=TARGET_DEPARTMENT_COL,
        batch_size=cfg.batch_size,
        partial_output_json=str(TEST_OUTPUT_JSON),
    )

    elapsed = time.time() - start_time
    logger.info("Department classification complete in %.1fs", elapsed)

    # Get validation stats for console summary
    stats = validate_classification_output(
        df,
        target_col=TARGET_DEPARTMENT_COL,
        expected_options=DEPARTMENTS,
        print_report=False,
        as_dict=True,
    )
    
    if stats:
        console.classification_summary(
            total_rows=stats["total_rows"],
            classified=stats["classified_rows"],
            unique_categories=stats["unique_categories"],
            total_categories=len(DEPARTMENTS),
            top_categories=stats["top_frequencies"],
            unexpected=stats.get("unexpected_values", []),
            output_path=str(TEST_OUTPUT_JSON),
            elapsed=elapsed,
        )

    jm.write(TEST_OUTPUT_JSON, df)
    logger.info("Output saved to %s", TEST_OUTPUT_JSON)
    return elapsed


def run_test_mode() -> int:
    """
    Run the test mode pipeline.
    
    This is the original pipeline behavior:
    - Downloads data from DB to local JSON
    - Classifies products locally
    - Saves results to local JSON
    - NEVER writes to database
    
    Returns exit code.
    """
    pipeline_start = time.time()
    
    try:
        load_dotenv()
        cfg = TestConfig.from_env()
        logger.info(
            "[TEST MODE] Pipeline starting with config: schema=%s, table=%s, "
            "limit=%d, batch_size=%d",
            cfg.schema,
            cfg.table,
            cfg.row_limit,
            cfg.batch_size,
        )

        console.start(
            "Test Mode Pipeline", f"Connecting to [{cfg.schema}].[{cfg.table}]..."
        )

        connector = DBConnector()
        connector.connect_and_verify(schema=cfg.schema, table=cfg.table)

        rows, cols, export_time = export_data(connector, cfg)
        console.data_loaded(
            source=f"[{cfg.schema}].[{cfg.table}]",
            rows=rows,
            columns=cols,
            elapsed=export_time,
        )
        
        df = load_json_data(TEST_INPUT_JSON)
        classify_data(df, cfg)

        total_elapsed = time.time() - pipeline_start
        logger.info("[TEST MODE] Pipeline finished successfully in %.1fs", total_elapsed)
        logger.info("[TEST MODE] Output saved to: %s", TEST_OUTPUT_JSON)
        console.pipeline_finished(success=True)
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        console.interrupted()
        return 130
    except PipelineError as e:
        logger.error("Pipeline error: %s", e)
        console.error("Pipeline Error", str(e))
        console.pipeline_finished(success=False)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        console.error("Unexpected Error", str(e))
        console.pipeline_finished(success=False)
        return 1


if __name__ == "__main__":
    exit(run_test_mode())
