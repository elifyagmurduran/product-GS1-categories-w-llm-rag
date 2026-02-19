"""
GS1 Classification Pipeline - RAG-based GS1 GPC classification.

Usage:
    python app/main.py

Safety Features:
- Only processes rows where GS1 columns are NULL
- Only updates rows where GS1 columns are NULL (double-check)
- Idempotent: safe to re-run multiple times
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from config.constants import (
    BATCH_SIZE,
    PRIMARY_KEY_COL,
)
from config.exceptions import PipelineError
from db.db_connector import DBConnector
from services import (
    AzureClient,
    PromptBuilder,
    run_gs1_classification,
    VectorStoreService,
    CandidateBuilder,
)
from utils.logging import get_logger, init_logging
from utils.console import console


# -------------------- Configuration -------------------- #
@dataclass
class ProductionConfig:
    schema: str = os.getenv("AZURE_SQL_SCHEMA", "playground")
    table: str = os.getenv("AZURE_SQL_TABLE", "promo_bronze")
    batch_size: int = BATCH_SIZE
    primary_key: str = PRIMARY_KEY_COL




# -------------------- Setup -------------------- #
init_logging("main")
logger = get_logger(__name__)


def main() -> int:
    """
    Run the GS1 RAG classification pipeline.
    
    Returns exit code.
    """
    pipeline_start = time.time()
    
    try:
        load_dotenv()
        cfg = ProductionConfig()
        
        return run_gs1_pipeline(cfg, pipeline_start)

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

def run_gs1_pipeline(cfg: ProductionConfig, pipeline_start: float) -> int:
    """Run the GS1 RAG classification pipeline."""
    logger.info("[GS1 MODE] Pipeline starting")
    console.start("GS1 RAG Pipeline", f"Connecting to [{cfg.schema}].[{cfg.table}]...")

    # Initialize DB connection
    connector = DBConnector()
    connector.connect_and_verify(schema=cfg.schema, table=cfg.table)

    # Count unclassified GS1 rows — before loading heavy resources
    total_unclassified = connector.count_unclassified_gs1_rows(
        table=cfg.table, schema=cfg.schema
    )

    if total_unclassified == 0:
        logger.info("No unclassified GS1 rows found. Nothing to do.")
        console.info("Complete", "No unclassified GS1 rows found.")
        console.pipeline_finished(success=True)
        return 0

    logger.info("Found %d unclassified GS1 rows to process", total_unclassified)
    console.info("Unclassified GS1 Rows", f"{total_unclassified} rows to process")

    # Lazy-load FAISS index and lookup (only if there are rows to process)
    console.info("Loading Resources", "Loading FAISS index and GS1 lookup...")
    load_start = time.time()
    vector_store = VectorStoreService.from_files()
    load_elapsed = time.time() - load_start
    logger.info("Loaded FAISS index: %d vectors in %.1fs", vector_store.index_size, load_elapsed)
    console.info("Resources Loaded", f"{vector_store.index_size} vectors in {load_elapsed:.1f}s")

    # Initialize LLM client with longer timeout for GS1 (complex prompts)
    client = AzureClient.from_env()
    if client is None:
        raise PipelineError("Azure OpenAI not configured (missing env vars)")
    logger.info("Azure client initialized: deployment=%s", client.deployment)

    # GS1 prompts are longer (candidates from RAG) - use longer timeout
    gs1_timeout = int(os.getenv("AZURE_OPENAI_TIMEOUT", "120"))
    logger.info("GS1 timeout configured: %ds", gs1_timeout)

    candidate_builder = CandidateBuilder()
    prompt_builder = PromptBuilder()

    # Process in batches
    total_updated = 0
    batch_num = 0
    total_batches = (total_unclassified + cfg.batch_size - 1) // cfg.batch_size
    console.classification_start(total_unclassified, cfg.batch_size, total_unclassified)

    while True:
        batch_num += 1
        batch_start = time.time()

        # Fetch batch (always offset=0 since we update processed rows)
        df = connector.fetch_unclassified_gs1_batch(
            batch_size=cfg.batch_size,
            table=cfg.table,
            schema=cfg.schema,
            primary_key=cfg.primary_key,
        )

        if df.empty:
            logger.info("No more unclassified GS1 rows. Stopping.")
            break

        logger.info("GS1 batch %d: %d rows", batch_num, len(df))

        # Show batch products in console
        row_ids = df[cfg.primary_key].tolist()
        product_names = (
            df["product_name"].fillna("(unknown)").tolist()
            if "product_name" in df.columns
            else [f"Row {r}" for r in row_ids]
        )
        console.batch_start(batch_num, total_batches, row_ids, product_names)

        # Run GS1 classification (RAG → LLM → parse)
        detail = run_gs1_classification(
            batch_df=df,
            vector_store=vector_store,
            candidate_builder=candidate_builder,
            prompt_builder=prompt_builder,
            client=client,
            primary_key=cfg.primary_key,
            timeout=gs1_timeout,
        )

        results = detail["results"]
        timing = detail["timing"]
        usage = detail["usage"]

        # --- Console: RAG retrieval details ---
        console.gs1_rag_details(detail["rag_details"])

        # --- Console: Candidate options ---
        console.gs1_candidates(detail["candidates_by_product"])

        # --- Console: Full prompt ---
        console.gs1_prompt(detail["prompt"])

        # --- Console: Token usage ---
        console.gs1_tokens(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        # Write results to DB
        db_start = time.time()
        updated = connector.update_gs1_classifications(
            updates=results,
            table=cfg.table,
            schema=cfg.schema,
            primary_key=cfg.primary_key,
        )
        db_time = time.time() - db_start
        total_updated += updated

        batch_elapsed = time.time() - batch_start

        # --- Console: DB write confirmation ---
        console.gs1_db_write(results)

        # --- Console: Timing breakdown ---
        console.gs1_timing_breakdown(
            rag_time=timing["rag_time"],
            llm_time=timing["llm_time"],
            db_time=db_time,
            total_time=batch_elapsed,
        )

        # --- Console: Running totals + progress ---
        console.gs1_running_totals(
            batch_num=batch_num,
            total_batches=total_batches,
            total_classified=len(results),
            total_unclassified=total_unclassified,
            total_updated=total_updated,
            pipeline_elapsed=time.time() - pipeline_start,
        )

        logger.info(
            "GS1 batch %d/%d: %d classified, %d updated in %.1fs. Total: %d/%d",
            batch_num, total_batches, len(results), updated,
            batch_elapsed, total_updated, total_unclassified,
        )

        # Safety: avoid infinite loop
        if updated == 0 and len(df) > 0:
            logger.warning("GS1 batch had %d rows but 0 updates. Breaking.", len(df))
            break

    total_elapsed = time.time() - pipeline_start
    logger.info("[GS1 MODE] Pipeline complete in %.1fs. Total updated: %d", total_elapsed, total_updated)
    console.info("Complete", f"GS1 classified {total_updated} rows in {total_elapsed:.1f}s")
    console.pipeline_finished(success=True)
    return 0


if __name__ == "__main__":
    exit(main())
