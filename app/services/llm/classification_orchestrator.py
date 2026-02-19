from __future__ import annotations

from collections import Counter
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .azure_client import AzureClient
from .prompt_builder import PromptBuilder
from helpers.data_operations import JsonManager
from config.constants import (
    GS1_NONE_VALUE,
    GS1_CONTEXT_COLUMNS,
    GS1_SYSTEM_MESSAGE,
    EMBEDDING_COLUMN,
)
from services.rag.vector_store import VectorStoreService
from services.rag.candidate_builder import CandidateBuilder
from utils.logging import get_logger
from utils.console import console

logger = get_logger(__name__)


class Batcher:
    def __init__(self, client: AzureClient, parser: "Parser", builder: PromptBuilder):
        self.client = client
        self.parser = parser
        self.builder = builder

    def iterate_unclassified_batches(
        self,
        df: pd.DataFrame,
        target_col: str,
        batch_size: int,
    ) -> Iterable[pd.DataFrame]:
        """Yield batches of rows whose target column is still null."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        mask = df[target_col].isna()
        indices = df[mask].index.tolist()
        for i in range(0, len(indices), batch_size):
            chunk_idx = indices[i : i + batch_size]
            batch_df = df.loc[chunk_idx].copy()
            batch_df.insert(0, "ROW_ID", batch_df.index)
            yield batch_df


class Parser:
    @staticmethod
    def extract_first_json_array(text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("[")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def parse_classification_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response expecting department classification."""
        if not response_text:
            logger.warning("Empty response text; nothing to parse.")
            return []
        json_part = self.extract_first_json_array(response_text)
        if not json_part:
            logger.warning("No JSON array detected in response.")
            return []
        try:
            data = json.loads(json_part)
        except Exception as e:
            logger.warning("JSON decode failed: %s", e)
            return []
        if not isinstance(data, list):
            logger.warning(
                "Top-level JSON is not a list (type=%s); skipping.", type(data).__name__
            )
            return []
        results: List[Dict[str, Any]] = []
        for idx, obj in enumerate(data):
            if not isinstance(obj, dict):
                logger.warning(
                    "Skipping non-dict entry at index %d (type=%s).",
                    idx,
                    type(obj).__name__,
                )
                continue
            # Check for required fields: row_id, department
            if "row_id" not in obj:
                logger.warning("Missing row_id in entry at index %d: %s", idx, obj)
                continue
            if "department" not in obj:
                logger.warning(
                    "Missing department key in entry at index %d: %s", idx, obj
                )
                continue
            try:
                row_id_val = int(obj["row_id"])
                department_val = str(obj["department"]).strip()
                
                if department_val == "":
                    logger.warning("Empty department at index %d; skipped.", idx)
                    continue
                
                results.append({"row_id": row_id_val, "department": department_val})
            except Exception as e:
                logger.warning(
                    "Failed coercion in entry at index %d: %s (err=%s)", idx, obj, e
                )
                continue
        return results


def run_classification(
    batcher: Batcher,
    df: pd.DataFrame,
    context: str,
    departments: List[str],
    department_context: Dict[str, str],
    target_col: str = "department_by_category",
    batch_size: int = 10,
    partial_output_json: Optional[str] = None,
) -> pd.DataFrame:
    """Classify products into department column."""
    # Ensure target column exists
    if target_col not in df.columns:
        df[target_col] = None
    
    total_rows = len(df)
    initial_unclassified = df[target_col].isna().sum()
    total_batches = (initial_unclassified + batch_size - 1) // batch_size
    
    logger.info(
        "Starting department classification: %d total rows, %d unclassified, "
        "batch_size=%d, total_batches=%d",
        total_rows,
        initial_unclassified,
        batch_size,
        total_batches,
    )
    
    # Console: show classification start
    console.classification_start(total_rows, batch_size, initial_unclassified)
    
    jm = JsonManager()
    batch_counter = 0
    
    # Get the product name column for display
    product_col = "product_name"
    
    for batch_df in batcher.iterate_unclassified_batches(df, target_col, batch_size):
        batch_start_time = time.time()
        batch_num = batch_counter + 1
        row_ids = batch_df["ROW_ID"].tolist()
        
        # Get product names for console display
        product_names = []
        if product_col in batch_df.columns:
            product_names = batch_df[product_col].fillna("(unknown)").tolist()
        else:
            product_names = [f"Row {rid}" for rid in row_ids]
        
        # Console: show batch start
        console.batch_start(batch_num, total_batches, row_ids, product_names)
        
        # Log detailed batch info
        logger.debug("Batch %d: row_ids=%s", batch_num, row_ids)
        logger.debug("Batch %d: products=%s", batch_num, product_names[:3])
        
        # Build prompt and send to LLM
        prompt = batcher.builder.build_classification_prompt(
            batch_df,
            context=context,
            departments=departments,
            department_context=department_context,
        )
        logger.debug("Batch %d: prompt length=%d chars", batch_num, len(prompt))
        
        response_text, usage = batcher.client.send(prompt)
        tokens_used = usage.get("total_tokens", 0) if usage else 0
        logger.debug(
            "Batch %d: response length=%d chars, tokens=%d",
            batch_num,
            len(response_text) if response_text else 0,
            tokens_used,
        )
        
        # Parse response
        parsed = batcher.parser.parse_classification_response(response_text or "")
        logger.debug("Batch %d: parsed %d items", batch_num, len(parsed))
        
        # Apply classifications (department)
        applied = 0
        failed = 0
        department_counts: Dict[str, int] = Counter()
        product_results = []
        
        # Create a mapping of row_id to classification for display
        classification_map = {}
        for obj in parsed:
            rid = obj.get("row_id")
            if isinstance(rid, int):
                classification_map[rid] = obj.get("department", "")
        
        # Apply department to dataframe
        for obj in parsed:
            row_id = obj.get("row_id")
            department = obj.get("department")
            
            if isinstance(row_id, int) and isinstance(department, str):
                if row_id in df.index and pd.isna(df.at[row_id, target_col]):
                    df.at[row_id, target_col] = department
                    department_counts[department] += 1
                    applied += 1
                else:
                    failed += 1
                    logger.warning(
                        "Batch %d: row_id %d not found or already classified",
                        batch_num,
                        row_id,
                    )
        
        failed += len(batch_df) - len(parsed)  # Count unparsed as failed
        
        # Build product results for display (show product name → department)
        for idx, row_id in enumerate(row_ids):
            product_name = product_names[idx] if idx < len(product_names) else f"Row {row_id}"
            assigned_department = classification_map.get(row_id, "(unclassified)")
            product_results.append(
                {"product": str(product_name), "segment": assigned_department}
            )
        
        batch_elapsed = time.time() - batch_start_time
        remaining_unclassified = df[target_col].isna().sum()
        
        # Console: show batch result with product mappings
        console.batch_result(
            classified=applied,
            requested=len(batch_df),
            elapsed=batch_elapsed,
            category_counts=dict(department_counts),
            failed=failed if failed > 0 else 0,
            tokens=tokens_used,
            product_results=product_results,
        )
        
        # Log detailed results
        logger.info(
            "Batch %d/%d: %d/%d classified in %.1fs. Departments: %s. Remaining: %d/%d",
            batch_num,
            total_batches,
            applied,
            len(batch_df),
            batch_elapsed,
            dict(department_counts),
            remaining_unclassified,
            total_rows,
        )
        
        # Save partial output after EVERY batch
        if partial_output_json:
            try:
                output_path = Path(partial_output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                jm.write(output_path, df)
                logger.debug("Partial JSON written to %s", partial_output_json)
            except Exception as e:
                logger.warning("Failed to write partial JSON '%s': %s", partial_output_json, e)
        
        batch_counter += 1
    
    logger.info("Classification loop complete: %d batches processed", batch_counter)
    return df


__all__ = ["Batcher", "Parser", "run_classification", "parse_gs1_response", "run_gs1_classification"]


# =====================================================================
# GS1 Classification — Parser & Orchestrator
# =====================================================================


def parse_gs1_response(
    response_text: str,
    candidate_map: Dict[int, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Parse LLM response and map choices back to GS1 column values.

    Args:
        response_text: Raw LLM response string.
        candidate_map: {row_id: {"A": candidate_data, "B": ...}}

    Returns:
        List of dicts with row_id and 6 GS1 column values.
    """
    parser = Parser()
    json_part = parser.extract_first_json_array(response_text or "")
    if not json_part:
        logger.warning("No JSON array found in GS1 response")
        return []

    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse GS1 response JSON: %s", e)
        return []

    if not isinstance(parsed, list):
        logger.warning("GS1 response is not a list")
        return []

    results: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        row_id = item.get("row_id")
        choice = item.get("choice", "").upper().strip()

        if row_id is None or not choice:
            logger.warning("Invalid GS1 response item: %s", item)
            continue

        row_id = int(row_id)
        row_candidates = candidate_map.get(row_id, {})
        candidate = row_candidates.get(choice)

        if candidate is None:
            logger.warning(
                "Row %d: choice '%s' not in candidates (available: %s)",
                row_id,
                choice,
                list(row_candidates.keys()),
            )
            continue

        hierarchy = candidate.get("hierarchy_path", [])
        result = {
            "row_id": row_id,
            "gs1_segment": hierarchy[0] if len(hierarchy) > 0 else GS1_NONE_VALUE,
            "gs1_family": hierarchy[1] if len(hierarchy) > 1 else GS1_NONE_VALUE,
            "gs1_class": hierarchy[2] if len(hierarchy) > 2 else GS1_NONE_VALUE,
            "gs1_brick": hierarchy[3] if len(hierarchy) > 3 else GS1_NONE_VALUE,
            "gs1_attribute": candidate.get("attribute_name", GS1_NONE_VALUE),
            "gs1_attribute_value": candidate.get("attribute_value", GS1_NONE_VALUE),
        }
        results.append(result)

    logger.debug("Parsed %d GS1 classifications from response", len(results))
    return results


def run_gs1_classification(
    batch_df: pd.DataFrame,
    vector_store: VectorStoreService,
    candidate_builder: CandidateBuilder,
    prompt_builder: PromptBuilder,
    client: AzureClient,
    primary_key: str = "id",
    timeout: int = 120,
) -> Dict[str, Any]:
    """Orchestrate GS1 classification for a batch of products.

    Steps:
        1. Read embeddings from embedding_context column.
        2. FAISS search for each product.
        3. Build candidates per product.
        4. Build prompt with all products + candidates.
        5. Send to LLM.
        6. Parse response and map to GS1 columns.

    Args:
        batch_df: DataFrame with product rows (must include embedding_context).
        vector_store: Loaded VectorStoreService.
        candidate_builder: CandidateBuilder instance.
        prompt_builder: PromptBuilder instance.
        client: AzureClient for LLM calls.
        primary_key: Name of the primary key column.

    Returns:
        Dict with keys:
            results: List of dicts with row_id and 6 GS1 column values.
            rag_details: Per-product RAG retrieval info.
            candidates_by_product: Per-product candidate lists.
            prompt: The full prompt sent to the LLM.
            response_text: Raw LLM response.
            usage: Token usage dict.
            timing: Dict with rag_time, llm_time, parse_time.
    """
    products: List[Dict[str, Any]] = []
    candidates_map: Dict[int, List[Dict[str, Any]]] = {}
    candidate_letter_map: Dict[int, Dict[str, Dict[str, Any]]] = {}

    # Per-product RAG details for console output
    rag_details: List[Dict[str, Any]] = []
    candidates_by_product: List[Dict[str, Any]] = []

    # Product name lookup
    product_names: Dict[int, str] = {}
    for _, row in batch_df.iterrows():
        rid = int(row[primary_key])
        pname = str(row.get("product_name", "")) if "product_name" in row.index else ""
        product_names[rid] = pname

    # --- Step 1-3: RAG retrieval + candidate building ---
    rag_start = time.time()

    for _, row in batch_df.iterrows():
        row_id = int(row[primary_key])

        # Parse embedding
        embedding_raw = row.get(EMBEDDING_COLUMN)
        if embedding_raw is None:
            logger.warning("Row %d: no embedding_context, skipping RAG search", row_id)
            continue

        try:
            if isinstance(embedding_raw, str):
                embedding = np.array(json.loads(embedding_raw), dtype=np.float32)
            elif isinstance(embedding_raw, (list, np.ndarray)):
                embedding = np.array(embedding_raw, dtype=np.float32)
            else:
                logger.warning("Row %d: unsupported embedding type %s", row_id, type(embedding_raw))
                continue
        except Exception as e:
            logger.warning("Row %d: failed to parse embedding: %s", row_id, e)
            continue

        # FAISS search
        raw_results = vector_store.search(embedding)
        logger.debug(
            "Row %d: FAISS returned %d results (top scores: %s)",
            row_id,
            len(raw_results),
            [f"{r['score']:.3f}" for r in raw_results[:5]],
        )

        # Collect RAG detail for console
        top_scores = [r["score"] for r in raw_results[:5]]
        matched_areas = []
        seen_areas = set()
        for r in raw_results[:10]:
            hp = r.get("hierarchy_path", [])
            if len(hp) >= 2:
                area = f"{hp[0]} > {hp[1]}"
                if area not in seen_areas:
                    seen_areas.add(area)
                    matched_areas.append(area)
        rag_details.append({
            "row_id": row_id,
            "product_name": product_names.get(row_id, ""),
            "num_raw_results": len(raw_results),
            "top_scores": top_scores,
            "matched_areas": matched_areas,
        })

        # Build candidates
        candidates = candidate_builder.build_candidates(raw_results)
        if not candidates:
            logger.warning("Row %d: no candidates after filtering", row_id)
            continue

        # Store candidates
        candidates_map[row_id] = candidates
        candidate_letter_map[row_id] = {c["letter"]: c for c in candidates}

        # Collect candidate detail for console
        candidates_by_product.append({
            "row_id": row_id,
            "product_name": product_names.get(row_id, ""),
            "candidates": [
                {
                    "letter": c["letter"],
                    "display_text": c["display_text"],
                    "best_score": c["best_score"],
                }
                for c in candidates
            ],
        })

        # Build product context dict
        product = {"row_id": row_id}
        for col in GS1_CONTEXT_COLUMNS:
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = ""
            product[col] = str(val)
        products.append(product)

    rag_time = time.time() - rag_start

    if not products:
        logger.warning("No products with valid embeddings/candidates in batch")
        return {
            "results": [],
            "rag_details": rag_details,
            "candidates_by_product": candidates_by_product,
            "prompt": "",
            "response_text": "",
            "usage": {},
            "timing": {"rag_time": rag_time, "llm_time": 0.0, "parse_time": 0.0},
        }

    # --- Step 4-5: Build prompt & send to LLM ---
    prompt = prompt_builder.build_gs1_classification_prompt(products, candidates_map)
    logger.debug("GS1 prompt: %d chars, %d products", len(prompt), len(products))

    llm_start = time.time()
    response_text, usage = client.send(
        prompt, system_message=GS1_SYSTEM_MESSAGE, timeout=timeout
    )
    llm_time = time.time() - llm_start

    tokens_used = usage.get("total_tokens", 0) if usage else 0
    logger.info("GS1 LLM response: %d tokens", tokens_used)

    # --- Step 6: Parse response ---
    parse_start = time.time()
    results = parse_gs1_response(response_text or "", candidate_letter_map)
    parse_time = time.time() - parse_start

    return {
        "results": results,
        "rag_details": rag_details,
        "candidates_by_product": candidates_by_product,
        "prompt": prompt,
        "response_text": response_text or "",
        "usage": usage or {},
        "timing": {
            "rag_time": rag_time,
            "llm_time": llm_time,
            "parse_time": parse_time,
        },
    }
