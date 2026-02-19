from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data")

# Production mode settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
PRIMARY_KEY_COL = os.getenv("PRIMARY_KEY_COL", "id")

# --------------- GS1 Classification ---------------
GS1_TARGET_COLUMNS = [
    "gs1_segment",
    "gs1_family",
    "gs1_class",
    "gs1_brick",
    "gs1_attribute",
    "gs1_attribute_value",
]

GS1_NONE_VALUE = "NONE"  # Sentinel for "processed but no value at this level"

# Vector store paths
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_gs1.index"
GS1_LOOKUP_PATH = VECTOR_STORE_DIR / "gs1_lookup.pkl"

# RAG search config
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "30"))
# Lower threshold to 0.50 - GS1 embeddings may not match as closely
# Use RAG_SCORE_THRESHOLD env var to override (e.g., 0.60, 0.70)
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.50"))
RAG_MAX_CANDIDATES_PER_PRODUCT = 12  # Max candidates shown to LLM per product

# Embedding config
EMBEDDING_COLUMN = "embedding_context"
EMBEDDING_DIMENSIONS = 1024

# GS1 context columns (what the LLM sees about each product)
GS1_CONTEXT_COLUMNS = [
    "product_name",
    "product_name_en",
    "packaging_value",
    "packaging_unit",
]

# GS1 system prompt
GS1_SYSTEM_MESSAGE = (
    "You are a product classification assistant using the GS1 GPC "
    "(Global Product Classification) standard. For each product, select "
    "the single best matching GS1 category from the provided candidates. "
    "Always make a selection \u2014 pick the closest match even if imperfect."
)


def get_int_env(key: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


__all__ = [
    "BATCH_SIZE",
    "PRIMARY_KEY_COL",
    "get_int_env",
    # GS1
    "GS1_TARGET_COLUMNS",
    "GS1_NONE_VALUE",
    "VECTOR_STORE_DIR",
    "FAISS_INDEX_PATH",
    "GS1_LOOKUP_PATH",
    "RAG_TOP_K",
    "RAG_SCORE_THRESHOLD",
    "RAG_MAX_CANDIDATES_PER_PRODUCT",
    "EMBEDDING_COLUMN",
    "EMBEDDING_DIMENSIONS",
    "GS1_CONTEXT_COLUMNS",
    "GS1_SYSTEM_MESSAGE",
]
