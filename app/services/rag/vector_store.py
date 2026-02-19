"""FAISS vector store service for GS1 RAG retrieval.

Loads the FAISS index and gs1_lookup.pkl, performs similarity search,
and returns raw results with metadata.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.constants import (
    FAISS_INDEX_PATH,
    GS1_LOOKUP_PATH,
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD,
    EMBEDDING_DIMENSIONS,
)
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss-cpu is required for RAG search. Install: pip install faiss-cpu"
    ) from e


class VectorStoreService:
    """Manages FAISS index and GS1 lookup for similarity search."""

    def __init__(self, index: Any, lookup: Dict[int, Dict[str, Any]]) -> None:
        self.index = index
        self.lookup = lookup

    @property
    def index_size(self) -> int:
        """Number of vectors in the FAISS index."""
        return self.index.ntotal

    @classmethod
    def from_files(
        cls,
        index_path: Optional[str | Path] = None,
        lookup_path: Optional[str | Path] = None,
    ) -> "VectorStoreService":
        """Load FAISS index and lookup from disk.

        Args:
            index_path: Path to faiss_gs1.index. Defaults to FAISS_INDEX_PATH.
            lookup_path: Path to gs1_lookup.pkl. Defaults to GS1_LOOKUP_PATH.

        Returns:
            Initialized VectorStoreService.

        Raises:
            FileNotFoundError: If files don't exist.
            PipelineError: If loading fails.
        """
        idx_path = Path(index_path or FAISS_INDEX_PATH)
        lkp_path = Path(lookup_path or GS1_LOOKUP_PATH)

        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        if not lkp_path.exists():
            raise FileNotFoundError(f"GS1 lookup not found: {lkp_path}")

        logger.info("Loading FAISS index from %s ...", idx_path)
        index = faiss.read_index(str(idx_path))
        logger.info("FAISS index loaded: %d vectors, dimension=%d", index.ntotal, index.d)

        logger.info("Loading GS1 lookup from %s ...", lkp_path)
        with open(lkp_path, "rb") as f:
            lookup = pickle.load(f)
        logger.info("GS1 lookup loaded: %d entries", len(lookup))

        return cls(index=index, lookup=lookup)

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = RAG_TOP_K,
        score_threshold: float = RAG_SCORE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Search the FAISS index for nearest neighbors.

        Args:
            embedding: 1-D numpy array (float32) of dimension EMBEDDING_DIMENSIONS.
            top_k: Number of nearest neighbors to retrieve.
            score_threshold: Minimum similarity score to include.

        Returns:
            List of dicts with keys: vector_id, score, and all metadata from lookup.
        """
        # Ensure correct shape: (1, d)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)

        # Log embedding stats for debugging
        emb_norm = np.linalg.norm(embedding)
        emb_mean = np.mean(embedding)
        emb_std = np.std(embedding)
        logger.debug(
            "Embedding stats: dim=%d, norm=%.4f, mean=%.4f, std=%.4f",
            embedding.shape[1], emb_norm, emb_mean, emb_std
        )

        # Normalize for cosine similarity (inner product on normalized vectors)
        if emb_norm > 0:
            embedding = embedding / emb_norm

        scores, ids = self.index.search(embedding, top_k)
        scores = scores[0]
        ids = ids[0]

        # Log raw scores for debugging (before threshold filter)
        valid_scores = [s for s, vid in zip(scores, ids) if vid != -1]
        if valid_scores:
            logger.debug(
                "Raw FAISS scores: top 5 = [%s], max=%.4f, min=%.4f",
                ", ".join(f"{s:.4f}" for s in valid_scores[:5]),
                max(valid_scores),
                min(valid_scores)
            )
        else:
            logger.warning("FAISS returned no valid results (all -1 sentinel values)")

        results: List[Dict[str, Any]] = []
        below_threshold_count = 0
        for score, vid in zip(scores, ids):
            if vid == -1:
                continue  # FAISS sentinel for empty slot
            if score < score_threshold:
                below_threshold_count += 1
                continue

            metadata = self.lookup.get(int(vid))
            if metadata is None:
                logger.debug("Vector ID %d not found in lookup, skipping", vid)
                continue

            result = {
                "vector_id": int(vid),
                "score": float(score),
                **metadata,
            }
            results.append(result)

        if below_threshold_count > 0:
            logger.debug(
                "Filtered out %d results below threshold %.2f (best filtered score: %.4f)",
                below_threshold_count,
                score_threshold,
                max([s for s, vid in zip(scores, ids) if vid != -1 and s < score_threshold], default=0.0)
            )

        logger.debug(
            "Search returned %d results after threshold filter (top_k=%d, threshold=%.2f)",
            len(results),
            top_k,
            score_threshold,
        )
        return results

    def search_batch(
        self,
        embeddings: List[np.ndarray],
        top_k: int = RAG_TOP_K,
        score_threshold: float = RAG_SCORE_THRESHOLD,
    ) -> List[List[Dict[str, Any]]]:
        """Search for multiple embeddings at once.

        Args:
            embeddings: List of 1-D numpy arrays.
            top_k: Number of nearest neighbors per query.
            score_threshold: Minimum similarity score.

        Returns:
            List of result lists, one per input embedding.
        """
        return [
            self.search(emb, top_k=top_k, score_threshold=score_threshold)
            for emb in embeddings
        ]


__all__ = ["VectorStoreService"]
