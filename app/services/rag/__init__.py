"""RAG (Retrieval-Augmented Generation) service for GS1 classification.

Exports:
    VectorStoreService: FAISS index + lookup loading and search.
    CandidateBuilder: Post-filter, deduplicate, structure candidates for LLM.
"""

from .vector_store import VectorStoreService
from .candidate_builder import CandidateBuilder

__all__ = ["VectorStoreService", "CandidateBuilder"]
