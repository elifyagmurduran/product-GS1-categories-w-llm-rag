"""LLM classification services.

Exports:
    AzureClient: Thin wrapper around Azure OpenAI chat completion endpoint.
    PromptBuilder: Builds structured classification prompts for batches.
    run_gs1_classification: GS1 RAG classification orchestrator.
    parse_gs1_response: GS1 response parser.
    VectorStoreService: FAISS index + lookup loading and search.
    CandidateBuilder: Post-filter, candidate structuring.
"""

from .llm.azure_client import AzureClient
from .llm.prompt_builder import PromptBuilder
from .llm.classification_orchestrator import (
    run_gs1_classification,
    parse_gs1_response,
)
from .rag.vector_store import VectorStoreService
from .rag.candidate_builder import CandidateBuilder

__all__ = [
    "AzureClient",
    "PromptBuilder",
    "run_gs1_classification",
    "parse_gs1_response",
    "VectorStoreService",
    "CandidateBuilder",
]