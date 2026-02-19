"""LLM classification services.

Exports:
	AzureClient: Thin wrapper around Azure OpenAI chat completion endpoint.
	PromptBuilder: Builds structured classification prompts for batches.
	Batcher: Iterates over unclassified dataframe rows in batches.
	Parser: Extracts JSON classification outputs from model responses.
	run_classification: High-level helper applying classifications to a DataFrame.
"""

from .azure_client import AzureClient
from .prompt_builder import PromptBuilder
from .classification_orchestrator import Batcher, Parser, run_classification

__all__ = [
	"AzureClient",
	"PromptBuilder",
	"Batcher",
	"Parser",
	"run_classification",
]
