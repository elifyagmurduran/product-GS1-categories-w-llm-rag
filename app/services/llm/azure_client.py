from __future__ import annotations

import os
import time
from typing import Optional

import requests
from config.exceptions import PipelineError
from utils.logging import get_logger
from utils.console import console

logger = get_logger(__name__)

# Generic system message for classification tasks
DEFAULT_SYSTEM_MESSAGE = (
    "You are a precise product classification assistant. "
    "Classify products based on the provided information and category options. "
    "Return structured results in the requested format."
)


class AzureClient:
    """Azure OpenAI client for LLM requests."""

    def __init__(self, api_key: str, deployment: str, endpoint: str, api_version: str):
        self.api_key = api_key
        self.deployment = deployment
        self.endpoint = endpoint
        self.api_version = api_version
        self.full_endpoint = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

    @classmethod
    def from_env(cls) -> Optional["AzureClient"]:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if all([api_key, deployment, endpoint, api_version]):
            return cls(api_key, deployment, endpoint, api_version)  # type: ignore
        return None

    def send(self, prompt: str, *, timeout: int = 60, system_message: str | None = None) -> tuple[str, dict]:
        """Send prompt to LLM and return response with usage stats.
        
        Args:
            prompt: User prompt text.
            timeout: Request timeout in seconds.
            system_message: Optional override for the system message.
                Defaults to a generic classification message if not provided.
        
        Returns:
            tuple of (response_message, usage_dict)
            usage_dict contains: prompt_tokens, completion_tokens, total_tokens
        
        Raises:
            PipelineError on failure.
        """
        if not self.full_endpoint or not self.api_key:
            raise PipelineError("AzureClient not configured")

        sys_msg = system_message or DEFAULT_SYSTEM_MESSAGE

        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
        }

        logger.info("Sending request to LLM...")
        logger.debug(
            "Endpoint: %s, Prompt length: %d chars", self.full_endpoint, len(prompt)
        )

        try:
            response = requests.post(
                self.full_endpoint, headers=headers, json=payload, timeout=timeout
            )
        except Exception as e:
            logger.error("Request failed: %s", e)
            raise PipelineError(f"LLM request failed: {e}") from e

        logger.debug("Response status: %d", response.status_code)

        if response.status_code != 200:
            error_msg = response.text[:500]
            try:
                error_msg = str(response.json())
            except Exception:
                pass
            logger.error("LLM error [%d]: %s", response.status_code, error_msg)
            raise PipelineError(f"LLM error [{response.status_code}]: {error_msg}")

        try:
            data = response.json()
        except Exception as e:
            raise PipelineError(f"Failed to parse LLM response: {e}") from e

        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        if usage:
            logger.info(
                "Tokens used: %d (prompt=%d, completion=%d)",
                usage.get("total_tokens", 0),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

        if message.strip().startswith("["):
            row_count = message.count("row_id")
            logger.info("Classified %d rows", row_count)
        else:
            logger.debug("Response: %s...", message[:100] if message else "(empty)")

        return message, usage


__all__ = ["AzureClient"]
