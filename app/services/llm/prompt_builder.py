from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd

from config.constants import GS1_CONTEXT_COLUMNS


class PromptBuilder:
    """Builds prompts for GS1 classification."""

    def __init__(self):
        pass

    # -------------------- GS1 Classification Prompt -------------------- #

    def build_gs1_classification_prompt(
        self,
        products: List[Dict],
        candidates_map: Dict[int, List[Dict]],
    ) -> str:
        """Build prompt for GS1 classification with RAG candidates.

        Args:
            products: List of dicts with row_id and product context columns.
            candidates_map: {row_id: [candidate_dict, ...]} from CandidateBuilder.

        Returns:
            Formatted prompt string for the LLM.
        """
        # Build product rows
        product_lines = []
        for p in products:
            row_obj = {"row_id": p["row_id"]}
            for col in GS1_CONTEXT_COLUMNS:
                val = p.get(col, "")
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    val = ""
                row_obj[col] = str(val)
            product_lines.append(json.dumps(row_obj, ensure_ascii=False))
        products_text = "\n".join(product_lines)

        # Build candidate blocks per product
        candidate_blocks = []
        for p in products:
            row_id = p["row_id"]
            candidates = candidates_map.get(row_id, [])
            if not candidates:
                continue
            block = f"Candidates for row {row_id}:"
            for cand in candidates:
                letter = cand["letter"]
                display = cand["display_text"]
                block += f"\n  [{letter}] {display}"
            candidate_blocks.append(block)
        candidates_text = "\n\n".join(candidate_blocks)

        # Build example response
        example_items = []
        for p in products:
            example_items.append(f'{{"row_id": {p["row_id"]}, "choice": "A"}}')
        example_json = "[\n  " + ",\n  ".join(example_items) + "\n]"

        prompt = (
            "Classify each product into a GS1 category by choosing the best candidate.\n\n"
            f"Products:\n{products_text}\n\n"
            f"{candidates_text}\n\n"
            f"Return ONLY a JSON array:\n{example_json}"
        )
        return prompt


__all__ = ["PromptBuilder"]
