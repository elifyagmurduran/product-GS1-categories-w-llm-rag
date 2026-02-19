"""Candidate builder for GS1 classification.

Takes raw FAISS search results, deduplicates by L4 brick,
attaches L5/L6 attribute info when available, and returns
structured candidate lists ready for the LLM prompt.
"""
from __future__ import annotations

from typing import Any, Dict, List

from config.constants import GS1_NONE_VALUE, RAG_MAX_CANDIDATES_PER_PRODUCT
from utils.logging import get_logger

logger = get_logger(__name__)

# Letters for candidate labelling
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CandidateBuilder:
    """Build structured candidate lists from raw FAISS results."""

    def __init__(self, max_candidates: int = RAG_MAX_CANDIDATES_PER_PRODUCT) -> None:
        self.max_candidates = max_candidates

    def build_candidates(
        self, raw_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert raw FAISS results into deduplicated, lettered candidates.

        Logic:
        - For L1-L4 hits: include directly as candidate brick paths.
        - For L5/L6 hits: trace up to L4 brick via hierarchy_path[:4].
        - Deduplicate by L4 brick path.
        - For each unique brick, note if L5/L6 info is available.
        - Sort by best score per brick.
        - Assign letters A, B, C, ...

        Args:
            raw_results: List of dicts from VectorStoreService.search().
                Each dict has: vector_id, score, level, hierarchy_path,
                title, code, and optionally parent fields.

        Returns:
            List of candidate dicts with keys:
                letter, hierarchy_path, hierarchy_string, brick_code,
                best_score, attribute_name, attribute_value, display_text
        """
        if not raw_results:
            return []

        # Group by L4 brick path (first 4 levels of hierarchy)
        brick_groups: Dict[str, Dict[str, Any]] = {}

        for result in raw_results:
            hierarchy_path = result.get("hierarchy_path", [])
            level = result.get("level", 0)
            score = result.get("score", 0.0)

            if not hierarchy_path:
                continue

            # Extract the L4 brick path (first 4 levels)
            brick_path = tuple(hierarchy_path[:4])
            brick_key = " > ".join(brick_path)

            # Get or create brick group
            if brick_key not in brick_groups:
                brick_groups[brick_key] = {
                    "hierarchy_path": list(brick_path),
                    "hierarchy_string": brick_key,
                    "brick_code": result.get("code", ""),
                    "best_score": score,
                    "attribute_name": GS1_NONE_VALUE,
                    "attribute_value": GS1_NONE_VALUE,
                }
            else:
                # Update best score if this match is better
                if score > brick_groups[brick_key]["best_score"]:
                    brick_groups[brick_key]["best_score"] = score

            # If this is an L5 or L6 hit, attach attribute info
            if level >= 5 and len(hierarchy_path) >= 5:
                group = brick_groups[brick_key]
                # Only update attribute if we don't have one yet or this score is better
                if group["attribute_name"] == GS1_NONE_VALUE or score > group["best_score"]:
                    # L5 = attribute name (index 4), L6 = attribute value (index 5)
                    group["attribute_name"] = (
                        hierarchy_path[4] if len(hierarchy_path) > 4 else GS1_NONE_VALUE
                    )
                    group["attribute_value"] = (
                        hierarchy_path[5] if len(hierarchy_path) > 5 else GS1_NONE_VALUE
                    )

        # Sort by best score descending
        sorted_bricks = sorted(
            brick_groups.values(),
            key=lambda x: x["best_score"],
            reverse=True,
        )

        # Limit to max candidates
        sorted_bricks = sorted_bricks[: self.max_candidates]

        # Assign letters and build display text
        candidates = []
        for i, brick in enumerate(sorted_bricks):
            if i >= len(_LETTERS):
                break

            letter = _LETTERS[i]

            # Build display text
            display = brick["hierarchy_string"]
            if brick["attribute_name"] != GS1_NONE_VALUE:
                attr_str = brick["attribute_name"]
                if brick["attribute_value"] != GS1_NONE_VALUE:
                    attr_str += f" > {brick['attribute_value']}"
                display += f" | Attribute: {attr_str}"

            candidate = {
                "letter": letter,
                "hierarchy_path": brick["hierarchy_path"],
                "hierarchy_string": brick["hierarchy_string"],
                "brick_code": brick["brick_code"],
                "best_score": brick["best_score"],
                "attribute_name": brick["attribute_name"],
                "attribute_value": brick["attribute_value"],
                "display_text": display,
            }
            candidates.append(candidate)

        logger.debug(
            "Built %d candidates from %d raw results",
            len(candidates),
            len(raw_results),
        )
        return candidates


__all__ = ["CandidateBuilder"]
