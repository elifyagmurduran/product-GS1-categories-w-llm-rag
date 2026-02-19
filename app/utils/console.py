"""Pretty console output for the classifier pipeline.

This module provides user-friendly terminal output with:
- Emojis for visual scanning
- Progress indicators
- Batch details with product samples
- Category breakdowns
- Final summary statistics

Usage:
    from utils.console import console
    console.start("Pipeline Started")
    console.batch_start(1, 10, [0, 1, 2], ["Product A", "Product B"])
    console.batch_result(10, 10, 2.4, {"Cheese": 4, "Meat": 3})
    console.success("Complete!")

Design principles:
- Isolated from logging (file logs are separate)
- Stateless methods (no side effects beyond printing)
- Easily extensible for new output types
- Configurable via environment variables
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ConsoleConfig:
    """Configuration for console output behavior."""
    colors_enabled: bool = True
    max_product_display: int = 3
    max_product_name_length: int = 35
    max_category_name_length: int = 40
    box_width: int = 60

    @classmethod
    def from_env(cls) -> "ConsoleConfig":
        """Load configuration from environment variables."""
        return cls(
            colors_enabled=os.getenv("CONSOLE_COLORS", "true").lower() == "true",
            max_product_display=int(os.getenv("CONSOLE_MAX_PRODUCTS", "3")),
            max_product_name_length=int(os.getenv("CONSOLE_MAX_PRODUCT_LEN", "35")),
        )


class Console:
    """Pretty console output handler for pipeline operations.
    
    Provides methods for different types of output:
    - Phase indicators (start, success, error, warning)
    - Progress tracking (batch progress, overall progress)
    - Data display (products, categories, statistics)
    
    All output goes to stdout and is designed to be human-readable.
    For machine-readable logs, use the logging module instead.
    """

    def __init__(self, config: Optional[ConsoleConfig] = None):
        self.config = config or ConsoleConfig.from_env()
        self._batch_times: List[float] = []

    # ==================== Helpers ====================

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _shorten_category(self, category: str) -> str:
        """Shorten category name for display."""
        # Common replacements for readability
        replacements = {
            "Prepared/Processed": "Prep.",
            "Unprepared/Unprocessed": "Unprep.",
            "- Prepared/Processed": "(Prep.)",
            "- Unprepared/Unprocessed": "(Unprep.)",
            "/Substitutes": "/Sub.",
            "Variety Packs": "VP",
            "Vegetables": "Veg.",
            "Beverages": "Bev.",
            "Alcoholic": "Alc.",
            "Non Alcoholic": "Non-Alc.",
        }
        result = category
        for old, new in replacements.items():
            result = result.replace(old, new)
        return self._truncate(result, self.config.max_category_name_length)

    def _print(self, *args, **kwargs) -> None:
        """Print to stdout with flush."""
        print(*args, **kwargs, flush=True)

    # ==================== Phase Indicators ====================

    def start(self, message: str, detail: Optional[str] = None) -> None:
        """Display pipeline/phase start message."""
        self._print(f"\n🚀 {message}")
        if detail:
            self._print(f"   └─ {detail}")

    def success(self, message: str, detail: Optional[str] = None) -> None:
        """Display success message."""
        self._print(f"\n✅ {message}")
        if detail:
            self._print(f"   └─ {detail}")

    def error(self, message: str, detail: Optional[str] = None) -> None:
        """Display error message."""
        self._print(f"\n❌ {message}")
        if detail:
            self._print(f"   └─ {detail}")

    def warning(self, message: str, detail: Optional[str] = None) -> None:
        """Display warning message."""
        self._print(f"\n⚠️  {message}")
        if detail:
            self._print(f"   └─ {detail}")

    def info(self, message: str, detail: Optional[str] = None) -> None:
        """Display info message."""
        self._print(f"\n📋 {message}")
        if detail:
            self._print(f"   └─ {detail}")

    def step(self, message: str, done: bool = False) -> None:
        """Display a step in progress or completed."""
        icon = "✓" if done else "..."
        self._print(f"   └─ {message} {icon}")

    # ==================== Data Loading ====================

    def data_loaded(
        self,
        source: str,
        rows: int,
        columns: int,
        elapsed: Optional[float] = None,
    ) -> None:
        """Display data loading result."""
        time_str = f" ({elapsed:.1f}s)" if elapsed else ""
        self._print(f"\n📥 Data Loaded{time_str}")
        self._print(f"   └─ {source}: {rows} rows, {columns} columns")

    def preprocessing(self, done: bool = False) -> None:
        """Display preprocessing status."""
        if done:
            self._print("📦 Preprocessing... ✓")
        else:
            self._print("📦 Preprocessing...")

    # ==================== Classification ====================

    def classification_start(
        self,
        total_rows: int,
        batch_size: int,
        unclassified: int,
    ) -> None:
        """Display classification start info."""
        num_batches = (unclassified + batch_size - 1) // batch_size
        self._print(f"\n🤖 Classification Starting")
        self._print(f"   └─ {unclassified} rows → {num_batches} batches of {batch_size}")
        self._batch_times = []

    def batch_start(
        self,
        batch_num: int,
        total_batches: int,
        row_ids: Sequence[int],
        product_names: Sequence[str],
    ) -> None:
        """Display batch start with product samples."""
        row_range = f"{min(row_ids)}-{max(row_ids)}" if row_ids else "?"
        
        # Simple clean header
        self._print(f"\n┌─ Batch {batch_num}/{total_batches} (Rows {row_range})")
        self._print(f"│  Processing {len(product_names)} products...")
        self._print(f"│")

    def batch_result(
        self,
        classified: int,
        requested: int,
        elapsed: float,
        category_counts: Dict[str, int],
        failed: int = 0,
        tokens: int = 0,
        product_results: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Display batch classification results with product→segment mapping."""
        self._batch_times.append(elapsed)
        
        # Show each product with its assigned segment
        if product_results:
            for item in product_results:
                product = self._truncate(item.get("product", ""), 45)
                segment = item.get("segment", "(unclassified)")
                self._print(f"│  • {product:<45} → {segment}")
        
        # Status line
        self._print(f"│")
        tokens_str = f", {tokens:,} tokens" if tokens > 0 else ""
        if failed > 0:
            self._print(f"│  ⚠️  {classified}/{requested} classified in {elapsed:.1f}s{tokens_str} ({failed} failed)")
        else:
            self._print(f"│  ✓ {classified}/{requested} classified in {elapsed:.1f}s{tokens_str}")
        
        # Box footer
        self._print(f"└{'─' * 60}")

    def progress_bar(
        self,
        current: int,
        total: int,
        width: int = 30,
        label: str = "Progress",
    ) -> None:
        """Display a progress bar."""
        pct = current / total if total > 0 else 0
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        self._print(f"\n📊 {label}: [{bar}] {current}/{total} ({pct*100:.0f}%)")

    # ==================== Final Summary ====================

    def classification_summary(
        self,
        total_rows: int,
        classified: int,
        unique_categories: int,
        total_categories: int,
        top_categories: List[Dict[str, Any]],
        unexpected: List[str],
        output_path: str,
        elapsed: Optional[float] = None,
    ) -> None:
        """Display final classification summary."""
        coverage_pct = (classified / total_rows * 100) if total_rows > 0 else 0
        
        self._print(f"\n✅ Classification Complete!")
        self._print(f"   ├─ Classified: {classified}/{total_rows} ({coverage_pct:.1f}%)")
        self._print(f"   ├─ Categories used: {unique_categories} of {total_categories}")
        
        # Top categories
        if top_categories:
            self._print(f"   ├─ Top 5:")
            for i, cat in enumerate(top_categories[:5], 1):
                name = self._shorten_category(cat["category"])
                self._print(f"   │    {i}. {name:<35} {cat['count']:>4} ({cat['pct']:.1f}%)")
        
        # Unexpected categories
        if unexpected:
            self._print(f"   ├─ ⚠️  Unexpected: {len(unexpected)} → {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        
        # Output location
        self._print(f"   └─ Saved: {output_path}")
        
        # Timing
        if elapsed or self._batch_times:
            total_time = elapsed or sum(self._batch_times)
            avg_time = sum(self._batch_times) / len(self._batch_times) if self._batch_times else 0
            self._print(f"\n⏱️  Total: {total_time:.1f}s", end="")
            if avg_time > 0:
                self._print(f" (avg {avg_time:.1f}s/batch)")
            else:
                self._print()

    # ==================== Pipeline Status ====================

    def pipeline_finished(self, success: bool = True) -> None:
        """Display pipeline completion status."""
        if success:
            self._print(f"\n{'─' * 50}")
            self._print("🎉 Pipeline finished successfully!")
            self._print(f"{'─' * 50}\n")
        else:
            self._print(f"\n{'─' * 50}")
            self._print("💥 Pipeline failed!")
            self._print(f"{'─' * 50}\n")

    def interrupted(self) -> None:
        """Display interruption message."""
        self._print("\n\n⚡ Interrupted by user")
        self._print("   └─ Partial results may have been saved")

    # ==================== GS1 Pipeline — Verbose Output ====================

    def gs1_rag_details(
        self,
        rag_details: List[Dict[str, Any]],
    ) -> None:
        """Display RAG retrieval details per product.

        Each item in rag_details:
            row_id, product_name, num_raw_results, top_scores, matched_areas
        """
        self._print(f"│")
        self._print(f"│  🔍 RAG Retrieval")
        for item in rag_details:
            rid = item.get("row_id", "?")
            pname = self._truncate(item.get("product_name", ""), 30)
            n_raw = item.get("num_raw_results", 0)
            top_scores = item.get("top_scores", [])
            areas = item.get("matched_areas", [])

            scores_str = ", ".join(f"{s:.3f}" for s in top_scores[:5])
            self._print(f"│  ┌─ Row {rid}: {pname}")
            self._print(f"│  │  Results: {n_raw} hits │ Top scores: [{scores_str}]")
            if areas:
                for area in areas[:4]:
                    self._print(f"│  │  ▸ {area}")
                if len(areas) > 4:
                    self._print(f"│  │  ▸ ...+{len(areas) - 4} more areas")
            self._print(f"│  └─")

    def gs1_candidates(
        self,
        candidates_by_product: List[Dict[str, Any]],
    ) -> None:
        """Display candidate options per product.

        Each item: row_id, product_name, candidates (list of letter, display_text, score)
        """
        self._print(f"│")
        self._print(f"│  📋 Candidate Options")
        for item in candidates_by_product:
            rid = item.get("row_id", "?")
            pname = self._truncate(item.get("product_name", ""), 30)
            candidates = item.get("candidates", [])
            self._print(f"│  ┌─ Row {rid}: {pname} ({len(candidates)} candidates)")
            for c in candidates:
                letter = c.get("letter", "?")
                display = self._truncate(c.get("display_text", ""), 75)
                score = c.get("best_score", 0.0)
                self._print(f"│  │  [{letter}] {display}  ({score:.3f})")
            self._print(f"│  └─")

    def gs1_prompt(self, prompt_text: str) -> None:
        """Display the full prompt being sent to the LLM."""
        self._print(f"│")
        self._print(f"│  📤 LLM Prompt ({len(prompt_text):,} chars)")
        self._print(f"│  ┌{'─' * 58}┐")
        for line in prompt_text.splitlines():
            truncated = self._truncate(line, 56)
            self._print(f"│  │ {truncated:<56} │")
        self._print(f"│  └{'─' * 58}┘")

    def gs1_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        """Display token usage breakdown."""
        self._print(f"│")
        self._print(
            f"│  🪙 Tokens: {total_tokens:,} total "
            f"(prompt: {prompt_tokens:,}, completion: {completion_tokens:,})"
        )

    def gs1_db_write(self, updates: List[Dict[str, Any]]) -> None:
        """Display DB write confirmation per row."""
        self._print(f"│")
        self._print(f"│  💾 DB Writes ({len(updates)} rows)")
        for u in updates:
            rid = u.get("row_id", "?")
            seg = u.get("gs1_segment", "?")
            fam = u.get("gs1_family", "?")
            cls_ = u.get("gs1_class", "?")
            brk = u.get("gs1_brick", "?")
            attr = u.get("gs1_attribute", "NONE")
            attrv = u.get("gs1_attribute_value", "NONE")
            path = f"{seg} > {fam} > {cls_} > {brk}"
            attr_str = ""
            if attr != "NONE":
                attr_str = f" | {attr}"
                if attrv != "NONE":
                    attr_str += f" > {attrv}"
            self._print(f"│  ✓ Row {rid}: {self._truncate(path + attr_str, 65)}")

    def gs1_timing_breakdown(
        self,
        rag_time: float,
        llm_time: float,
        db_time: float,
        total_time: float,
    ) -> None:
        """Display per-batch timing breakdown."""
        self._print(f"│")
        bar_width = 30
        parts = [
            ("RAG", rag_time),
            ("LLM", llm_time),
            ("DB", db_time),
        ]
        other = max(0, total_time - rag_time - llm_time - db_time)

        self._print(f"│  ⏱  Timing: {total_time:.2f}s total")
        for label, t in parts:
            pct = (t / total_time * 100) if total_time > 0 else 0
            filled = int(bar_width * t / total_time) if total_time > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            self._print(f"│     {label:<4} [{bar}] {t:.2f}s ({pct:.0f}%)")
        if other > 0.01:
            pct = (other / total_time * 100) if total_time > 0 else 0
            self._print(f"│     Other: {other:.2f}s ({pct:.0f}%)")

    def gs1_running_totals(
        self,
        batch_num: int,
        total_batches: int,
        total_classified: int,
        total_unclassified: int,
        total_updated: int,
        pipeline_elapsed: float,
    ) -> None:
        """Display running totals and progress after each batch."""
        remaining = total_unclassified - total_updated
        pct = (total_updated / total_unclassified * 100) if total_unclassified > 0 else 0
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        avg_time = pipeline_elapsed / batch_num if batch_num > 0 else 0
        remaining_batches = total_batches - batch_num
        eta = avg_time * remaining_batches

        self._print(f"│")
        self._print(f"│  📊 Progress: [{bar}] {total_updated}/{total_unclassified} ({pct:.1f}%)")
        self._print(
            f"│     Batches: {batch_num}/{total_batches} │ "
            f"Remaining: {remaining} rows │ "
            f"Avg: {avg_time:.1f}s/batch │ "
            f"ETA: {eta:.0f}s"
        )
        self._print(f"└{'─' * 60}")


# ==================== Singleton Instance ====================
# This allows: from utils.console import console
console = Console()

__all__ = ["Console", "ConsoleConfig", "console"]
