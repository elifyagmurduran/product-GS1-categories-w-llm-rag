from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


class JsonManager:
    """Simple JSON I/O with pandas integration and atomic writes."""
    
    def __init__(self, encoding: str = "utf-8", indent: int = 2):
        self.encoding = encoding
        self.indent = indent

    def write(self, path: Path | str, data: Any) -> None:
        """Write data to JSON file atomically.
        
        Uses pandas for DataFrame serialization (which handles NaN properly),
        otherwise writes Python objects directly.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Use pandas native JSON handling for DataFrames
        if isinstance(data, pd.DataFrame):
            tmp = p.with_suffix(p.suffix + ".tmp")
            data.to_json(tmp, orient="records", indent=self.indent, force_ascii=False)
            tmp.replace(p)
        else:
            # For non-DataFrame data, write directly
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(
                pd.io.json.dumps(data, indent=self.indent),
                encoding=self.encoding
            )
            tmp.replace(p)

    def load(self, path: Path | str) -> Any | None:
        """Load JSON file, return None if not exists."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            return pd.read_json(p.read_text(encoding=self.encoding))
        except Exception:
            return None


def validate_classification_output(
    df: pd.DataFrame,
    target_col: str = "classification",
    expected_options: Optional[List[str]] = None,
    top_n: int = 10,
    as_dict: bool = False,
    print_report: bool = False,
) -> Dict[str, Any] | None:
    """Validate classification output and return statistics.
    
    Args:
        df: DataFrame with classification results
        target_col: Name of the column containing classifications
        expected_options: List of valid category options
        top_n: Number of top categories to include in stats
        as_dict: If True, return stats dict; if False, return None
        
    Returns:
        Statistics dict if as_dict=True, else None
    """
    if target_col not in df.columns:
        logger.warning("Target column '%s' not found; cannot validate.", target_col)
        return None

    total_rows = len(df)
    non_null = df[target_col].notna().sum()
    null = total_rows - non_null
    coverage_pct = (non_null / total_rows * 100) if total_rows else 0.0

    value_counts = df[target_col].dropna().value_counts()
    top_freq = value_counts.head(top_n)
    unique_assigned = value_counts.shape[0]

    unexpected_values: List[str] = []
    unused_expected: List[str] = []
    if expected_options:
        assigned_set = set(value_counts.index.tolist())
        expected_set = set(expected_options)
        unexpected_values = sorted(list(assigned_set - expected_set))
        unused_expected = sorted(list(expected_set - assigned_set))

    stats: Dict[str, Any] = {
        "total_rows": total_rows,
        "classified_rows": non_null,
        "unclassified_rows": null,
        "coverage_pct": round(coverage_pct, 2),
        "unique_categories": unique_assigned,
        "top_frequencies": [
            {
                "category": idx,
                "count": int(cnt),
                "pct": round((cnt / non_null * 100) if non_null else 0.0, 2),
            }
            for idx, cnt in top_freq.items()
        ],
    }

    if expected_options:
        stats["unexpected_values"] = unexpected_values
        stats["unused_expected"] = unused_expected

    # Log validation results (detailed, for debugging)
    logger.info(
        "Validation: %d/%d classified (%.1f%%), %d unique categories",
        non_null,
        total_rows,
        coverage_pct,
        unique_assigned,
    )
    if unexpected_values:
        logger.warning("Unexpected categories found: %s", unexpected_values)
    logger.debug(
        "Top categories: %s",
        [(c["category"], c["count"]) for c in stats["top_frequencies"][:5]],
    )
    logger.debug("Unused expected categories: %d", len(unused_expected))

    return stats if as_dict else None


__all__ = ["JsonManager", "validate_classification_output"]
