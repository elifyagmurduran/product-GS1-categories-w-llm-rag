"""Custom exceptions for the classifier pipeline."""


class PipelineError(Exception):
    """Base exception for pipeline errors. Raise this instead of sys.exit(1)."""
    pass
