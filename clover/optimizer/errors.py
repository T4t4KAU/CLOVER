"""Optimizer parse error types shared by task-specific parsers."""


class OptimizerParseError(ValueError):
    """Raised when remote decomposition output cannot be lowered into a Logic DAG."""
