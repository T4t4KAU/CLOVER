"""Warning handling for benchmark entrypoints."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def suppress_benchmark_warnings() -> Iterator[None]:
    """Silence warnings while running benchmark commands."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
