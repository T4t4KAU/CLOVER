"""Sandbox-visible resource handles."""

from clover.executor.handles.base import SandboxHandle, ValueHandle
from clover.executor.handles.projector import SandboxProjector
from clover.executor.handles.table import TableHandle

__all__ = [
    "SandboxHandle",
    "SandboxProjector",
    "TableHandle",
    "ValueHandle",
]
