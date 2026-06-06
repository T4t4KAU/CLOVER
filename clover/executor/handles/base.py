"""Generic handles exposed inside Agent sandboxes."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxHandle:
    """Base class for sandbox-visible resource projections."""

    resource_id: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "resource_id": self.resource_id,
            "role": self.role,
            "metadata": copy.deepcopy(self.metadata),
        }

    def close(self) -> None:
        """Release handle-owned references."""


class ValueHandle(SandboxHandle):
    """Handle for scalar or JSON-like values."""

    def __init__(
        self,
        value: Any,
        *,
        resource_id: str | None = None,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            resource_id=resource_id,
            role=role,
            metadata=copy.deepcopy(metadata or {}),
        )
        self.value = copy.deepcopy(value)

    def copy(self) -> "ValueHandle":
        return ValueHandle(
            self.value,
            resource_id=self.resource_id,
            role=self.role,
            metadata=self.metadata,
        )

    def unwrap(self) -> Any:
        return copy.deepcopy(self.value)

    def close(self) -> None:
        self.value = None

    def __repr__(self) -> str:
        return f"ValueHandle(value={self.value!r})"
