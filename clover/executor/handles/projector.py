"""Project executor resources into sandbox-visible handles."""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from clover.executor.handles.base import SandboxHandle, ValueHandle
from clover.executor.handles.table import TableHandle
from clover.tools.table_reasoning.pandas_backend import PandasTable


class SandboxProjector:
    """Generic ResourceObject -> SandboxHandle projector."""

    target = "sandbox"

    def project(
        self,
        resource: Any,
        *,
        role: str,
    ) -> SandboxHandle | Any:
        value = resource.materialize(target="pandas")
        metadata = copy.deepcopy(getattr(resource, "metadata", {}))
        if getattr(resource, "kind", None) == "table":
            return _table_handle(
                value,
                resource_id=getattr(resource, "id", None),
                role=role,
                metadata=metadata,
            )
        return ValueHandle(
            value,
            resource_id=getattr(resource, "id", None),
            role=role,
            metadata=metadata,
        )


def _table_handle(
    value: Any,
    *,
    resource_id: str | None,
    role: str,
    metadata: dict[str, Any],
) -> TableHandle:
    if isinstance(value, TableHandle):
        return value.copy()
    if isinstance(value, PandasTable):
        return TableHandle(
            value.frame,
            group_keys=value.group_keys,
            resource_id=resource_id,
            role=role,
            metadata=metadata,
        )
    if isinstance(value, pd.DataFrame):
        return TableHandle(
            value,
            group_keys=metadata.get("group_keys"),
            resource_id=resource_id,
            role=role,
            metadata=metadata,
        )
    frame = getattr(value, "frame", None)
    if isinstance(frame, pd.DataFrame):
        return TableHandle(
            frame,
            group_keys=getattr(value, "group_keys", metadata.get("group_keys", [])),
            resource_id=resource_id,
            role=role,
            metadata=metadata,
        )
    raise TypeError(f"Cannot project table resource from {type(value).__name__}")
