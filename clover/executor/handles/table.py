"""Table handles exposed inside Agent sandboxes."""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from clover.executor.handles.base import SandboxHandle
from clover.tools.table_reasoning.pandas_backend import PandasTable


class TableHandle(SandboxHandle):
    """Agent-facing table value that keeps frame data and metadata together."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        group_keys: list[dict[str, Any]] | None = None,
        resource_id: str | None = None,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_copy = copy.deepcopy(metadata or {})
        if group_keys is not None:
            metadata_copy["group_keys"] = copy.deepcopy(group_keys)
        super().__init__(
            resource_id=resource_id,
            role=role,
            metadata=metadata_copy,
        )
        self.frame = copy_frame(frame)
        self.group_keys = copy.deepcopy(metadata_copy.get("group_keys") or [])

    def copy(self) -> "TableHandle":
        return TableHandle(
            self.frame,
            group_keys=self.group_keys,
            resource_id=self.resource_id,
            role=self.role,
            metadata=self.metadata,
        )

    def with_frame(
        self,
        frame: pd.DataFrame,
        *,
        group_keys: list[dict[str, Any]] | None = None,
    ) -> "TableHandle":
        return TableHandle(
            frame,
            group_keys=self.group_keys if group_keys is None else group_keys,
            resource_id=self.resource_id,
            role=self.role,
            metadata=self.metadata,
        )

    def to_pandas(self, *, copy_frame: bool = True) -> pd.DataFrame:
        if not copy_frame:
            return self.frame
        copied = self.frame.copy()
        copied.attrs = {}
        return copied

    def to_pandas_table(self) -> PandasTable:
        return PandasTable(
            copy_frame(self.frame),
            group_keys=copy.deepcopy(self.group_keys),
        )

    def close(self) -> None:
        self.frame = pd.DataFrame()
        self.group_keys = []
        self.metadata.clear()

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self) -> Any:
        return iter(self.frame)

    def __contains__(self, key: object) -> bool:
        return key in self.frame

    def __getitem__(self, key: Any) -> Any:
        return self.frame.__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.frame.__setitem__(key, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.frame, name)

    def __repr__(self) -> str:
        return (
            f"TableHandle(rows={len(self.frame)}, "
            f"columns={list(self.frame.columns)!r}, "
            f"group_keys={self.group_keys!r})"
        )


def copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied.attrs = {}
    return copied
