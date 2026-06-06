"""Runtime items passed between local pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from clover.runtime.task import RuntimeTaskItem


TaskItemT = TypeVar("TaskItemT", bound=RuntimeTaskItem)


@dataclass
class RuntimeCommandItem(Generic[TaskItemT]):
    """One task paired with a command emitted for local lowering."""

    task: TaskItemT
    content: str
    content_type: str


@dataclass
class RuntimeWorkItem(Generic[TaskItemT]):
    """One task paired with a parsed local work artifact."""

    task: TaskItemT
    command_output: str
    output_type: str
    logic_dag: dict[str, Any]


@dataclass
class RuntimeObservationItem(Generic[TaskItemT]):
    """One task paired with an observation returned by local execution."""

    task: TaskItemT
    observation: Any
    observation_type: str = "observation"
