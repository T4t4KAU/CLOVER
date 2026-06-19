"""TableFact adapter built on CLOVER's TableBench-compatible table layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.tablebench.adapter import TablebenchTask, load_tablebench_task


def load_tablefact_task(
    tablefact_root: str | Path,
    dataset_id: str,
    case_id: str | None = None,
    case_index: int | None = None,
    dsl_builder_slm_config: dict[str, Any] | None = None,
    dsl_builder_client: Any | None = None,
) -> TablebenchTask:
    """Load one converted TableFact case as a boolean table reasoning task."""

    task = load_tablebench_task(
        tablebench_root=tablefact_root,
        dataset_id=dataset_id,
        case_id=case_id,
        case_index=case_index,
        dsl_builder_slm_config=dsl_builder_slm_config,
        dsl_builder_client=dsl_builder_client,
    )
    metadata = dict(task.metadata)
    metadata["dataset"] = "tablefact"
    return TablebenchTask(
        task_dsl=task.task_dsl,
        base_dir=task.base_dir,
        metadata=metadata,
    )

