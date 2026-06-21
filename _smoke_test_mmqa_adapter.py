"""Smoke test for MMQA adapter: load one case, build DSL, preprocess."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.mmqa.adapter import (
    MMQA_DSL_MODE_BUILDER_AGENT,
    first_mmqa_dataset,
    iter_mmqa_dataset_dirs,
    load_mmqa_task,
    run_mmqa_case,
)


def main() -> None:
    mmqa_root = Path("/tmp/mmqa_full")

    print("=== dataset dirs ===")
    dirs = iter_mmqa_dataset_dirs(mmqa_root)
    print(f"total datasets: {len(dirs)}")
    first_id = first_mmqa_dataset(mmqa_root)
    print(f"first dataset_id: {first_id}")

    print("\n=== load_mmqa_task (case_index=0) ===")
    task = load_mmqa_task(
        mmqa_root=mmqa_root,
        dataset_id=first_id,
        case_index=0,
        dsl_builder_slm_config={},
    )
    print(f"case_id: {task.metadata['case_id']}")
    print(f"question: {task.metadata['case']['question']}")
    print(f"table_count: {task.metadata['table_count']}")
    print(f"source_files: {task.metadata['source_files']}")
    print(f"table_names: {task.metadata['table_names']}")
    print(f"foreign_keys: {task.metadata['foreign_keys']}")
    print(f"primary_keys: {task.metadata['primary_keys']}")
    print(f"answer_type: {task.metadata['answer_type']}")
    print(f"expected_answer: {task.metadata['expected_answer']}")
    print(f"dsl_builder mode: {task.metadata['dsl_builder']['mode']}")
    print(f"task_dsl:")
    print(json.dumps(task.task_dsl, ensure_ascii=False, indent=2))

    print("\n=== run_mmqa_case (preprocess) ===")
    summary = run_mmqa_case(
        mmqa_root=mmqa_root,
        dataset_id=first_id,
        case_id=None,
        case_index=0,
        output_root=Path("/tmp/mmqa_run"),
        run_name="smoke",
        dsl_builder_slm_config={},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Verify preprocess output
    case_dir = Path(summary["case_dir"])
    local_dsl = json.loads((case_dir / "local_dsl.json").read_text())
    remote_dsl = json.loads((case_dir / "remote_dsl.json").read_text())
    print(f"\nlocal_dsl sources: {len(local_dsl['sources'])}")
    for src in local_dsl["sources"]:
        print(f"  id={src['id']} file={src['file']} cols={len(src['schema'].get('columns', []))}")
    print(f"remote_dsl sources: {len(remote_dsl['sources'])}")
    print(f"local_dsl hints: {local_dsl.get('hints', {}).keys()}")
    print(f"task_type: {local_dsl['task_type']}")

    assert task.metadata["dsl_builder"]["mode"] == MMQA_DSL_MODE_BUILDER_AGENT
    assert len(local_dsl["sources"]) == task.metadata["table_count"]
    assert local_dsl["task_type"] == "table_reasoning.query"
    assert local_dsl.get("hints", {}).get("foreign_keys") is not None or not task.metadata["foreign_keys"]

    print("\n=== All adapter tests passed ===")


if __name__ == "__main__":
    main()
