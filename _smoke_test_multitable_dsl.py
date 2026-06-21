"""Smoke test for BuildMultiTableDSLTool."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from clover.resource import (
    BUILD_MULTITABLE_DSL_TOOL_NAME,
    MULTITABLE_BUILDER_AGENT_MODE,
    BuildMultiTableDSLTool,
    build_multitable_task_dsl_with_builder_agent,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        table1 = tmpdir / "table_1.csv"
        table1.write_text(
            "Department_ID,Name,Num_Employees\n1,State,30266\n2,Treasury,115897\n",
            encoding="utf-8",
        )
        table2 = tmpdir / "table_2.csv"
        table2.write_text(
            "department_ID,head_ID,temporary_acting\n2,5,Yes\n1,3,No\n",
            encoding="utf-8",
        )

        result = BuildMultiTableDSLTool().run(
            question="Which department headed by a temporary acting manager has the most employees?",
            table_paths=[table1, table2],
            source_files=["table_1.csv", "table_2.csv"],
            answer_type="string",
            table_names={"table_1": "department", "table_2": "management"},
            foreign_keys=["head_ID", "department_ID"],
            primary_keys=["Department_ID", "head_ID"],
        )
        print("=== Test 1: BuildMultiTableDSLTool.run ===")
        print(f"builder_mode: {result.builder_mode}")
        print(f"tool_call: {json.dumps(result.tool_call, ensure_ascii=False)}")
        print(f"task_dsl:\n{json.dumps(result.task_dsl, ensure_ascii=False, indent=2)}")
        print(f"diagnostics keys: {list(result.diagnostics.keys())}")
        print(f"table_count: {result.diagnostics['table_count']}")
        print(f"profiles count: {len(result.diagnostics['profiles'])}")
        print()

        result2 = build_multitable_task_dsl_with_builder_agent(
            question="Which department headed by a temporary acting manager has the most employees?",
            table_paths=[table1, table2],
            source_files=["table_1.csv", "table_2.csv"],
            answer_type="string",
            table_names={"table_1": "department", "table_2": "management"},
            foreign_keys=["head_ID", "department_ID"],
            primary_keys=["Department_ID", "head_ID"],
            slm_config={},
        )
        print("=== Test 2: build_multitable_task_dsl_with_builder_agent ===")
        print(f"builder_mode: {result2.builder_mode}")
        print(f"task_dsl == result.task_dsl: {result2.task_dsl == result.task_dsl}")
        print()

        print("=== Test 3: constants ===")
        print(f"BUILD_MULTITABLE_DSL_TOOL_NAME: {BUILD_MULTITABLE_DSL_TOOL_NAME}")
        print(f"MULTITABLE_BUILDER_AGENT_MODE: {MULTITABLE_BUILDER_AGENT_MODE}")
        print()

        print("=== Test 4: error handling ===")
        try:
            BuildMultiTableDSLTool().run(
                question="test",
                table_paths=[],
                source_files=[],
            )
            print("ERROR: should have raised")
        except ValueError as exc:
            print(f"empty tables raises: {exc}")

        try:
            BuildMultiTableDSLTool().run(
                question="test",
                table_paths=[table1],
                source_files=["table_1.csv", "table_2.csv"],
            )
            print("ERROR: should have raised")
        except ValueError as exc:
            print(f"length mismatch raises: {exc}")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
