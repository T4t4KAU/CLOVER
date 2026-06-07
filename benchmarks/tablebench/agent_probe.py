"""Probe table SLM Agent code completion on failed TableBench cases.

This utility isolates the existing table sandbox Agent Loop from the full
runtime. It replays a failed SQL-derived plan until the first empty Filter
node, then asks the local SLM to fill `solve(...)` for that node.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.tablebench.metrics import score_tablebench_answer
from clover.executor.agents.base import FastPathDecision
from clover.executor.agents.table_reasoning import _extract_action_json
from clover.executor.agents.template_tree import (
    render_agent_loop_prompt,
    template_leaf_key_for_local_slm_prompt,
)
from clover.executor.context import NodeExecutionContext
from clover.executor.local_slm import load_slm_config
from clover.executor.resources import ResourceStore
from clover.executor.result import json_ready, summarize_output
from clover.executor.sandbox.core import AgentSandbox
from clover.executor.sandbox.table_reasoning import TableReasoningSandboxPolicy
from clover.executor.slm_dispatcher import (
    LocalSlmSequenceDispatcher,
    LocalSlmSequenceRequest,
)
from clover.optimizer.table_reasoning.sql_parser import parse_remote_sql_to_logic_dag
from clover.tools import build_static_tool_call
from clover.tools.table_reasoning.pandas_backend import (
    PandasTable,
    PandasTableReasoningExecutor,
)


PROMPT_KIND = "table_reasoning_agent_loop"


def main() -> None:
    args = _parse_args()
    selected_cases = _select_cases(
        args.run_dir,
        max_cases=args.max_cases,
        case_ids=set(args.case_id or ()),
        only_empty_answer=not args.include_non_empty_answer,
        case_filter=args.case_filter,
    )
    if not selected_cases:
        raise SystemExit("No matching failed cases found.")

    slm_config = load_slm_config(args.slm_config)
    if args.model:
        slm_config["model"] = args.model
    if args.base_url:
        slm_config["base_url"] = args.base_url
    if args.max_tokens is not None:
        slm_config["max_tokens"] = args.max_tokens
    if args.timeout is not None:
        slm_config["timeout"] = args.timeout
    slm_config["temperature"] = args.temperature

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatcher = LocalSlmSequenceDispatcher(
        slm_config=slm_config,
        max_parallel_sequences=1,
        max_pending_sequences=16,
        slm_scheduler="fifo",
    )
    records: list[dict[str, Any]] = []
    try:
        for case_dir in selected_cases:
            for prompt_mode in args.prompt_mode:
                record = probe_case(
                    case_dir,
                    prompt_mode=prompt_mode,
                    dispatcher=dispatcher,
                    slm_config=slm_config,
                    max_iterations=args.max_iterations,
                )
                records.append(record)
                _write_json(
                    output_dir / f"{case_dir.name}__{prompt_mode}.json",
                    record,
                )
                _print_record(record)
    finally:
        dispatcher.close()

    summary = _summarize(records)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def probe_case(
    case_dir: Path,
    *,
    prompt_mode: str,
    dispatcher: LocalSlmSequenceDispatcher,
    slm_config: dict[str, Any],
    max_iterations: int,
) -> dict[str, Any]:
    case_result = _read_json(case_dir / "case_result.json")
    local_dsl = _read_json(case_dir / "local_dsl.json")
    remote_dsl = _read_json(case_dir / "remote_dsl.json")
    sql = _extract_case_sql(case_result)
    dag = parse_remote_sql_to_logic_dag(sql, remote_dsl)
    plan = dag["query_plans"][0]
    try:
        prepared = _prepare_empty_filter_target(local_dsl, plan["nodes"])
    except Exception as exc:  # noqa: BLE001 - diagnostic probe should keep going.
        return {
            "case_dir": case_dir.name,
            "dataset_id": case_result.get("dataset_id"),
            "case_id": case_result.get("case_id"),
            "prompt_mode": prompt_mode,
            "ok": False,
            "skipped": True,
            "skip_reason": f"prepare_error:{type(exc).__name__}: {exc}",
            "sql": sql,
        }
    if prepared is None:
        return {
            "case_dir": case_dir.name,
            "prompt_mode": prompt_mode,
            "ok": False,
            "skipped": True,
            "skip_reason": "no_empty_filter_target",
            "sql": sql,
        }

    target_node = prepared["target_node"]
    context, store = _make_node_context(
        local_dsl=local_dsl,
        node=target_node,
        dependency_outputs=prepared["dependency_outputs"],
        slm_config=slm_config,
        dispatcher=dispatcher,
        max_iterations=max_iterations,
    )
    sandbox = AgentSandbox(context, TableReasoningSandboxPolicy())
    observations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    output = None
    accepted = False
    error: str | None = None
    try:
        sandbox.start(
            decision=FastPathDecision(hit=True, tool="filter", backend="pandas"),
            trigger="fast_path_empty_output",
            error=None,
        )
        leaf_key = template_leaf_key_for_local_slm_prompt(
            prompt_kind=PROMPT_KIND,
            task_type=context.task_type,
            node=target_node,
        )
        for iteration in range(1, max_iterations + 1):
            view = sandbox.view(observations)
            prompt = _render_probe_prompt(
                task_type=context.task_type,
                view=view,
                iteration=iteration,
                prompt_mode=prompt_mode,
                steps=steps,
            )
            sequence = dispatcher.generate(
                LocalSlmSequenceRequest(
                    prompt=prompt,
                    leaf_key=leaf_key,
                    prompt_kind=PROMPT_KIND,
                    node_id=str(target_node.get("id") or ""),
                    job_id=f"{case_dir.name}__{prompt_mode}",
                    iteration=iteration,
                    slm_config=slm_config,
                )
            )
            response_text = sequence.text
            try:
                action = _extract_action_json(response_text)
            except Exception as exc:  # noqa: BLE001 - diagnostic probe.
                observation = {
                    "type": "invalid_action_json",
                    "ok": False,
                    "error": {"message": str(exc)},
                }
                observations.append(observation)
                steps.append(
                    {
                        "iteration": iteration,
                        "prompt_chars": len(prompt),
                        "response_text": response_text,
                        "action": None,
                        "observation": observation,
                        "accepted": False,
                    }
                )
                continue

            action_result = sandbox.run_action(action)
            observation = action_result.observation
            step = {
                "iteration": iteration,
                "prompt_chars": len(prompt),
                "response_text": response_text,
                "action": _compact_action(action),
                "observation": json_ready(observation),
                "accepted": action_result.accepted,
                "output_summary": (
                    json_ready(summarize_output(action_result.output))
                    if action_result.accepted
                    else None
                ),
                "sequence": sequence.trace_metadata(),
            }
            steps.append(step)
            if action_result.accepted:
                accepted = True
                output = action_result.output
                break
            if observation is not None:
                observations.append(observation)
    except Exception as exc:  # noqa: BLE001 - diagnostic probe.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sandbox.close()
        store.close_all()

    final_answer = None
    final_error = None
    final_score = None
    if accepted:
        try:
            final_answer = _finish_plan_after_target(
                local_dsl=local_dsl,
                nodes=plan["nodes"],
                target_node=target_node,
                artifacts_before=prepared["artifacts_before"],
                target_output=output,
            )
            score = score_tablebench_answer(
                expected=case_result.get("expected_raw"),
                actual=final_answer,
                qtype=case_result.get("qtype"),
                qsubtype=case_result.get("qsubtype"),
            )
            final_score = {
                "metric": score.metric,
                "score": score.score,
                "expected": score.expected,
                "actual": score.actual,
            }
        except Exception as exc:  # noqa: BLE001 - diagnostic probe.
            final_error = f"{type(exc).__name__}: {exc}"

    return {
        "case_dir": case_dir.name,
        "dataset_id": case_result.get("dataset_id"),
        "case_id": case_result.get("case_id"),
        "prompt_mode": prompt_mode,
        "ok": accepted and final_error is None,
        "accepted": accepted,
        "iterations": len(steps),
        "question": local_dsl.get("question"),
        "qtype": case_result.get("qtype"),
        "qsubtype": case_result.get("qsubtype"),
        "sql": sql,
        "target_node": target_node,
        "fast_path_empty_summary": prepared["empty_summary"],
        "final_answer": json_ready(final_answer),
        "final_error": final_error,
        "final_score": final_score,
        "loop_error": error,
        "steps": steps,
    }


def _prepare_empty_filter_target(
    local_dsl: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    resources = {str(source["id"]): source for source in local_dsl.get("sources", [])}
    external_params = {"question": local_dsl.get("question")}
    executor = PandasTableReasoningExecutor(
        resources=resources,
        external_params=external_params,
    )
    artifacts: dict[str, Any] = {}
    for node in nodes:
        call = build_static_tool_call(
            local_dsl["task_type"],
            node,
            resources=resources,
            upstream_outputs=artifacts,
            external_params=external_params,
        )
        call["upstream_outputs"] = dict(artifacts)
        output = executor.execute_call(call)
        if node.get("op") == "Filter" and _is_empty_table(output):
            return {
                "target_node": node,
                "dependency_outputs": {
                    name: artifacts[name]
                    for name in node.get("dependency", [])
                    if name in artifacts
                },
                "artifacts_before": dict(artifacts),
                "empty_summary": json_ready(summarize_output(output)),
            }
        output_name = node.get("output")
        if isinstance(output_name, str):
            artifacts[output_name] = output
    return None


def _make_node_context(
    *,
    local_dsl: dict[str, Any],
    node: dict[str, Any],
    dependency_outputs: dict[str, Any],
    slm_config: dict[str, Any],
    dispatcher: LocalSlmSequenceDispatcher,
    max_iterations: int,
) -> tuple[NodeExecutionContext, ResourceStore]:
    store = ResourceStore(table_cache={})
    for name, value in dependency_outputs.items():
        store.put_artifact(name, value, retained=True)
    view = store.node_view(
        node_id=str(node.get("id") or ""),
        dependencies=list(node.get("dependency", [])),
        sources=list(node.get("input", [])),
    )
    view.pin()
    context = NodeExecutionContext(
        task_type=str(local_dsl["task_type"]),
        node=node,
        resource_view=view,
        external_params={"question": local_dsl.get("question")},
        table_cache={},
        slm_config=slm_config,
        slm_dispatcher=dispatcher,
        agent_loop_max_iterations=max_iterations,
    )
    return context, store


def _render_probe_prompt(
    *,
    task_type: str,
    view: Any,
    iteration: int,
    prompt_mode: str,
    steps: list[dict[str, Any]],
) -> str:
    if prompt_mode == "repair_state_template":
        return _render_repair_state_template_prompt(view=view, steps=steps)
    if prompt_mode == "zero_row_literal_repair":
        return _render_zero_row_literal_repair_prompt(view=view, steps=steps)

    base = render_agent_loop_prompt(
        task_type=task_type,
        view=view,
        iteration=iteration,
    )
    if prompt_mode == "original":
        return base
    if prompt_mode not in {"repair_hint", "repair_hint_strong"}:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    extra = [
        "",
        "Repair instructions:",
        "- This is a repair task after an automatic table operation returned empty output.",
        "- Do not restart blindly. Use the previous code and the observation to change the failing logic.",
        "- If a text equality filter returned empty rows, do not repeat the same exact equality.",
        "- Normalize text before matching: strip, casefold, collapse whitespace, and ignore spacing around punctuation such as commas and hyphens.",
        "- Use the shown diagnostic values to match the closest real cell value, then return the original rows/columns required by the operation.",
        "- Return only one JSON object with field s containing a complete solve function.",
        "- The code string must contain exactly one top-level function definition: def solve(...).",
        "- Do not include top-level imports, result = solve(...), assertions, or markdown inside the code string.",
        "- Do not use regex strings or backslashes in the JSON code string; they often make invalid JSON. Use map(norm_text) instead.",
    ]
    if steps:
        last = steps[-1]
        action = last.get("action") or {}
        previous_code = action.get("code")
        if previous_code:
            extra.extend(
                [
                    "",
                    "Previous code:",
                    "```python",
                    previous_code,
                    "```",
                ]
            )
        extra.extend(
            [
                "",
                "Previous observation:",
                "```json",
                json.dumps(last.get("observation"), ensure_ascii=False, indent=2),
                "```",
            ]
        )
    if prompt_mode == "repair_hint_strong":
        extra.extend(
            [
                "",
                "For text filters, prefer this exact repair pattern:",
                "```python",
                "def norm_text(x):",
                "    return ''.join(ch for ch in str(x).casefold() if ch.isalnum())",
                "# Apply the same norm_text function to BOTH the table cells and the target literal.",
                "# For a pandas Series, use df[col].astype(str).map(norm_text); do not call norm_text(series).",
                "# This pattern needs no imports.",
                "# Avoid pandas .str.replace(..., regex=True) here; regex backslashes can break JSON.",
                "# Keep filtering on the original df so returned rows preserve original columns and values.",
                "```",
                "If equality still returns no rows, choose rows whose normalized cell contains the normalized target or whose normalized target contains the normalized cell.",
            ]
        )
    return base + "\n" + "\n".join(extra)


def _render_repair_state_template_prompt(
    *,
    view: Any,
    steps: list[dict[str, Any]],
) -> str:
    """Render a compact repair prompt with all dynamic facts at the tail."""

    policy = _repair_state_policy(steps)
    payload = _repair_state_payload(view=view, steps=steps)
    return "\n".join(
        [
            'Return only JSON: {"s":"def solve(...):\\n    ...\\n    return result"}.',
            '"s" must be exactly one top-level def solve function.',
            "No code outside solve. No markdown. No prose.",
            "Use only function arguments plus pd, np, helpers, print.",
            "Repair with the smallest change.",
            "",
            *policy,
            "",
            "Case:",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ]
    )


def _render_zero_row_literal_repair_prompt(
    *,
    view: Any,
    steps: list[dict[str, Any]],
) -> str:
    """Render the narrow zero-row literal repair prompt."""

    payload = _repair_state_payload(view=view, steps=steps)
    policy = _repair_state_policy(steps)
    return "\n".join(
        [
            'Return only JSON: {"s":"def solve(...):\\n    ...\\n    return result"}.',
            '"s" must be exactly one top-level def solve function.',
            "No code outside solve. No markdown. No prose.",
            "Use only function arguments plus pd, np, helpers, print.",
            "The previous table filter returned 0 rows.",
            "Keep the same selected columns and output shape.",
            "If the filtered column exists, keep that column.",
            "Repair only the matching logic unless the column is absent.",
            "Use same-column evidence values; do not invent a new literal.",
            "For text literals, normalize both cell and target:",
            "def norm_text(x): return ''.join(ch for ch in str(x).casefold() if ch.isalnum())",
            "Try normalized equality first; if needed, same-column containment.",
            "",
            *policy,
            "",
            "Case:",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ]
    )


def _repair_state_policy(steps: list[dict[str, Any]]) -> list[str]:
    observation = _last_observation(steps)
    observation_type = observation.get("type") if isinstance(observation, dict) else None
    error = observation.get("error") if isinstance(observation, dict) else None
    error_message = ""
    if isinstance(error, dict):
        error_message = str(error.get("message") or error.get("msg") or "")

    if observation_type == "invalid_action_json":
        return [
            "The last answer was not valid JSON.",
            "Return one JSON object with key s.",
            "Avoid regex/backslash strings in JSON.",
        ]
    if observation_type == "invalid_solve_function":
        return [
            "The last code shape was invalid.",
            "Keep only one top-level def solve(...).",
            "Move any import inside solve or avoid imports.",
            "Remove result=solve(...), assertions, and extra statements.",
        ]
    if observation_type == "python_error":
        lines = [
            "The last solve raised an error.",
            "Use the shown error and columns.",
            "Keep only one top-level def solve(...).",
        ]
        if "re" in error_message.lower() or "not defined" in error_message.lower():
            lines.append("Avoid regex/imports; use the no-import normalizer below.")
        return lines + _empty_text_match_policy()
    return [
        "The check returned an empty DataFrame.",
        *_empty_text_match_policy(),
    ]


def _empty_text_match_policy() -> list[str]:
    return [
        "Do not repeat exact text equality.",
        "Do not use raw str.contains on the target text.",
        "Compare normalized cell text and normalized target text.",
        "Use this shape inside solve:",
        "def norm_text(x): return ''.join(ch for ch in str(x).casefold() if ch.isalnum())",
        "target = norm_text(<literal>)  # target must be normalized too",
        "mask = df[<col>].astype(str).map(norm_text).eq(target)",
        "Define norm_text inside solve; do not import it.",
        "Return matching rows from the original df.",
    ]


def _repair_state_payload(
    *,
    view: Any,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    view_payload = view.to_dict() if hasattr(view, "to_dict") else {}
    world = view_payload.get("world") if isinstance(view_payload, dict) else {}
    if not isinstance(world, dict):
        world = {}
    task_code = view_payload.get("task") if isinstance(view_payload, dict) else ""
    task_code = task_code if isinstance(task_code, str) else ""
    observation = _last_observation(steps)
    previous_code = _last_action_code(steps)
    payload: dict[str, Any] = {
        "sig": _extract_solve_signature(task_code),
        "goal": _extract_task_goal(task_code),
        "prev": previous_code,
        "check": _visible_check_message(observation),
        "evidence": _visible_evidence(world=world, observation=observation),
    }
    columns = _visible_columns(world=world, observation=observation)
    if columns:
        payload["cols"] = columns
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }


def _last_observation(steps: list[dict[str, Any]]) -> dict[str, Any]:
    for step in reversed(steps):
        observation = step.get("observation")
        if isinstance(observation, dict):
            return observation
    return {}


def _last_action_code(steps: list[dict[str, Any]]) -> str | None:
    for step in reversed(steps):
        action = step.get("action")
        if isinstance(action, dict):
            code = action.get("code")
            if isinstance(code, str) and code.strip():
                return _compact_code(code)
    return None


def _extract_solve_signature(task_code: str) -> str:
    for line in task_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def solve("):
            return stripped
    return "def solve(df):"


def _extract_task_goal(task_code: str) -> str:
    lines = task_code.splitlines()
    in_docstring = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == '"""':
            if in_docstring:
                break
            in_docstring = True
            continue
        if in_docstring and stripped:
            collected.append(stripped)
    if collected:
        return _truncate_text(" ".join(collected), 220)
    return ""


def _visible_check_message(observation: dict[str, Any]) -> str:
    if not observation:
        return "The previous check returned an empty DataFrame."
    observation_type = observation.get("type")
    error = observation.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("msg")
        if isinstance(message, str) and message:
            return _truncate_text(message, 220)
    if observation_type == "invalid_action_json":
        return "The previous answer was not valid JSON."
    if observation_type == "invalid_solve_function":
        return "The previous code was not a valid solve function."
    return "The previous check failed."


def _visible_evidence(
    *,
    world: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    values: dict[str, list[Any]] = {}
    _merge_value_evidence(values, _diag_value_evidence(world))
    _merge_value_evidence(values, _observation_value_evidence(observation))
    return {
        column: items[:8]
        for column, items in values.items()
        if items
    }


def _diag_value_evidence(world: dict[str, Any]) -> dict[str, list[Any]]:
    diag = world.get("diag")
    if not isinstance(diag, dict):
        return {}
    inputs = diag.get("inputs")
    if not isinstance(inputs, dict):
        return {}
    values: dict[str, list[Any]] = {}
    for input_diag in inputs.values():
        if not isinstance(input_diag, dict):
            continue
        input_values = input_diag.get("values")
        if not isinstance(input_values, dict):
            continue
        for column, items in input_values.items():
            if not isinstance(items, list):
                continue
            values[str(column)] = [
                item.get("v") if isinstance(item, dict) else item
                for item in items
            ]
    return values


def _observation_value_evidence(observation: dict[str, Any]) -> dict[str, list[Any]]:
    feedback = observation.get("feedback")
    if not isinstance(feedback, dict):
        return {}
    column_values = feedback.get("column_values")
    if not isinstance(column_values, dict):
        return {}
    values: dict[str, list[Any]] = {}
    for column, items in column_values.items():
        if not isinstance(items, list):
            continue
        values[str(column)] = [
            item.get("value") if isinstance(item, dict) else item
            for item in items
        ]
    return values


def _merge_value_evidence(
    target: dict[str, list[Any]],
    source: dict[str, list[Any]],
) -> None:
    for column, items in source.items():
        selected = target.setdefault(column, [])
        for item in items:
            if item is None:
                continue
            if item not in selected:
                selected.append(item)


def _visible_columns(
    *,
    world: dict[str, Any],
    observation: dict[str, Any],
) -> list[str]:
    feedback = observation.get("feedback")
    if isinstance(feedback, dict):
        columns_payload = feedback.get("columns")
        if isinstance(columns_payload, dict):
            columns: list[str] = []
            for items in columns_payload.values():
                if isinstance(items, list):
                    columns.extend(str(item) for item in items)
            if columns:
                return columns[:24]
        if isinstance(columns_payload, list):
            return [str(item) for item in columns_payload[:24]]
    inputs = world.get("inputs")
    if isinstance(inputs, dict) and len(inputs) == 1:
        summary = next(iter(inputs.values()))
        if isinstance(summary, dict) and isinstance(summary.get("cols"), list):
            return [str(item) for item in summary["cols"][:24]]
    return []


def _compact_code(code: str) -> str:
    return _truncate_text(code.strip(), 900)


def _finish_plan_after_target(
    *,
    local_dsl: dict[str, Any],
    nodes: list[dict[str, Any]],
    target_node: dict[str, Any],
    artifacts_before: dict[str, Any],
    target_output: Any,
) -> Any:
    resources = {str(source["id"]): source for source in local_dsl.get("sources", [])}
    external_params = {"question": local_dsl.get("question")}
    executor = PandasTableReasoningExecutor(
        resources=resources,
        external_params=external_params,
    )
    artifacts = dict(artifacts_before)
    target_output_name = target_node.get("output")
    if isinstance(target_output_name, str):
        artifacts[target_output_name] = target_output
    past_target = False
    final_answer = None
    for node in nodes:
        if node is target_node or node.get("id") == target_node.get("id"):
            past_target = True
            continue
        if not past_target:
            continue
        call = build_static_tool_call(
            local_dsl["task_type"],
            node,
            resources=resources,
            upstream_outputs=artifacts,
            external_params=external_params,
        )
        call["upstream_outputs"] = dict(artifacts)
        output = executor.execute_call(call)
        output_name = node.get("output")
        if isinstance(output_name, str):
            artifacts[output_name] = output
        final_answer = output
    return final_answer


def _select_cases(
    run_dir: Path,
    *,
    max_cases: int,
    case_ids: set[str],
    only_empty_answer: bool,
    case_filter: str,
) -> list[Path]:
    cases: list[Path] = []
    for case_result_path in sorted((run_dir / "cases").glob("*/case_result.json")):
        case_dir = case_result_path.parent
        record = _read_json(case_result_path)
        if case_ids and record.get("case_id") not in case_ids and case_dir.name not in case_ids:
            continue
        if record.get("tablebench_score") == 1.0:
            continue
        if only_empty_answer and _standard_text(record.get("final_answer_standard_text")):
            continue
        if not record.get("current_sql"):
            continue
        if case_filter == "all-zero-sql-empty-answer" and not _case_has_all_zero_sql_results(
            run_dir=run_dir,
            case_dir=case_dir,
            case_result=record,
        ):
            continue
        cases.append(case_dir)
        if len(cases) >= max_cases:
            break
    return cases


def _case_has_all_zero_sql_results(
    *,
    run_dir: Path,
    case_dir: Path,
    case_result: dict[str, Any],
) -> bool:
    if _standard_text(case_result.get("final_answer_standard_text")):
        return False
    acts = _extract_sql_acts(case_result.get("current_sql"))
    if not acts:
        return False
    results = [
        _execute_case_sql_row_count(
            run_dir=run_dir,
            case_dir=case_dir,
            sql=str(act.get("q") or ""),
        )
        for act in acts
    ]
    return all(count == 0 for count in results)


def _extract_sql_acts(current_sql: Any) -> list[dict[str, Any]]:
    if not isinstance(current_sql, str) or not current_sql.strip():
        return []
    try:
        payload = json.loads(current_sql)
    except json.JSONDecodeError:
        return [{"op": "sql", "q": current_sql}]
    if not isinstance(payload, dict):
        return []
    sql = payload.get("sql")
    if isinstance(sql, str) and sql.strip():
        return [{"op": "sql", "q": sql}]
    acts = payload.get("acts")
    if not isinstance(acts, list):
        return []
    return [
        act
        for act in acts
        if isinstance(act, dict)
        and act.get("op") == "sql"
        and isinstance(act.get("q"), str)
        and str(act.get("q")).strip()
    ]


def _execute_case_sql_row_count(
    *,
    run_dir: Path,
    case_dir: Path,
    sql: str,
) -> int | None:
    context_path = case_dir / "context.json"
    if not context_path.exists():
        context_path = run_dir / "cases" / case_dir.name / "context.json"
    context = _read_json(context_path)
    source_map = context.get("source_map") if isinstance(context, dict) else None
    if not isinstance(source_map, dict):
        return None
    table_source = source_map.get("table_1")
    if not isinstance(table_source, dict):
        return None
    table_path = table_source.get("path")
    if not isinstance(table_path, str) or not table_path:
        return None
    frame = pd.read_csv(table_path)
    connection = sqlite3.connect(":memory:")
    try:
        frame.to_sql("table_1", connection, index=False, if_exists="replace")
        output = pd.read_sql_query(sql, connection)
        return int(len(output))
    except Exception:
        return None
    finally:
        connection.close()


def _extract_case_sql(case_result: dict[str, Any]) -> str:
    value = case_result.get("current_sql")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case_result has no current_sql")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict):
        sql = payload.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql
        acts = payload.get("acts")
        if isinstance(acts, list):
            for act in acts:
                if isinstance(act, dict) and act.get("op") == "sql":
                    q = act.get("q")
                    if isinstance(q, str) and q.strip():
                        return q
    raise ValueError("Unable to extract SQL from current_sql")


def _is_empty_table(value: Any) -> bool:
    if isinstance(value, PandasTable):
        return value.frame.empty
    frame = getattr(value, "frame", None)
    return bool(getattr(frame, "empty", False))


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    payload = {"action": action.get("action")}
    code = action.get("code")
    if isinstance(code, str):
        payload["code"] = code
    return payload


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, int]] = {}
    for record in records:
        mode = str(record.get("prompt_mode"))
        stats = by_mode.setdefault(
            mode,
            {
                "total": 0,
                "skipped": 0,
                "accepted": 0,
                "final_correct": 0,
                "loop_errors": 0,
            },
        )
        stats["total"] += 1
        if record.get("skipped"):
            stats["skipped"] += 1
            continue
        if record.get("accepted"):
            stats["accepted"] += 1
        if record.get("loop_error"):
            stats["loop_errors"] += 1
        score = record.get("final_score")
        if isinstance(score, dict) and float(score.get("score") or 0.0) >= 1.0:
            stats["final_correct"] += 1
    return {
        "total_records": len(records),
        "by_prompt_mode": by_mode,
    }


def _print_record(record: dict[str, Any]) -> None:
    score = record.get("final_score") or {}
    print(
        json.dumps(
            {
                "case": record.get("case_dir"),
                "mode": record.get("prompt_mode"),
                "accepted": record.get("accepted"),
                "iterations": record.get("iterations"),
                "score": score.get("score"),
                "actual": score.get("actual"),
                "loop_error": record.get("loop_error"),
                "final_error": record.get("final_error"),
                "skipped": record.get("skipped"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _standard_text(value: Any) -> str:
    return str(value or "").strip()


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="TableBench eval run directory containing cases/*/case_result.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for probe records.",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument(
        "--prompt-mode",
        action="append",
        choices=(
            "original",
            "repair_hint",
            "repair_hint_strong",
            "repair_state_template",
            "zero_row_literal_repair",
        ),
        default=[],
    )
    parser.add_argument(
        "--case-filter",
        choices=("failed", "all-zero-sql-empty-answer"),
        default="failed",
        help="Optional extra case filter for targeted probes.",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--slm-config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-non-empty-answer", action="store_true")
    args = parser.parse_args()
    if not args.prompt_mode:
        args.prompt_mode = ["original", "repair_hint"]
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    return args


if __name__ == "__main__":
    main()
