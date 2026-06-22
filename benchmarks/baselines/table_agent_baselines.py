"""Run ReAcTable and Orchestra-style baselines on CLOVER table datasets.

The implementation intentionally keeps only the runnable core of the two
baselines: prompt construction, SQL/Python table execution, single-model
OpenAI-compatible calls, native CLOVER dataset selection, native scoring, and
run summaries. It does not vendor source datasets, notebooks, venvs, caches, or
third-party framework copies from the original repos.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import re
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.utils import display_path, json_ready, safe_divide, write_jsonl
from clover.config import load_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "runs"

REACTABLE_DEFAULT_TEMPLATE = {
    "prompt_template": (
        "The database table DF is shown as follows:\n{}\n\n"
        "Answer the following question based on the data above: \"{}\". "
        "Generate SQL or Python code step-by-step given the question and table "
        "to answer the question correctly. For each step, generate SQL code to "
        "process the query or Python code to reformat the data. Output the code "
        "braced by \"```\" and an external executor will process the code "
        "generated and feed an intermediate table back to you. Answer the "
        "question directly if confident."
    ),
    "intermediate_prompt_template": {
        "SQL": "Intermediate table:\n{}",
        "Python": "Intermediate table:\n{}",
    },
}

ORCHESTRA_TEMPLATE = {
    "prompt_template_coding": "The database table DF is shown as follows:\n{}\n\nInstruction:\n{}",
    "prompt_template_reasoning": (
        "The table is shown as follows:\n{}\n\n"
        "Answer the following question based on the data above: \"{}\"."
    ),
    "intermediate_prompt_template": {
        "SQL": "Intermediate table:\n{}",
        "Python": "Intermediate table:\n{}",
        "Reasoner": "Intermediate table:\n{}",
        "Coder": "Intermediate table:\n{}\n\nInstruction:\n{}",
    },
}

ORCHESTRA_REASONING_SYSTEM = """
## Role
You are the reasoning agent in a collaborative two-agent system for table
question answering.

## Responsibilities
1. Generate clear instructions for the coding agent to process the table.
2. After receiving an intermediate table, either provide the final answer or
   provide another instruction.
3. In each response, generate exactly one of:
   - Instruction: ...
   - Answer: ...

Do not write SQL or Python code yourself. Keep answers concise.
""".strip()

ORCHESTRA_CODING_SYSTEM = """
## Role
You are the coding agent in a collaborative two-agent system for table question
answering.

Generate SQL or Python code from the instruction and table. The code will be
executed by an external system and the resulting table will be returned to the
reasoning agent.

Output only one code block:
- SQL: ```...```
- Python: ```...```
""".strip()

ORCHESTRA_DECISION_SYSTEM = """
You are a table question answering assistant. Given the conversation and table
reasoning context, provide one concise final answer.

Start with "Answer: ".
""".strip()

ORCHESTRA_TABLEFACT_REASONING_SYSTEM = """
## Role
You are the reasoning agent in a collaborative two-agent system for table fact
verification.

Generate either:
- Instruction: ...
- Answer: yes
- Answer: no

Do not write SQL or Python code yourself. The final answer must be yes or no.
""".strip()

ORCHESTRA_TABLEFACT_DECISION_SYSTEM = """
You are a table fact verification assistant. Given the conversation and table
reasoning context, judge whether the statement is entailed by the table.

Start with "Answer: " and answer only yes or no.
""".strip()

ORCHESTRA_REASONING_EXAMPLES = """
Below are compact examples.

########
Input:
The table is shown as follows:
[HEAD]: name|career_win_loss
---
[ROW] 1: Australian Open|22-18
[ROW] 2: Indian Wells|16-13

Answer the following question based on the data above:
"did he win more at the australian open or indian wells?"

Output:
Instruction: Retrieve the career win-loss record for Australian Open and Indian Wells.

Input:
Intermediate table:
[HEAD]: name|career_win_loss
---
[ROW] 1: Australian Open|22-18
[ROW] 2: Indian Wells|16-13

Output:
Answer: Australian Open
""".strip()

ORCHESTRA_CODING_EXAMPLES = """
Below are compact examples.

########
Input:
The database table DF is shown as follows:
[HEAD]: name|career_win_loss
---
[ROW] 1: Australian Open|22-18
[ROW] 2: Indian Wells|16-13

Instruction:
Retrieve the career win-loss record for Australian Open and Indian Wells.

Output:
SQL: ```SELECT name, career_win_loss FROM DF WHERE name="Australian Open" OR name="Indian Wells";```
""".strip()

ORCHESTRA_TABLEFACT_REASONING_EXAMPLES = """
Below is a compact example.

########
Input:
The table is shown as follows:
[HEAD]: year|score
---
[ROW] 1: 2004|270

Answer the following question based on the data above:
"Determine whether the following statement is entailed by the table. Answer true
if it is entailed and false if it is refuted: in 2004 the score is less than
270"

Output:
Instruction: Select the score for year 2004.

Input:
Intermediate table:
[HEAD]: score
---
[ROW] 1: 270

Output:
Answer: no
""".strip()

ORCHESTRA_TABLEFACT_CODING_EXAMPLES = """
Below is a compact example.

########
Input:
The database table DF is shown as follows:
[HEAD]: year|score
---
[ROW] 1: 2004|270

Instruction:
Select the score for year 2004.

Output:
SQL: ```SELECT score FROM DF WHERE year=2004;```
""".strip()


@dataclass(frozen=True)
class Step:
    kind: str
    content: str
    raw: str


@dataclass(frozen=True)
class ScoreResult:
    metric: str
    score: float
    correct: bool
    expected: str
    actual: str


@dataclass
class ModelCallRecord:
    role: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float
    prompt_chars: int
    response_chars: int
    prompt_preview: str
    response_preview: str


@dataclass
class CaseRunStats:
    calls: list[ModelCallRecord] = field(default_factory=list)

    def add(self, record: ModelCallRecord) -> None:
        self.calls.append(record)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(call.prompt_tokens for call in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.completion_tokens for call in self.calls)

    @property
    def total_tokens(self) -> int:
        total = sum(call.total_tokens for call in self.calls)
        if total:
            return total
        return self.input_tokens + self.output_tokens

    @property
    def max_context_tokens(self) -> int:
        return max((call.prompt_tokens for call in self.calls), default=0)


class EdgeModel:
    """OpenAI-compatible single edge-model caller with per-case accounting."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.tokenizer_name = configured_tokenizer_name(self.config)

    def generate(
        self,
        prompt: str,
        *,
        stats: CaseRunStats,
        role: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        full_prompt = combine_system_prompt(system_prompt, prompt)
        config = dict(self.config)
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
            config["max_output_tokens"] = max_tokens

        started = time.perf_counter()
        from clover.supervisor.client import extract_token_usage, generate_remote_text

        result = generate_remote_text(full_prompt, config)
        elapsed = time.perf_counter() - started
        usage = extract_token_usage(result.response_payload)
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = count_text_tokens(full_prompt, tokenizer_name=self.tokenizer_name)
        if completion_tokens <= 0:
            completion_tokens = count_text_tokens(result.text, tokenizer_name=self.tokenizer_name)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        stats.add(
            ModelCallRecord(
                role=role,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                elapsed_seconds=elapsed,
                prompt_chars=len(full_prompt),
                response_chars=len(result.text),
                prompt_preview=preview(full_prompt),
                response_preview=preview(result.text),
            )
        )
        return result.text


class TableExecutor:
    """Small SQL/Python executor compatible with the original baselines' DF loop."""

    def __init__(self, source_df: Any) -> None:
        self.source_df = source_df
        self.series_dfs = [source_df]
        self.execution_errors: list[str] = []

    def execute(self, df: Any, code: str, code_type: str) -> Any | None:
        normalized_type = str(code_type or "").strip().lower()
        if normalized_type.startswith("sql"):
            return self._execute_sql(df, code)
        if normalized_type.startswith("python"):
            return self._execute_python(df, code)
        self.execution_errors.append(f"Unsupported code type: {code_type}")
        return None

    def execute_with_fallback(self, df: Any, code: str, code_type: str) -> Any | None:
        renewed_df = self.execute(df, code, code_type)
        index = len(self.series_dfs) - 1
        while index >= 0 and renewed_df is None:
            renewed_df = self.execute(self.series_dfs[index], code, code_type)
            index -= 1
        return renewed_df

    def _execute_sql(self, df: Any, sql: str) -> Any | None:
        import pandas as pd

        conn = sqlite3.connect(":memory:")
        try:
            df.to_sql("DF", conn, if_exists="replace", index=False)
            for index, hist_df in enumerate(self.series_dfs):
                hist_df.to_sql(f"DF{index}", conn, if_exists="replace", index=False)
            return pd.read_sql_query(normalize_sql(sql), conn)
        except Exception as exc:  # noqa: BLE001 - model-generated code can fail arbitrarily.
            self.execution_errors.append(f"SQL execution failed: {exc}; SQL={sql}")
            return None
        finally:
            conn.close()

    def _execute_python(self, df: Any, code: str) -> Any | None:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - benchmark env has these deps.
            self.execution_errors.append(f"Python execution dependency missing: {exc}")
            return None

        local_vars: dict[str, Any] = {
            "DF": df.copy(),
            "pd": pd,
            "np": np,
            "re": re,
            "math": math,
        }
        for index, hist_df in enumerate(self.series_dfs):
            local_vars[f"DF{index}"] = hist_df.copy()
        try:
            stdout_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer):
                exec(code, {"__builtins__": __builtins__}, local_vars)  # noqa: S102
            renewed_df = local_vars.get("DF")
            if renewed_df is None:
                self.execution_errors.append("Python code did not leave a DF variable")
                return None
            return renewed_df
        except Exception as exc:  # noqa: BLE001 - model-generated code can fail arbitrarily.
            self.execution_errors.append(f"Python execution failed: {exc}; code={code}")
            return None


class ReAcTableRunner:
    def __init__(
        self,
        *,
        model: EdgeModel,
        datasets_root: Path,
        repeat_times: int,
        max_iters: int,
        max_demo: int,
        line_limit: int | float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.datasets_root = datasets_root
        self.repeat_times = repeat_times
        self.max_iters = max_iters
        self.max_demo = max_demo
        self.line_limit = line_limit
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.template = load_reactable_template(datasets_root)
        self.demo_prompt = build_reactable_demo_prompt(
            datasets_root=datasets_root,
            template=self.template,
            max_demo=max_demo,
        )

    def run_case(
        self,
        case: dict[str, Any],
        table_path: Path,
        stats: CaseRunStats,
    ) -> dict[str, Any]:
        predictions = []
        traces = []
        for repeat_index in range(self.repeat_times):
            prediction, trace = self._run_episode(case, table_path, stats, repeat_index)
            predictions.append(prediction)
            traces.append(trace)
        return {
            "prediction": majority_vote(predictions),
            "all_predictions": predictions,
            "trace": traces,
        }

    def _run_episode(
        self,
        case: dict[str, Any],
        table_path: Path,
        stats: CaseRunStats,
        repeat_index: int,
    ) -> tuple[str, dict[str, Any]]:
        df = read_normalized_table(table_path)
        executor = TableExecutor(df)
        prompt_template = self.template["prompt_template"]
        intermediate_templates = self.template.get("intermediate_prompt_template") or {}
        prompt = self.demo_prompt + prompt_template.format(
            table_formatter(df, line_limit=self.line_limit),
            case["question"],
        ) + "\n"
        code_history: set[str] = set()
        original_outputs: list[str] = []
        steps = []

        for iteration in range(1, self.max_iters + 2):
            output = self.model.generate(
                prompt,
                stats=stats,
                role="reactable",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            step = parse_step(output)
            original_outputs.append(output)
            steps.append(
                {
                    "iteration": iteration,
                    "kind": step.kind,
                    "content_preview": preview(step.content),
                    "raw_preview": preview(output),
                }
            )
            if iteration > self.max_iters:
                return clean_answer(step.content or output), {
                    "repeat_index": repeat_index,
                    "steps": steps,
                    "original_outputs": original_outputs,
                    "execution_errors": executor.execution_errors,
                    "stop_reason": "max_iters",
                }
            if step.kind == "Answer":
                return clean_answer(step.content), {
                    "repeat_index": repeat_index,
                    "steps": steps,
                    "original_outputs": original_outputs,
                    "execution_errors": executor.execution_errors,
                    "stop_reason": "answer",
                }
            if step.kind not in {"SQL", "Python"}:
                forced = self._force_answer(prompt, stats)
                return forced, {
                    "repeat_index": repeat_index,
                    "steps": steps,
                    "original_outputs": original_outputs,
                    "execution_errors": executor.execution_errors,
                    "stop_reason": "unsupported_output",
                }

            renewed_df = executor.execute_with_fallback(df, step.content, step.kind)
            if renewed_df is None or step.content in code_history:
                forced = self._force_answer(prompt, stats)
                return forced, {
                    "repeat_index": repeat_index,
                    "steps": steps,
                    "original_outputs": original_outputs,
                    "execution_errors": executor.execution_errors,
                    "stop_reason": "execution_failed_or_repeated",
                }
            code_history.add(step.content)
            df = renewed_df
            executor.series_dfs.append(renewed_df)
            intermediate_template = (
                intermediate_templates.get(step.kind) or "Intermediate table:\n{}"
            )
            prompt = (
                prompt.strip("\n")
                + "\n\n"
                + normalize_step_for_prompt(output)
                + "\n\n"
                + intermediate_template.format(
                    table_formatter(df, line_limit=self.line_limit),
                    case["question"],
                )
            )

        forced = self._force_answer(prompt, stats)
        return forced, {
            "repeat_index": repeat_index,
            "steps": steps,
            "original_outputs": original_outputs,
            "execution_errors": executor.execution_errors,
            "stop_reason": "loop_exhausted",
        }

    def _force_answer(self, prompt: str, stats: CaseRunStats) -> str:
        output = self.model.generate(
            prompt.strip("\n") + "\n\nAnswer: ```",
            stats=stats,
            role="reactable_force_answer",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return clean_answer(extract_answer_content(output) or output)


class ChatMemory:
    def __init__(self, *, system_prompt: str, examples: str, role: str, model: EdgeModel) -> None:
        self.system_prompt = system_prompt
        self.examples = examples.strip()
        self.role = role
        self.model = model
        self.turns: list[tuple[str, str]] = []

    def call(
        self,
        prompt: str,
        *,
        stats: CaseRunStats,
        temperature: float,
        max_tokens: int,
    ) -> str:
        rendered = self.render_prompt(prompt)
        output = self.model.generate(
            rendered,
            stats=stats,
            role=self.role,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=self.system_prompt,
        )
        self.turns.append((prompt, output))
        return output

    def render_prompt(self, next_prompt: str | None = None) -> str:
        parts = []
        if self.examples:
            parts.append(self.examples)
        for user_text, assistant_text in self.turns:
            parts.append(f"User:\n{user_text}\n\nAssistant:\n{assistant_text}")
        if next_prompt is not None:
            parts.append(f"User:\n{next_prompt}\n\nAssistant:")
        return "\n\n".join(parts).strip()

    def reasoning_path(self) -> str:
        return self.render_prompt(None)


class OrchestraRunner:
    def __init__(
        self,
        *,
        model: EdgeModel,
        mode: str,
        repeat_times: int,
        max_iters: int,
        line_limit: int | float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.mode = normalize_orchestra_mode(mode)
        self.repeat_times = repeat_times
        self.max_iters = max_iters
        self.line_limit = line_limit
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.template = ORCHESTRA_TEMPLATE

    def run_case(
        self,
        case: dict[str, Any],
        table_path: Path,
        stats: CaseRunStats,
    ) -> dict[str, Any]:
        main_predictions = []
        three_agent_predictions = []
        two_agent_predictions = []
        traces = []
        for repeat_index in range(self.repeat_times):
            prediction, two_agent_prediction, three_agent_prediction, trace = (
                self._run_episode(
                    case,
                    table_path,
                    stats,
                    repeat_index,
                )
            )
            main_predictions.append(prediction)
            if three_agent_prediction is not None:
                three_agent_predictions.append(three_agent_prediction)
            two_agent_predictions.append(two_agent_prediction)
            traces.append(trace)
        return {
            "prediction": majority_vote(main_predictions),
            "all_predictions": main_predictions,
            "three_agent_prediction": majority_vote(three_agent_predictions)
            if three_agent_predictions
            else None,
            "three_agent_all_predictions": three_agent_predictions,
            "two_agent_prediction": majority_vote(two_agent_predictions),
            "two_agent_all_predictions": two_agent_predictions,
            "trace": traces,
        }

    def _run_episode(
        self,
        case: dict[str, Any],
        table_path: Path,
        stats: CaseRunStats,
        repeat_index: int,
    ) -> tuple[str, str, str | None, dict[str, Any]]:
        df = read_normalized_table(table_path)
        executor = TableExecutor(df)
        is_tablefact = case.get("dataset") == "tablefact"
        reasoner = ChatMemory(
            system_prompt=(
                ORCHESTRA_TABLEFACT_REASONING_SYSTEM
                if is_tablefact
                else ORCHESTRA_REASONING_SYSTEM
            ),
            examples=(
                ORCHESTRA_TABLEFACT_REASONING_EXAMPLES
                if is_tablefact
                else ORCHESTRA_REASONING_EXAMPLES
            ),
            role="orchestra_reasoner",
            model=self.model,
        )
        coder = ChatMemory(
            system_prompt=ORCHESTRA_CODING_SYSTEM,
            examples=(
                ORCHESTRA_TABLEFACT_CODING_EXAMPLES
                if is_tablefact
                else ORCHESTRA_CODING_EXAMPLES
            ),
            role="orchestra_coder",
            model=self.model,
        )

        current_reasoner_prompt = self.template["prompt_template_reasoning"].format(
            table_formatter(df, line_limit=self.line_limit),
            case["question"],
        )
        current_coder_df = df
        code_history: set[str] = set()
        steps = []
        two_agent_prediction = ""

        for iteration in range(1, self.max_iters + 1):
            reasoner_output = reasoner.call(
                current_reasoner_prompt,
                stats=stats,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            reasoner_step = parse_step(reasoner_output)
            steps.append(
                {
                    "iteration": iteration,
                    "agent": "reasoner",
                    "kind": reasoner_step.kind,
                    "content_preview": preview(reasoner_step.content),
                    "raw_preview": preview(reasoner_output),
                }
            )
            if reasoner_step.kind == "Answer":
                two_agent_prediction = clean_answer(reasoner_step.content)
                break
            instruction = reasoner_step.content
            if reasoner_step.kind != "Instruction" or not instruction:
                two_agent_prediction = self._force_reasoner_answer(reasoner, stats)
                break

            if current_coder_df is df:
                current_coder_prompt = self.template["prompt_template_coding"].format(
                    table_formatter(current_coder_df, line_limit=self.line_limit),
                    instruction,
                )
            else:
                current_coder_prompt = self.template["intermediate_prompt_template"][
                    "Coder"
                ].format(
                    table_formatter(current_coder_df, line_limit=self.line_limit),
                    instruction,
                )
            coder_output = coder.call(
                current_coder_prompt,
                stats=stats,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            coder_step = parse_step(coder_output)
            steps.append(
                {
                    "iteration": iteration,
                    "agent": "coder",
                    "kind": coder_step.kind,
                    "content_preview": preview(coder_step.content),
                    "raw_preview": preview(coder_output),
                }
            )
            if coder_step.kind not in {"SQL", "Python"}:
                two_agent_prediction = self._force_reasoner_answer(reasoner, stats)
                break
            renewed_df = executor.execute_with_fallback(
                current_coder_df,
                coder_step.content,
                coder_step.kind,
            )
            if renewed_df is None or coder_step.content in code_history:
                two_agent_prediction = self._force_reasoner_answer(reasoner, stats)
                break
            code_history.add(coder_step.content)
            executor.series_dfs.append(renewed_df)
            current_coder_df = renewed_df
            current_reasoner_prompt = self.template["intermediate_prompt_template"][
                "Reasoner"
            ].format(
                table_formatter(renewed_df, line_limit=self.line_limit)
            )

        if not two_agent_prediction:
            two_agent_prediction = self._force_reasoner_answer(reasoner, stats)

        three_agent_prediction = None
        if self.mode != "2agent":
            decision_prediction = self._decision_answer(
                reasoner=reasoner,
                stats=stats,
                is_tablefact=is_tablefact,
            )
            three_agent_prediction = decision_prediction or two_agent_prediction

        final_prediction = (
            two_agent_prediction
            if self.mode == "2agent"
            else three_agent_prediction or two_agent_prediction
        )
        return final_prediction, two_agent_prediction, three_agent_prediction, {
            "repeat_index": repeat_index,
            "orchestra_mode": self.mode,
            "decision_agent_used": self.mode != "2agent",
            "steps": steps,
            "reasoning_path_preview": preview(reasoner.reasoning_path(), 2000),
            "execution_errors": executor.execution_errors,
        }

    def _force_reasoner_answer(self, reasoner: ChatMemory, stats: CaseRunStats) -> str:
        output = reasoner.call(
            "Please provide an answer directly based on current information, "
            "starting with 'Answer: '.",
            stats=stats,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return clean_answer(extract_answer_content(output) or output)

    def _decision_answer(
        self,
        *,
        reasoner: ChatMemory,
        stats: CaseRunStats,
        is_tablefact: bool,
    ) -> str:
        system_prompt = (
            ORCHESTRA_TABLEFACT_DECISION_SYSTEM
            if is_tablefact
            else ORCHESTRA_DECISION_SYSTEM
        )
        prompt = (
            reasoner.reasoning_path()
            + "\n\nPlease provide an answer directly based on current information, "
            "starting with 'Answer: '."
        )
        output = self.model.generate(
            prompt,
            stats=stats,
            role="orchestra_decision",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=system_prompt,
        )
        return clean_answer(extract_answer_content(output) or output)


def run_baseline(
    *,
    baseline: str,
    orchestra_mode: str = "3agent",
    dataset: str,
    model_config: dict[str, Any],
    dataset_root: Path,
    datasets_root: Path,
    output_dir: Path,
    max_cases: int | None = None,
    sample_size: int | None = None,
    case_ids: set[str] | None = None,
    dataset_id: str | None = None,
    qtypes: set[str] | None = None,
    qsubtypes: set[str] | None = None,
    split: str | None = None,
    subset: str | None = None,
    seed: int = 20260528,
    max_workers: int = 8,
    repeat_times: int = 1,
    max_iters: int = 5,
    max_demo: int = 5,
    line_limit: int | float = 10,
    temperature: float | None = None,
    max_tokens: int = 1024,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized_baseline = normalize_baseline(baseline)
    normalized_orchestra_mode = normalize_orchestra_mode(orchestra_mode)
    normalized_dataset = normalize_dataset(dataset)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = select_baseline_cases(
        dataset=normalized_dataset,
        dataset_root=dataset_root,
        max_cases=max_cases,
        sample_size=sample_size,
        case_ids=case_ids or set(),
        dataset_id=dataset_id,
        qtypes=qtypes or set(),
        qsubtypes=qsubtypes or set(),
        split=split,
        subset=subset,
        seed=seed,
    )

    edge_model = EdgeModel(model_config)
    if temperature is None:
        temperature = 0.6 if normalized_baseline == "reactable" else 0.7
    if normalized_baseline == "reactable":
        runner: Any = ReAcTableRunner(
            model=edge_model,
            datasets_root=datasets_root,
            repeat_times=repeat_times,
            max_iters=max_iters,
            max_demo=max_demo,
            line_limit=line_limit,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        runner = OrchestraRunner(
            model=edge_model,
            mode=normalized_orchestra_mode,
            repeat_times=repeat_times,
            max_iters=max_iters,
            line_limit=line_limit,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    progress_bar = BaselineProgress(len(selected_cases)) if progress else None
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()

    def process(sample_index: int, sampled_case: dict[str, Any]) -> dict[str, Any]:
        runtime_case_id = runtime_case_id_for(sampled_case, sample_index)
        case_dir = output_dir / "cases" / runtime_case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        table_path = dataset_root / sampled_case["dataset_id"] / "table.csv"
        stats = CaseRunStats()
        case_started = time.perf_counter()
        base_record = base_case_record(
            dataset=normalized_dataset,
            sampled_case=sampled_case,
            sample_index=sample_index,
            runtime_case_id=runtime_case_id,
            case_dir=case_dir,
        )
        try:
            run_output = runner.run_case(sampled_case, table_path, stats)
            if normalized_baseline == "orchestra":
                three_agent_prediction = run_output.get("three_agent_prediction")
                two_agent_prediction = run_output.get("two_agent_prediction") or ""
                two_agent_score = score_prediction(
                    dataset=normalized_dataset,
                    case=sampled_case,
                    prediction=two_agent_prediction,
                )
                three_agent_score = (
                    score_prediction(
                        dataset=normalized_dataset,
                        case=sampled_case,
                        prediction=three_agent_prediction,
                    )
                    if three_agent_prediction is not None
                    else None
                )
                if normalized_orchestra_mode == "2agent":
                    prediction = two_agent_prediction
                    score = two_agent_score
                else:
                    prediction = three_agent_prediction or two_agent_prediction
                    score = three_agent_score or two_agent_score
            else:
                prediction = run_output["prediction"]
                score = score_prediction(
                    dataset=normalized_dataset,
                    case=sampled_case,
                    prediction=prediction,
                )
            record = {
                **base_record,
                "runtime_ok": True,
                "prediction": json_ready(prediction),
                "prediction_preview": preview(prediction),
                "final_answer_standard_text": score.actual,
                "answer_correct": bool(score.correct),
                "metric": score.metric,
                "score": score.score,
                "all_predictions": json_ready(run_output.get("all_predictions")),
                "error": None,
            }
            if normalized_baseline == "orchestra":
                three_agent_attempted = normalized_orchestra_mode != "2agent"
                record.update(
                    {
                        "orchestra_mode": normalized_orchestra_mode,
                        "three_agent_prediction": json_ready(three_agent_prediction),
                        "three_agent_prediction_preview": preview(three_agent_prediction),
                        "three_agent_all_predictions": json_ready(
                            run_output.get("three_agent_all_predictions")
                        ),
                        "three_agent_final_answer_standard_text": (
                            three_agent_score.actual if three_agent_score else None
                        ),
                        "three_agent_answer_correct": (
                            bool(three_agent_score.correct)
                            if three_agent_score
                            else None
                        ),
                        "three_agent_metric": (
                            three_agent_score.metric
                            if three_agent_score
                            else metric_name_for(normalized_dataset, sampled_case)
                        ),
                        "three_agent_score": (
                            three_agent_score.score if three_agent_score else None
                        ),
                        "two_agent_prediction": json_ready(two_agent_prediction),
                        "two_agent_prediction_preview": preview(two_agent_prediction),
                        "two_agent_all_predictions": json_ready(
                            run_output.get("two_agent_all_predictions")
                        ),
                        "two_agent_final_answer_standard_text": two_agent_score.actual,
                        "two_agent_answer_correct": bool(two_agent_score.correct),
                        "two_agent_metric": two_agent_score.metric,
                        "two_agent_score": two_agent_score.score,
                    }
                )
            else:
                record.update(
                    {
                        "two_agent_prediction": json_ready(
                            run_output.get("two_agent_prediction")
                        ),
                        "two_agent_all_predictions": json_ready(
                            run_output.get("two_agent_all_predictions")
                        ),
                    }
                )
            write_json(case_dir / "trace.json", json_ready(run_output.get("trace")))
        except Exception as exc:  # noqa: BLE001 - keep benchmark running by case.
            record = {
                **base_record,
                "runtime_ok": False,
                "prediction": None,
                "prediction_preview": None,
                "final_answer_standard_text": None,
                "answer_correct": False,
                "metric": metric_name_for(normalized_dataset, sampled_case),
                "score": 0.0,
                "all_predictions": [],
                "error": format_exception(exc),
            }
            if normalized_baseline == "orchestra":
                record.update(
                    {
                        "orchestra_mode": normalized_orchestra_mode,
                        "three_agent_prediction": None,
                        "three_agent_prediction_preview": None,
                        "three_agent_all_predictions": [],
                        "three_agent_final_answer_standard_text": None,
                        "three_agent_answer_correct": False
                        if three_agent_attempted
                        else None,
                        "three_agent_metric": metric_name_for(
                            normalized_dataset, sampled_case
                        ),
                        "three_agent_score": 0.0 if three_agent_attempted else None,
                        "two_agent_prediction": None,
                        "two_agent_prediction_preview": None,
                        "two_agent_all_predictions": [],
                        "two_agent_final_answer_standard_text": None,
                        "two_agent_answer_correct": False,
                        "two_agent_metric": metric_name_for(
                            normalized_dataset, sampled_case
                        ),
                        "two_agent_score": 0.0,
                    }
                )
        record.update(model_stats_record(stats))
        record["elapsed_seconds"] = time.perf_counter() - case_started
        write_json(case_dir / "case_result.json", record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(process, sample_index, sampled_case): sample_index
            for sample_index, sampled_case in enumerate(selected_cases)
        }
        for future in as_completed(futures):
            record = future.result()
            with records_lock:
                records.append(record)
                if progress_bar is not None:
                    progress_bar.update(records)

    if progress_bar is not None:
        progress_bar.close()
    records.sort(key=lambda item: item["sample_index"])
    cases_index = output_dir / "cases_index.jsonl"
    mismatch_cases = output_dir / "answer_mismatch_cases.jsonl"
    failure_cases = output_dir / "failure_cases.jsonl"
    write_jsonl(cases_index, records)
    write_jsonl(
        mismatch_cases,
        [
            mismatch_record(record)
            for record in records
            if record.get("runtime_ok") and not record.get("answer_correct")
        ],
    )
    write_jsonl(failure_cases, [record for record in records if not record.get("runtime_ok")])
    summary = build_summary(
        baseline=normalized_baseline,
        orchestra_mode=normalized_orchestra_mode
        if normalized_baseline == "orchestra"
        else None,
        dataset=normalized_dataset,
        records=records,
        selected_cases=selected_cases,
        output_dir=output_dir,
        model_config=model_config,
        elapsed_seconds=time.perf_counter() - started,
        max_workers=max_workers,
        repeat_times=repeat_times,
        max_iters=max_iters,
        max_demo=max_demo if normalized_baseline == "reactable" else None,
        line_limit=line_limit,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        sample_size=sample_size,
        split=split,
        subset=subset,
        cases_index=cases_index,
        mismatch_cases=mismatch_cases,
        failure_cases=failure_cases,
    )
    write_json(output_dir / "run_summary.json", summary)
    return summary


def select_baseline_cases(
    *,
    dataset: str,
    dataset_root: Path,
    max_cases: int | None,
    sample_size: int | None,
    case_ids: set[str],
    dataset_id: str | None,
    qtypes: set[str],
    qsubtypes: set[str],
    split: str | None,
    subset: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    if dataset == "tablebench":
        from benchmarks.tablebench.eval import select_tablebench_cases

        selected = select_tablebench_cases(
            tablebench_root=dataset_root,
            max_cases=max_cases,
            case_ids=case_ids,
            dataset_id=dataset_id,
            qtypes=qtypes or {"FactChecking", "NumericalReasoning"},
            qsubtypes=qsubtypes,
            include_visualization=False,
            sample_size=sample_size,
            seed=seed,
        )
    elif dataset == "wikitq":
        from benchmarks.wikitq.eval import select_wikitq_cases

        selected = select_wikitq_cases(
            wikitq_root=dataset_root,
            max_cases=max_cases,
            case_ids=case_ids,
            dataset_id=dataset_id,
            split=split or "pristine-unseen-tables",
            sample_size=sample_size,
            seed=seed,
        )
    else:
        from benchmarks.tablefact.eval import select_tablefact_cases

        selected = select_tablefact_cases(
            tablefact_root=dataset_root,
            max_cases=max_cases,
            case_ids=case_ids,
            dataset_id=dataset_id,
            split=split or "test",
            subset=subset or "small",
            sample_size=sample_size,
            seed=seed,
        )
    return [dict(case, dataset=dataset) for case in selected]


def load_tablebench_metrics() -> Any:
    return load_module_from_repo_path(
        "clover_benchmark_tablebench_metrics_light",
        REPO_ROOT / "benchmarks" / "tablebench" / "metrics.py",
    )


def load_wikitq_metrics() -> Any:
    return load_module_from_repo_path(
        "clover_benchmark_wikitq_metrics_light",
        REPO_ROOT / "benchmarks" / "wikitq" / "metrics.py",
    )


def load_module_from_repo_path(module_name: str, path: Path) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def score_prediction(dataset: str, case: dict[str, Any], prediction: Any) -> Any:
    if dataset == "wikitq":
        return load_wikitq_metrics().score_wikitq_answer(
            expected=case.get("expected_answer"),
            expected_canon=case.get("expected_canon"),
            actual=prediction,
        )
    score = load_tablebench_metrics().score_tablebench_answer(
        expected=case.get("expected_answer"),
        actual=prediction,
        qtype=case.get("qtype") or "FactChecking",
        qsubtype=case.get("qsubtype"),
    )
    if dataset == "tablefact":
        return ScoreResult(
            metric="accuracy",
            score=score.score,
            correct=score.correct,
            expected=score.expected,
            actual=score.actual,
        )
    return score


def metric_name_for(dataset: str, case: dict[str, Any]) -> str:
    if dataset == "wikitq":
        return "denotation_em"
    if dataset == "tablefact":
        return "accuracy"
    return load_tablebench_metrics().tablebench_metric_name(
        case.get("qtype"),
        case.get("qsubtype"),
    )


def base_case_record(
    *,
    dataset: str,
    sampled_case: dict[str, Any],
    sample_index: int,
    runtime_case_id: str,
    case_dir: Path,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "runtime_case_id": runtime_case_id,
        "dataset": dataset,
        "dataset_id": sampled_case["dataset_id"],
        "case_id": sampled_case["case_id"],
        "case_index": sampled_case.get("case_index"),
        "question": sampled_case.get("question"),
        "qtype": sampled_case.get("qtype"),
        "qsubtype": sampled_case.get("qsubtype"),
        "split": sampled_case.get("split"),
        "answer_type": sampled_case.get("answer_type"),
        "expected_raw": json_ready(sampled_case.get("expected_answer")),
        "expected_standard_text": json_ready(sampled_case.get("expected_answer")),
        "metric": metric_name_for(dataset, sampled_case),
        "score": 0.0,
        "runtime_ok": False,
        "answer_correct": False,
        "case_dir": display_path(case_dir),
    }


def model_stats_record(stats: CaseRunStats) -> dict[str, Any]:
    return {
        "model_calls": stats.call_count,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "total_tokens": stats.total_tokens,
        "max_context_tokens": stats.max_context_tokens,
        "model_call_trace": [json_ready(call.__dict__) for call in stats.calls],
    }


def build_summary(
    *,
    baseline: str,
    orchestra_mode: str | None = None,
    dataset: str,
    records: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    output_dir: Path,
    model_config: dict[str, Any],
    elapsed_seconds: float,
    max_workers: int,
    repeat_times: int,
    max_iters: int,
    max_demo: int | None,
    line_limit: int | float,
    temperature: float,
    max_tokens: int,
    seed: int,
    sample_size: int | None,
    split: str | None,
    subset: str | None,
    cases_index: Path,
    mismatch_cases: Path,
    failure_cases: Path,
) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record.get("answer_correct"))
    runtime_successes = sum(1 for record in records if record.get("runtime_ok"))
    model_calls = sum(int(record.get("model_calls", 0) or 0) for record in records)
    input_tokens = sum(int(record.get("input_tokens", 0) or 0) for record in records)
    output_tokens = sum(int(record.get("output_tokens", 0) or 0) for record in records)
    total_tokens = sum(int(record.get("total_tokens", 0) or 0) for record in records)
    max_context_sum = sum(int(record.get("max_context_tokens", 0) or 0) for record in records)
    elapsed_sum = sum(float(record.get("elapsed_seconds", 0.0) or 0.0) for record in records)
    summary = {
        "run_name": output_dir.name,
        "stage": f"{dataset}_{baseline}_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "dataset": dataset,
        "orchestra_mode": orchestra_mode if baseline == "orchestra" else None,
        "edge_only": True,
        "cloud_model": None,
        "edge_model": config_summary(model_config),
        "sample_size": len(selected_cases),
        "requested_sample_size": sample_size,
        "seed": seed,
        "split": split,
        "subset": subset,
        "parallel_workers": max_workers,
        "repeat_times": repeat_times,
        "max_iters": max_iters,
        "max_demo": max_demo,
        "line_limit": "inf"
        if isinstance(line_limit, float) and math.isinf(line_limit)
        else line_limit,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "total_cases": total,
        "runtime_successes": runtime_successes,
        "runtime_failures": total - runtime_successes,
        "correct": correct,
        "accuracy": safe_divide(correct, total),
        "accuracy_on_all_cases": safe_divide(correct, total),
        "accuracy_on_successes": safe_divide(correct, runtime_successes),
        "model_calls": model_calls,
        "model_calls_per_query": safe_divide(model_calls, total),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "avg_max_context_tokens_per_query": safe_divide(max_context_sum, total),
        "avg_generated_tokens_per_query": safe_divide(output_tokens, total),
        "avg_total_tokens_per_query": safe_divide(total_tokens, total),
        "average_case_seconds": safe_divide(elapsed_sum, total),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_per_query_wall": safe_divide(elapsed_seconds, total),
        "scores_by_metric": score_groups(records, "metric"),
        "scores_by_qtype": score_groups(records, "qtype"),
        "scores_by_qsubtype": score_groups(records, "qsubtype"),
        "run_dir": display_path(output_dir),
        "cases_index": display_path(cases_index),
        "answer_mismatch_cases": display_path(mismatch_cases),
        "failure_cases": display_path(failure_cases),
        "baseline_contract": {
            "uses_single_model": True,
            "uses_edge_model_only": True,
            "uses_cloud_model": False,
            "uses_clover_datasets": True,
            "supports_case_concurrency": True,
        },
    }
    summary["brief_metrics"] = {
        "ACC": summary["accuracy_on_all_cases"],
        "model_calls": model_calls,
        "avg_model_calls_per_query": summary["model_calls_per_query"],
        "avg_max_context_tokens_per_query": summary["avg_max_context_tokens_per_query"],
        "avg_generated_tokens_per_query": summary["avg_generated_tokens_per_query"],
        "avg_time_seconds_per_query": summary["average_case_seconds"],
    }
    if baseline == "orchestra":
        add_orchestra_summary_metrics(summary, records, total)
    return summary


def add_orchestra_summary_metrics(
    summary: dict[str, Any], records: list[dict[str, Any]], total: int
) -> None:
    for mode_name, field_prefix in (
        ("2agent", "two_agent"),
        ("3agent", "three_agent"),
    ):
        evaluated_records = [
            record
            for record in records
            if record.get(f"{field_prefix}_answer_correct") is not None
        ]
        correct = sum(
            1
            for record in evaluated_records
            if record.get(f"{field_prefix}_answer_correct")
        )
        score_sum = sum(
            float(record.get(f"{field_prefix}_score") or 0.0)
            for record in evaluated_records
        )
        evaluated = len(evaluated_records)
        accuracy = safe_divide(correct, evaluated) if evaluated else None
        summary[f"orchestra_{mode_name}_correct"] = correct
        summary[f"orchestra_{mode_name}_evaluated"] = evaluated
        summary[f"orchestra_{mode_name}_accuracy"] = accuracy
        summary[f"orchestra_{mode_name}_score"] = (
            safe_divide(score_sum, evaluated) if evaluated else None
        )
        summary["brief_metrics"][f"ACC_{mode_name}"] = accuracy
    summary["orchestra_mode_note"] = (
        "ACC is computed from the selected orchestra_mode. "
        "When orchestra_mode=both, ACC follows 3agent while both mode-specific "
        "ACC values are reported."
    )


def score_groups(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get(key) or "unknown"), []).append(record)
    result = {}
    for group_key, group_records in sorted(groups.items()):
        total = len(group_records)
        correct = sum(1 for record in group_records if record.get("answer_correct"))
        runtime_successes = sum(1 for record in group_records if record.get("runtime_ok"))
        score_sum = sum(float(record.get("score") or 0.0) for record in group_records)
        result[group_key] = {
            "total": total,
            "runtime_successes": runtime_successes,
            "correct": correct,
            "accuracy": safe_divide(correct, total),
            "score": safe_divide(score_sum, total),
        }
    return result


def mismatch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record.get("sample_index"),
        "runtime_case_id": record.get("runtime_case_id"),
        "dataset_id": record.get("dataset_id"),
        "case_id": record.get("case_id"),
        "qtype": record.get("qtype"),
        "qsubtype": record.get("qsubtype"),
        "metric": record.get("metric"),
        "score": record.get("score"),
        "question": record.get("question"),
        "expected": record.get("expected_standard_text"),
        "actual": record.get("final_answer_standard_text"),
        "prediction": record.get("prediction"),
        "model_calls": record.get("model_calls"),
        "max_context_tokens": record.get("max_context_tokens"),
        "elapsed_seconds": record.get("elapsed_seconds"),
    }


def load_reactable_template(datasets_root: Path) -> dict[str, Any]:
    path = (
        datasets_root
        / "WikiTableQuestions"
        / "prompt_template"
        / "original-sql-py-no-intermediate.json"
    )
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return REACTABLE_DEFAULT_TEMPLATE


def build_reactable_demo_prompt(
    *,
    datasets_root: Path,
    template: dict[str, Any],
    max_demo: int,
) -> str:
    demo_path = datasets_root / "WikiTableQuestions" / "few-shot-demo" / "WikiTQ-sql-py.json"
    if max_demo <= 0 or not demo_path.is_file():
        return ""
    try:
        demos = json.loads(demo_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    if not isinstance(demos, list):
        return ""
    prompt_template = (
        template.get("prompt_template") or REACTABLE_DEFAULT_TEMPLATE["prompt_template"]
    )
    intermediate_templates = template.get("intermediate_prompt_template") or {}
    parts = []
    for demo in demos[:max_demo]:
        if not isinstance(demo, dict):
            continue
        tables = demo.get("tables") or []
        responses = demo.get("responses") or []
        utterance = str(demo.get("utterance") or "")
        for index, table in enumerate(tables):
            if index == 0:
                parts.append(prompt_template.format(table, utterance))
            else:
                previous_response = str(responses[index - 1]) if index - 1 < len(responses) else ""
                if previous_response.startswith("SQL:"):
                    tmpl = intermediate_templates.get("SQL") or "Intermediate table:\n{}"
                    parts.append(tmpl.format(table, utterance))
                elif previous_response.startswith("Python:"):
                    tmpl = intermediate_templates.get("Python") or "Intermediate table:\n{}"
                    parts.append(tmpl.format(table, utterance))
                else:
                    parts.append(f"Intermediate table:\n{table}")
            if index < len(responses):
                parts.append(str(responses[index]))
    return ("\n\n".join(parts).strip() + "\n\n") if parts else ""


def read_normalized_table(table_path: Path) -> Any:
    import pandas as pd

    df = pd.read_csv(table_path, on_bad_lines="skip", low_memory=False)
    df.columns = unique_names([normalize_col_name(str(column)) for column in df.columns])
    return normalize_dataframe(df)


def normalize_dataframe(df: Any) -> Any:
    for col in list(df.columns):
        try:
            if df[col].dtype == object:
                df[col] = df[col].apply(normalize_cell_value)
        except Exception:
            continue
    return df


def normalize_cell_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.replace("|", " ").strip()
    if stripped.replace(" ", "").lower() in {"n.a", "n/a", "n.a.", "n-a", "nan", "none", "null"}:
        return None
    return stripped


def normalize_col_name(col_name: str) -> str:
    replacements = {
        ".": "",
        " ": "_",
        "\\": "_",
        "(": "",
        ")": "",
        "?": "",
        "\n": "_",
        "&": "",
        ":": "_",
        "/": "_",
        ",": "_",
        "-": "_",
        "'": "",
        "%": "percent",
        "#": "num",
    }
    name = col_name.strip().lower()
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = "c_" + name
    if name in {"from", "select", "where", "group", "order", "by"}:
        name = "c_" + name
    return name


def unique_names(names: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    result = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            result.append(name)
        else:
            result.append(f"{name}_{counts[name]}")
    return result


def table_formatter(
    df: Any,
    *,
    separator: str = "|",
    line_limit: int | float = 10,
) -> str:
    cols = [str(column).replace("\n", " ").replace(" ", "_").lower() for column in df.columns]
    lines = ["[HEAD]: " + separator.join(cols), "---"]
    row_count = int(getattr(df, "shape", [0])[0])
    if row_count == 0:
        lines.append("EMPTY TABLE")
        return "\n".join(lines)

    unlimited = isinstance(line_limit, float) and math.isinf(line_limit)
    limit = row_count if unlimited else max(1, int(line_limit))
    omitted = False
    for index in range(row_count):
        if not unlimited and index >= limit - 2 and index < row_count - 2:
            if not omitted:
                lines.append("...")
                omitted = True
            continue
        values = [
            str(value).replace("nan", "NULL").replace("\n", " ")
            for value in df.iloc[index].tolist()
        ]
        lines.append(f"[ROW] {index + 1}: " + separator.join(values))
    return "\n".join(lines)


def parse_step(text: str) -> Step:
    raw = str(text or "").strip()
    fence_language = re.match(r"^```\s*(sql|python)\b", raw, flags=re.I)
    if fence_language:
        fenced = extract_fenced_content(raw)
        return Step(
            kind="SQL" if fence_language.group(1).lower() == "sql" else "Python",
            content=(fenced or "").strip(),
            raw=raw,
        )
    cleaned = strip_outer_markdown_fence(raw)
    first_line = cleaned.splitlines()[0] if cleaned.splitlines() else ""
    prefix_match = re.match(
        r"^\s*(SQL|Python|Answer|Instruction|Final Answer)\s*:\s*(.*)$",
        first_line,
        re.I,
    )
    if prefix_match:
        kind = prefix_match.group(1)
        kind = "Answer" if kind.lower() == "final answer" else kind[:1].upper() + kind[1:].lower()
        if kind == "Sql":
            kind = "SQL"
        content = extract_fenced_content(cleaned)
        if content is None:
            content = cleaned.split(":", 1)[1].strip() if ":" in cleaned else cleaned
        return Step(
            kind=kind,
            content=clean_answer(content) if kind == "Answer" else content.strip(),
            raw=raw,
        )
    answer = extract_answer_content(cleaned)
    if answer:
        return Step(kind="Answer", content=clean_answer(answer), raw=raw)
    fenced = extract_fenced_content(cleaned)
    if fenced is not None:
        lowered = cleaned.lower()
        if "python" in lowered:
            return Step(kind="Python", content=fenced.strip(), raw=raw)
        if "sql" in lowered:
            return Step(kind="SQL", content=fenced.strip(), raw=raw)
    return Step(kind="Unknown", content=cleaned, raw=raw)


def extract_answer_content(text: str) -> str:
    pattern = re.compile(r"(?:final\s+answer|answer)\s*:\s*(.*)", re.I | re.S)
    matches = list(pattern.finditer(str(text or "")))
    if not matches:
        return ""
    last = matches[-1].group(1).strip()
    fenced = extract_fenced_content(last)
    return fenced.strip() if fenced is not None else last.splitlines()[0].strip()


def extract_fenced_content(text: str) -> str | None:
    matches = re.findall(
        r"```(?:[a-zA-Z0-9_+-]+[ \t]*\n)?(.*?)```",
        str(text or ""),
        flags=re.S,
    )
    if matches:
        return matches[-1]
    parts = str(text or "").split("```")
    if len(parts) >= 2:
        return parts[-1] if len(parts) % 2 == 0 else parts[-2]
    return None


def strip_outer_markdown_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = extract_fenced_content(stripped)
        if inner is not None:
            return inner.strip()
    return stripped


def clean_answer(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        fenced = extract_fenced_content(text)
        if fenced is not None:
            text = fenced.strip()
    else:
        text = strip_outer_markdown_fence(text)
    text = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", text, flags=re.I).strip()
    if text.startswith("```"):
        fenced = extract_fenced_content(text)
        if fenced is not None:
            text = fenced.strip()
    text = text.strip("`").strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.endswith("`."):
        text = text[:-2].strip()
    if text.endswith(".") and len(text) <= 32:
        text = text[:-1].strip()
    return text.strip().strip("\"'")


def normalize_step_for_prompt(output: str) -> str:
    stripped = str(output or "").strip()
    if "```" in stripped and not stripped.rstrip().endswith("```."):
        if stripped.rstrip().endswith("```"):
            return stripped.rstrip() + "."
    return stripped


def normalize_sql(sql: str) -> str:
    return (
        str(sql or "")
        .replace("–", "-")
        .replace("—", "-")
        .replace("―", "-")
        .replace("−", "-")
    )


def majority_vote(predictions: list[str]) -> str:
    if not predictions:
        return ""
    return Counter(predictions).most_common(1)[0][0]


def runtime_case_id_for(sampled_case: dict[str, Any], sample_index: int) -> str:
    return f"sample_{sample_index:05d}__{sampled_case['dataset_id']}__{sampled_case['case_id']}"


def normalize_dataset(dataset: str) -> str:
    value = str(dataset or "").strip().lower()
    aliases = {"wikitablequestions": "wikitq", "tabfact": "tablefact"}
    value = aliases.get(value, value)
    if value not in {"tablebench", "wikitq", "tablefact"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return value


def normalize_baseline(baseline: str) -> str:
    value = str(baseline or "").strip().lower()
    aliases = {"react": "reactable", "reac_table": "reactable", "reac-table": "reactable"}
    value = aliases.get(value, value)
    if value not in {"reactable", "orchestra"}:
        raise ValueError(f"Unsupported baseline: {baseline}")
    return value


def normalize_orchestra_mode(mode: str | None) -> str:
    value = str(mode or "3agent").strip().lower().replace("_", "-")
    aliases = {
        "2-agent": "2agent",
        "two-agent": "2agent",
        "two": "2agent",
        "3-agent": "3agent",
        "three-agent": "3agent",
        "three": "3agent",
    }
    value = aliases.get(value, value)
    if value not in {"2agent", "3agent", "both"}:
        raise ValueError(f"Unsupported orchestra mode: {mode}")
    return value


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": config.get("provider"),
        "api_type": config.get("api_type"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens", config.get("max_output_tokens")),
    }


def combine_system_prompt(system_prompt: str | None, prompt: str) -> str:
    if not system_prompt:
        return prompt
    return f"{system_prompt.strip()}\n\n{prompt}".strip()


def preview(value: Any, max_length: int = 240) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 3)] + "..."


def count_text_tokens(text: str, *, tokenizer_name: str | None = None) -> int:
    try:
        from clover.executor.token_count import count_tokens

        return count_tokens(text, tokenizer_name=tokenizer_name)
    except Exception:  # noqa: BLE001 - summaries should survive missing optional deps.
        stripped = str(text or "").strip()
        return 0 if not stripped else max(1, (len(stripped) + 3) // 4)


def configured_tokenizer_name(config: dict[str, Any] | None = None) -> str | None:
    try:
        from clover.executor.token_count import configured_tokenizer_name as _configured

        return _configured(config)
    except Exception:  # noqa: BLE001
        if isinstance(config, dict):
            for key in ("tokenizer", "tokenizer_name", "model"):
                value = config.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None


def format_exception(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class BaselineProgress:
    def __init__(self, total: int, *, width: int = 24) -> None:
        self.total = total
        self.width = width
        self._last_length = 0

    def update(self, records: list[dict[str, Any]]) -> None:
        if self.total <= 0:
            return
        completed = len(records)
        correct = sum(1 for record in records if record.get("answer_correct"))
        failures = sum(1 for record in records if not record.get("runtime_ok"))
        filled = min(self.width, int(self.width * completed / self.total))
        acc = correct / completed if completed else 0.0
        text = (
            f"\r[{'#' * filled}{'-' * (self.width - filled)}] "
            f"{completed}/{self.total} correct={correct} fail={failures} acc={acc:.3f}"
        )
        padding = " " * max(0, self._last_length - len(text))
        print(text + padding, file=sys.stderr, end="", flush=True)
        self._last_length = len(text)

    def close(self) -> None:
        if self.total > 0:
            print(file=sys.stderr, flush=True)


def print_summary(summary: dict[str, Any]) -> None:
    metric = summary.get("brief_metrics", {})
    lines = [
        ("Baseline", summary.get("baseline")),
        ("Dataset", summary.get("dataset")),
        ("Edge Model", (summary.get("edge_model") or {}).get("model")),
        (
            "Orchestra Mode",
            summary.get("orchestra_mode") if summary.get("baseline") == "orchestra" else None,
        ),
        ("Cases", summary.get("total_cases")),
        ("ACC", format_optional_float(metric.get("ACC"), precision=4)),
        (
            "ACC 2-Agent",
            format_optional_float(metric.get("ACC_2agent"), precision=4)
            if summary.get("baseline") == "orchestra"
            else None,
        ),
        (
            "ACC 3-Agent",
            format_optional_float(metric.get("ACC_3agent"), precision=4)
            if summary.get("baseline") == "orchestra"
            else None,
        ),
        ("Model Calls", summary.get("model_calls")),
        (
            "Calls / Query",
            format_optional_float(metric.get("avg_model_calls_per_query"), precision=4),
        ),
        (
            "Avg Max Ctx Tok / Query",
            format_optional_float(metric.get("avg_max_context_tokens_per_query"), precision=2),
        ),
        (
            "Avg Generated Tok / Query",
            format_optional_float(metric.get("avg_generated_tokens_per_query"), precision=2),
        ),
        (
            "Avg Time / Query (s)",
            format_optional_float(metric.get("avg_time_seconds_per_query"), precision=4),
        ),
        ("Run Dir", summary.get("run_dir")),
    ]
    lines = [(label, value) for label, value in lines if value is not None]
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        print(f"{label.ljust(width)} : {value}")


def format_optional_float(value: Any, *, precision: int) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{precision}f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ReAcTable or Orchestra-style edge-only baselines."
    )
    parser.add_argument("--baseline", required=True, choices=("reactable", "orchestra"))
    parser.add_argument("--dataset", required=True, choices=("tablebench", "wikitq", "tablefact"))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--qtype", action="append", default=[])
    parser.add_argument("--qsubtype", action="append", default=[])
    parser.add_argument("--wikitq-split", default="pristine-unseen-tables")
    parser.add_argument("--tablefact-split", default="test")
    parser.add_argument("--tablefact-subset", default="small")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--max-demo", type=int, default=5)
    parser.add_argument(
        "--orchestra-mode",
        default="3agent",
        choices=(
            "2agent",
            "2-agent",
            "two",
            "two-agent",
            "3agent",
            "3-agent",
            "three",
            "three-agent",
            "both",
        ),
        help=(
            "Orchestra scoring mode. 2agent uses the reasoner+coder answer; "
            "3agent uses the final decision agent; both reports both and keeps "
            "3agent as the main ACC."
        ),
    )
    parser.add_argument("--line-limit", default="10")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = normalize_dataset(args.dataset)
    baseline = normalize_baseline(args.baseline)
    datasets_root = args.datasets_root.expanduser().resolve()
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else (datasets_root / dataset).resolve()
    )
    if dataset == "wikitq":
        split = args.wikitq_split
    elif dataset == "tablefact":
        split = args.tablefact_split
    else:
        split = None
    subset = args.tablefact_subset if dataset == "tablefact" else None
    selected = select_baseline_cases(
        dataset=dataset,
        dataset_root=dataset_root,
        max_cases=args.max_cases,
        sample_size=args.sample_size,
        case_ids=set(args.case_id),
        dataset_id=args.dataset_id,
        qtypes=set(args.qtype),
        qsubtypes=set(args.qsubtype),
        split=split,
        subset=subset,
        seed=args.seed,
    )
    if args.validate_only:
        print(f"Validated {baseline}/{dataset}: {len(selected)} cases")
        return 0
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    if args.repeat_times <= 0:
        raise SystemExit("--repeat-times must be positive")
    if args.max_iters <= 0:
        raise SystemExit("--max-iters must be positive")
    if args.max_demo < 0:
        raise SystemExit("--max-demo must be non-negative")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    run_name = args.run_name or f"{dataset}_{baseline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = (args.output_root / run_name).expanduser().resolve()
    model_config = load_model_config(args.model_config.expanduser().resolve())
    summary = run_baseline(
        baseline=baseline,
        orchestra_mode=args.orchestra_mode,
        dataset=dataset,
        model_config=model_config,
        dataset_root=dataset_root,
        datasets_root=datasets_root,
        output_dir=output_dir,
        max_cases=args.max_cases,
        sample_size=args.sample_size,
        case_ids=set(args.case_id),
        dataset_id=args.dataset_id,
        qtypes=set(args.qtype),
        qsubtypes=set(args.qsubtype),
        split=split,
        subset=subset,
        seed=args.seed,
        max_workers=args.max_workers,
        repeat_times=args.repeat_times,
        max_iters=args.max_iters,
        max_demo=args.max_demo,
        line_limit=parse_line_limit(args.line_limit),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
        progress=not args.no_progress,
    )
    print_summary(summary)
    return 0


def parse_line_limit(value: str) -> int | float:
    normalized = str(value).strip().lower()
    if normalized in {"inf", "infinity", "all", "full"}:
        return float("inf")
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError("--line-limit must be positive or 'inf'")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
