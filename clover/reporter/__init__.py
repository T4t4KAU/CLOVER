"""Reporter prompt rendering and decision parsing."""

from clover.reporter.core import ReporterResult, run_reporter
from clover.reporter.decision import (
    ReporterDecision,
    ReporterParseError,
    extract_reporter_json,
    parse_reporter_decision,
)
from clover.reporter.template_tree import (
    available_task_types,
    initial_report_template_paths,
    render_initial_report_prompt,
    render_report_prompt,
    render_reporter_instruction_prompt,
    reporter_payload,
    sql_repair_template_paths,
    template_paths_for_task_type,
)

__all__ = [
    "ReporterDecision",
    "ReporterParseError",
    "ReporterResult",
    "available_task_types",
    "extract_reporter_json",
    "initial_report_template_paths",
    "parse_reporter_decision",
    "render_initial_report_prompt",
    "render_report_prompt",
    "render_reporter_instruction_prompt",
    "reporter_payload",
    "run_reporter",
    "sql_repair_template_paths",
    "template_paths_for_task_type",
]
