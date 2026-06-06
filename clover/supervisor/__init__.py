"""Remote supervisor APIs for decomposition, synthesis, and model calls."""

from clover.supervisor.agent import (
    SUPERVISOR_ACTION_DECOMPOSE,
    SUPERVISOR_ACTION_SYNTHESIZE,
    SUPERVISOR_ACTIONS,
    SupervisorAgent,
    SupervisorStepResult,
)
from clover.supervisor.client import (
    RemoteLLMConfigError,
    RemoteLLMResult,
    create_remote_llm_client,
    extract_token_usage,
    generate_remote_text,
)
from clover.supervisor.decompose import (
    available_task_types as available_decompose_task_types,
    render_initial_task_prompt,
)
from clover.supervisor.decision import (
    SupervisorAction,
    SupervisorDecision,
    SupervisorParseError,
    extract_supervisor_json,
    parse_supervisor_decision,
)
from clover.supervisor.observations import build_compact_document_observation
from clover.supervisor.synthesis_templates import (
    available_task_types as available_synthesis_task_types,
    render_initial_synthesis_prompt,
    render_synthesis_prompt,
    synthesis_payload,
)

__all__ = [
    "RemoteLLMConfigError",
    "RemoteLLMResult",
    "SUPERVISOR_ACTION_DECOMPOSE",
    "SUPERVISOR_ACTION_SYNTHESIZE",
    "SUPERVISOR_ACTIONS",
    "SupervisorAgent",
    "SupervisorAction",
    "SupervisorDecision",
    "SupervisorParseError",
    "SupervisorStepResult",
    "available_decompose_task_types",
    "available_synthesis_task_types",
    "build_compact_document_observation",
    "create_remote_llm_client",
    "extract_supervisor_json",
    "extract_token_usage",
    "generate_remote_text",
    "parse_supervisor_decision",
    "render_initial_synthesis_prompt",
    "render_initial_task_prompt",
    "render_synthesis_prompt",
    "synthesis_payload",
]
