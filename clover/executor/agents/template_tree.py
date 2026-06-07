"""Prompt template tree for local NodeAgents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from clover.executor.node_views import NodeView
from clover.executor.agents.table_repair_prompt import empty_filter_repair_case_json
from clover.executor.result import json_ready
from clover.executor.slm_scheduler import TemplateLeafSpec, ThreadedPrefixTemplateTree
from clover.executor.token_count import configured_tokenizer_name, count_tokens
from clover.task_types import agent_kind_for_task_type, public_task_types


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class TemplateNode:
    """A node in the local NodeAgent prompt template tree."""

    name: str
    template: str | None = None
    additional_templates: tuple[str, ...] = ()
    children: tuple["TemplateNode", ...] = ()
    leaf_description: str = ""

    def template_paths(self) -> tuple[str, ...]:
        paths = (self.template,) if self.template else ()
        return paths + self.additional_templates

    def child(self, name: str) -> "TemplateNode":
        for child in self.children:
            if child.name == name:
                return child
        raise KeyError(f"Template node not found under {self.name!r}: {name!r}")

    def find_path(self, names: tuple[str, ...]) -> tuple["TemplateNode", ...]:
        if not names:
            return (self,)
        next_node = self.child(names[0])
        return (self,) + next_node.find_path(names[1:])


TABLE_NUMBER_LEAF_KEY = (
    "agent:data",
    "family:table_reasoning",
    "interface:solve_python",
    "tool:pandas_env",
    "contract:number",
    "mode:initial",
)
TABLE_STRING_LEAF_KEY = (
    "agent:data",
    "family:table_reasoning",
    "interface:solve_python",
    "tool:pandas_env",
    "contract:string",
    "mode:initial",
)
TABLE_BOOLEAN_LEAF_KEY = (
    "agent:data",
    "family:table_reasoning",
    "interface:solve_python",
    "tool:pandas_env",
    "contract:boolean",
    "mode:initial",
)
TABLE_NUMBER_EMPTY_FILTER_REPAIR_LEAF_KEY = (
    "agent:data",
    "family:table_reasoning",
    "interface:solve_python",
    "tool:pandas_env",
    "contract:number",
    "mode:empty_filter_repair",
)
TABLE_STRING_EMPTY_FILTER_REPAIR_LEAF_KEY = (
    "agent:data",
    "family:table_reasoning",
    "interface:solve_python",
    "tool:pandas_env",
    "contract:string",
    "mode:empty_filter_repair",
)
TABLE_BOOLEAN_EMPTY_FILTER_REPAIR_LEAF_KEY = (
    "agent:data",
    "family:table_reasoning",
    "interface:solve_python",
    "tool:pandas_env",
    "contract:boolean",
    "mode:empty_filter_repair",
)
TABLE_EVIDENCE_LEAF_KEY = (
    "agent:data",
    "family:table_reasoning",
    "interface:evidence_python",
    "tool:pandas_env",
    "contract:evidence_json",
    "mode:initial",
)
DOCUMENT_WORKER_LEAF_KEY = (
    "agent:data",
    "family:document_reasoning",
    "interface:chunk_worker",
    "tool:text_excerpt",
    "contract:evidence_json",
    "mode:initial",
)


NODE_AGENT_TEMPLATE_TREE = TemplateNode(
    name="root",
    children=(
        TemplateNode(
            name="agent:data",
            children=(
                TemplateNode(
                    name="family:table_reasoning",
                    children=(
                        TemplateNode(
                            name="interface:solve_python",
                            children=(
                                TemplateNode(
                                    name="tool:pandas_env",
                                    children=(
                                        TemplateNode(
                                            name="contract:number",
                                            children=(
                                                TemplateNode(
                                                    name="mode:initial",
                                                    template="common/root.md",
                                                    additional_templates=(
                                                        "table_reasoning/agent_loop.md",
                                                    ),
                                                    leaf_description=(
                                                        "Table solve with numeric "
                                                        "output contract."
                                                    ),
                                                ),
                                                TemplateNode(
                                                    name="mode:empty_filter_repair",
                                                    template=(
                                                        "table_reasoning/"
                                                        "empty_filter_repair.md"
                                                    ),
                                                    leaf_description=(
                                                        "Table empty Filter repair "
                                                        "with numeric output contract."
                                                    ),
                                                ),
                                            ),
                                        ),
                                        TemplateNode(
                                            name="contract:string",
                                            children=(
                                                TemplateNode(
                                                    name="mode:initial",
                                                    template="common/root.md",
                                                    additional_templates=(
                                                        "table_reasoning/agent_loop.md",
                                                    ),
                                                    leaf_description=(
                                                        "Table solve with string or "
                                                        "entity output contract."
                                                    ),
                                                ),
                                                TemplateNode(
                                                    name="mode:empty_filter_repair",
                                                    template=(
                                                        "table_reasoning/"
                                                        "empty_filter_repair.md"
                                                    ),
                                                    leaf_description=(
                                                        "Table empty Filter repair "
                                                        "with string output contract."
                                                    ),
                                                ),
                                            ),
                                        ),
                                        TemplateNode(
                                            name="contract:boolean",
                                            children=(
                                                TemplateNode(
                                                    name="mode:initial",
                                                    template="common/root.md",
                                                    additional_templates=(
                                                        "table_reasoning/agent_loop.md",
                                                    ),
                                                    leaf_description=(
                                                        "Table solve with boolean "
                                                        "output contract."
                                                    ),
                                                ),
                                                TemplateNode(
                                                    name="mode:empty_filter_repair",
                                                    template=(
                                                        "table_reasoning/"
                                                        "empty_filter_repair.md"
                                                    ),
                                                    leaf_description=(
                                                        "Table empty Filter repair "
                                                        "with boolean output contract."
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        TemplateNode(
                            name="interface:evidence_python",
                            template="table_reasoning/evidence.md",
                            children=(
                                TemplateNode(
                                    name="tool:pandas_env",
                                    children=(
                                        TemplateNode(
                                            name="contract:evidence_json",
                                            children=(
                                                TemplateNode(
                                                    name="mode:initial",
                                                    leaf_description=(
                                                        "Table-local evidence "
                                                        "collection with Python."
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                TemplateNode(
                    name="family:document_reasoning",
                    children=(
                        TemplateNode(
                            name="interface:chunk_worker",
                            template="document_reasoning/worker.md",
                            children=(
                                TemplateNode(
                                    name="tool:text_excerpt",
                                    children=(
                                        TemplateNode(
                                            name="contract:evidence_json",
                                            children=(
                                                TemplateNode(
                                                    name="mode:initial",
                                                    leaf_description=(
                                                        "Document chunk-local evidence "
                                                        "extraction."
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


DEFAULT_TEMPLATE_LEAF_BY_AGENT_KIND = {
    "table_reasoning": TABLE_NUMBER_LEAF_KEY,
    "document_reasoning": DOCUMENT_WORKER_LEAF_KEY,
}


def render_agent_loop_prompt(
    *,
    task_type: str,
    view: NodeView,
    iteration: int,
) -> str:
    """Render a local NodeAgent Agent Loop prompt."""

    return _render_templates(
        template_paths_for_task_type(task_type),
        context=_agent_loop_template_context(view=view, iteration=iteration),
    )


def render_table_empty_filter_repair_prompt(
    *,
    view: NodeView,
    iteration: int,
    steps: list[dict[str, Any]] | None = None,
) -> str:
    """Render the compact table Filter repair prompt used after empty output."""

    del iteration
    return _render_templates(
        template_paths_for_leaf_key(TABLE_NUMBER_EMPTY_FILTER_REPAIR_LEAF_KEY),
        context={
            "CASE_JSON": empty_filter_repair_case_json(
                view=view,
                steps=steps or [],
            ),
        },
    )


def render_document_worker_prompt(
    *,
    chunk_text: str,
    local_instruction: str,
    advice: str = "",
) -> str:
    """Render the minimal chunk-local document worker prompt."""

    return _render_templates(
        template_paths_for_leaf_key(DOCUMENT_WORKER_LEAF_KEY),
        context={
            "chunk_text": chunk_text,
            "local_instruction": local_instruction,
            "advice": advice,
        },
    )


def render_table_evidence_prompt(
    *,
    prompt_code: str,
    feedback: str,
    iteration: int,
    last_iteration: bool,
) -> str:
    """Render the table evidence collection prompt."""

    return _render_templates(
        template_paths_for_leaf_key(TABLE_EVIDENCE_LEAF_KEY),
        context={
            "prompt_code": prompt_code,
            "feedback": feedback,
            "iteration": iteration,
            "last_iteration": last_iteration,
        },
    )


def template_paths_for_task_type(task_type: str) -> tuple[str, ...]:
    return template_paths_for_leaf_key(_default_leaf_key_for_task_type(task_type))


def template_paths_for_leaf_key(leaf_key: tuple[str, ...]) -> tuple[str, ...]:
    return _template_paths_for_nodes(NODE_AGENT_TEMPLATE_TREE.find_path(leaf_key))


def template_leaf_key_for_local_slm_prompt(
    *,
    prompt_kind: str,
    task_type: str,
    node: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return the registered static-template leaf for a local SLM prompt."""

    del task_type
    if prompt_kind == "document_worker":
        return DOCUMENT_WORKER_LEAF_KEY
    if prompt_kind == "table_evidence":
        return TABLE_EVIDENCE_LEAF_KEY
    if prompt_kind == "table_reasoning_agent_loop":
        return _table_reasoning_leaf_key_for_node(node or {})
    if prompt_kind == "table_reasoning_empty_filter_repair":
        return _table_reasoning_empty_filter_repair_leaf_key_for_node(node or {})
    raise ValueError(f"Unsupported local SLM prompt kind: {prompt_kind!r}")


def slm_template_leaf_specs(
    *,
    tokenizer_name: str | None = None,
) -> tuple[TemplateLeafSpec, ...]:
    """Return stable prompt-template leaves for local SLM scheduling."""

    if tokenizer_name:
        return collect_slm_template_leaf_specs(
            NODE_AGENT_TEMPLATE_TREE,
            tokenizer_name=tokenizer_name,
        )
    return NODE_AGENT_SLM_TEMPLATE_LEAVES


def build_slm_template_scheduler_tree(
    *,
    tokenizer_name: str | None = None,
) -> ThreadedPrefixTemplateTree:
    """Build the threaded template tree used by second-level SLM scheduling."""

    return ThreadedPrefixTemplateTree(slm_template_leaf_specs(tokenizer_name=tokenizer_name))


def _render_templates(
    template_paths: tuple[str, ...],
    *,
    context: dict[str, Any],
) -> str:
    env = _template_environment()
    rendered_parts = []
    for template_path in template_paths:
        rendered_parts.append(env.get_template(template_path).render(**context).strip())
    return "\n\n".join(part for part in rendered_parts if part)


def _agent_loop_template_context(*, view: NodeView, iteration: int) -> dict[str, Any]:
    view_json = _to_pretty_json(json_ready(view.to_dict()))
    return {
        "iteration": iteration,
        "view": view,
        "view_json": view_json,
        "SANDBOX_VIEW": view_json,
    }


def _default_leaf_key_for_task_type(task_type: str) -> tuple[str, ...]:
    try:
        agent_kind = agent_kind_for_task_type(task_type)
    except ValueError as exc:
        available = ", ".join(public_task_types())
        raise ValueError(
            f"Unsupported NodeAgent task_type: {task_type!r}. "
            f"Available task types: {available}"
        ) from exc
    try:
        return DEFAULT_TEMPLATE_LEAF_BY_AGENT_KIND[agent_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported NodeAgent agent kind: {agent_kind!r}") from exc


def _table_reasoning_leaf_key_for_node(node: dict[str, Any]) -> tuple[str, ...]:
    params = node.get("params") if isinstance(node, dict) else None
    answer = params.get("answer") if isinstance(params, dict) else None
    answer_type = answer.get("type") if isinstance(answer, dict) else None
    selected = str(answer_type or "").strip().lower()
    if selected == "boolean":
        return TABLE_BOOLEAN_LEAF_KEY
    if selected in {"string", "entity", "list"}:
        return TABLE_STRING_LEAF_KEY
    return TABLE_NUMBER_LEAF_KEY


def _table_reasoning_empty_filter_repair_leaf_key_for_node(
    node: dict[str, Any],
) -> tuple[str, ...]:
    initial_leaf = _table_reasoning_leaf_key_for_node(node)
    if initial_leaf == TABLE_BOOLEAN_LEAF_KEY:
        return TABLE_BOOLEAN_EMPTY_FILTER_REPAIR_LEAF_KEY
    if initial_leaf == TABLE_STRING_LEAF_KEY:
        return TABLE_STRING_EMPTY_FILTER_REPAIR_LEAF_KEY
    return TABLE_NUMBER_EMPTY_FILTER_REPAIR_LEAF_KEY


def _template_paths_for_nodes(nodes: tuple[TemplateNode, ...]) -> tuple[str, ...]:
    paths: list[str] = []
    for node in nodes:
        paths.extend(node.template_paths())
    return tuple(paths)


def collect_slm_template_leaf_specs(
    node: TemplateNode,
    *,
    tokenizer_name: str | None = None,
    path: tuple[str, ...] = (),
    template_paths: tuple[str, ...] = (),
    static_token_count: int = 0,
) -> tuple[TemplateLeafSpec, ...]:
    node_template_paths = template_paths + node.template_paths()
    node_static_tokens = static_token_count + _template_paths_static_token_count(
        node.template_paths(),
        tokenizer_name=tokenizer_name,
    )
    if not node.children:
        if not path:
            return ()
        return (
            TemplateLeafSpec(
                key=path,
                template_paths=node_template_paths,
                description=node.leaf_description,
                static_token_count=node_static_tokens,
            ),
        )

    specs: list[TemplateLeafSpec] = []
    for child in _ordered_children(node.children, tokenizer_name=tokenizer_name):
        specs.extend(
            collect_slm_template_leaf_specs(
                child,
                tokenizer_name=tokenizer_name,
                path=path + (child.name,),
                template_paths=node_template_paths,
                static_token_count=node_static_tokens,
            )
        )
    return tuple(specs)


def _ordered_children(
    children: tuple[TemplateNode, ...],
    *,
    tokenizer_name: str | None,
) -> tuple[TemplateNode, ...]:
    keyed_children = []
    for index, child in enumerate(children):
        delta_tokens = _template_paths_static_token_count(
            child.template_paths(),
            tokenizer_name=tokenizer_name,
        )
        keyed_children.append((delta_tokens, index, child))
    keyed_children.sort(key=lambda item: (item[0], item[1]))
    return tuple(child for _, _, child in keyed_children)


def _template_paths_static_token_count(
    template_paths: tuple[str, ...],
    *,
    tokenizer_name: str | None,
) -> int:
    if not template_paths:
        return 0
    text = "\n\n".join(_static_template_source(path) for path in template_paths)
    return count_tokens(text, tokenizer_name=tokenizer_name)


def _static_template_source(template_path: str) -> str:
    text = (TEMPLATES_DIR / template_path).read_text(encoding="utf-8")
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.DOTALL)
    return text


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _to_pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


NODE_AGENT_SLM_TEMPLATE_LEAVES: tuple[TemplateLeafSpec, ...] = (
    collect_slm_template_leaf_specs(
        NODE_AGENT_TEMPLATE_TREE,
        tokenizer_name=configured_tokenizer_name(),
    )
)
