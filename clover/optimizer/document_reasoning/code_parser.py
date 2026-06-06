"""Parse document reasoning remote Python code into CLOVER Logic DAGs."""

from __future__ import annotations

import ast
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from clover.optimizer.ir import DOCUMENT_REASONING_TASK_TYPE
from clover.optimizer.errors import OptimizerParseError


CHUNKS_ALL = "all"
DOCUMENT_COLLECTION_NAMES = frozenset({"context"})
DOCUMENT_CHUNKING_FUNCTIONS = frozenset({"chunk_by_section", "chunk_by_page"})
DOCUMENT_CHUNKING_ALIASES = {
    "chunk_by_section": "chunk_by_section",
    "chunk_by_page": "chunk_by_page",
    "chunk_on_pages": "chunk_by_page",
    "chunk_pages": "chunk_by_page",
}
DOCUMENT_OUTPUT_CONTRACT = {
    "format": "json",
    "fields": ["answer", "explanation", "citation"],
}


class DocumentPlanParseError(OptimizerParseError):
    """Raised when document Python code cannot become a Logic DAG."""


@dataclass(frozen=True)
class _DocumentSourceInfo:
    source_id: str
    default_chunking: dict[str, Any]


@dataclass(frozen=True)
class _ResourceViewSpec:
    id: str
    source_id: str
    op: str
    params: dict[str, Any]


@dataclass(frozen=True)
class _ChunkSelector:
    view_id: str
    chunks: str | tuple[str, ...]


@dataclass(frozen=True)
class _ManifestSpec:
    selector: _ChunkSelector
    local_instruction: str
    local_guidance: str


def parse_remote_document_code_to_logic_dag(
    remote_response: str,
    remote_dsl: dict[str, Any],
) -> dict[str, Any]:
    """Parse document Python code into a document numerical Logic DAG."""

    source_info = _validate_document_reasoning_dsl(remote_dsl)
    code = extract_document_python_code(remote_response)
    tree = _parse_python_module(code)
    prepare_jobs = _required_function(tree, "prepare_jobs")
    transform_outputs = _required_function(tree, "transform_outputs")

    specs, resource_views = _extract_manifest_specs(prepare_jobs, source_info)
    map_groups = _map_groups_from_specs(specs, source_info)
    return {
        "task_type": DOCUMENT_REASONING_TASK_TYPE,
        "resource_processing": [
            _resource_processing_payload(index, view)
            for index, view in enumerate(resource_views)
        ],
        "map_groups": map_groups,
        "static_collectors": [
            _transform_outputs_collector_payload(code, transform_outputs)
        ],
        "edges": [],
    }


def extract_document_python_code(remote_response: str) -> str:
    """Return the Python code block from a Remote LLM document response."""

    if not isinstance(remote_response, str) or not remote_response.strip():
        raise DocumentPlanParseError("Remote document Python response is empty")

    text = remote_response.strip()
    blocks = _extract_fenced_python_blocks(text)
    if blocks:
        for block in blocks:
            if "def prepare_jobs" in block and "def transform_outputs" in block:
                return block
        return blocks[0]
    return text


def _validate_document_reasoning_dsl(remote_dsl: dict[str, Any]) -> _DocumentSourceInfo:
    if remote_dsl.get("task_type") != DOCUMENT_REASONING_TASK_TYPE:
        raise DocumentPlanParseError(
            "Document code parser requires task_type document_reasoning"
        )
    sources = remote_dsl.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DocumentPlanParseError(
            "document_reasoning requires one document source"
        )
    document_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("type") == "document"
        and source.get("source_type") == "pdf"
    ]
    if len(document_sources) != 1:
        raise DocumentPlanParseError(
            "document_reasoning requires exactly one PDF document source"
        )

    source = document_sources[0]
    source_id = source.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise DocumentPlanParseError("document source requires a string id")
    chunking = source.get("schema", {}).get("chunking", {})
    chunk_count = chunking.get("chunk_count")
    if not isinstance(chunk_count, int) or chunk_count < 1:
        raise DocumentPlanParseError("document source schema requires chunk_count")
    return _DocumentSourceInfo(
        source_id=source_id,
        default_chunking=dict(chunking),
    )


def _parse_python_module(code: str) -> ast.Module:
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        raise DocumentPlanParseError(
            f"Unable to parse document Python code: {exc.msg}"
        ) from exc


def _required_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    raise DocumentPlanParseError(f"Document Python code must define {name}()")


def _extract_manifest_specs(
    prepare_jobs: ast.FunctionDef,
    source_info: _DocumentSourceInfo,
) -> tuple[list[_ManifestSpec], list[_ResourceViewSpec]]:
    visitor = _PrepareJobsVisitor(source_info)
    visitor.visit_prepare_function(prepare_jobs)
    if not visitor.specs:
        raise DocumentPlanParseError("prepare_jobs() did not create any JobManifest")
    return visitor.specs, visitor.resource_views


def _map_groups_from_specs(
    specs: list[_ManifestSpec],
    source_info: _DocumentSourceInfo,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in specs:
        key = (
            spec.selector.view_id,
            spec.local_instruction,
            spec.local_guidance,
        )
        accumulator = grouped.setdefault(
            key,
            {
                "all_count": 0,
                "alias_counts": {},
            },
        )
        if spec.selector.chunks == CHUNKS_ALL:
            accumulator["all_count"] += 1
            continue
        for alias in spec.selector.chunks:
            alias_counts = accumulator["alias_counts"]
            alias_counts[alias] = alias_counts.get(alias, 0) + 1

    groups = []
    for index, ((view_id, instruction, guidance), accumulator) in enumerate(
        grouped.items()
    ):
        chunks: str | list[str]
        replicas = 1
        if accumulator["all_count"]:
            chunks = CHUNKS_ALL
            replicas = int(accumulator["all_count"])
        else:
            alias_counts = accumulator["alias_counts"]
            count_values = set(alias_counts.values())
            if len(count_values) != 1:
                raise DocumentPlanParseError(
                    "Repeated specific chunk manifests must use the same sample count"
                )
            replicas = int(next(iter(count_values), 1))
            chunks = _sort_chunk_aliases(list(alias_counts))
        if replicas < 1:
            replicas = 1
        group = {
            "id": f"G{index}",
            "op": "map",
            "inputs": {
                "resource_view": view_id,
                "chunks": chunks,
            },
            "params": {
                "local_instruction": instruction,
                "local_guidance": guidance,
                "output_contract": {
                    "format": DOCUMENT_OUTPUT_CONTRACT["format"],
                    "fields": list(DOCUMENT_OUTPUT_CONTRACT["fields"]),
                },
            },
        }
        if replicas > 1:
            group["replicas"] = replicas
        groups.append(
            group
        )
    return groups


def _resource_processing_payload(index: int, view: _ResourceViewSpec) -> dict[str, Any]:
    return {
        "id": f"RP{index}",
        "op": view.op,
        "source": view.source_id,
        "output": view.id,
        "params": dict(view.params),
    }


def _transform_outputs_collector_payload(
    code: str,
    function: ast.FunctionDef,
) -> dict[str, Any]:
    args = function.args.args
    if len(args) != 1:
        raise DocumentPlanParseError("transform_outputs() must accept one jobs argument")
    source = _function_source(code, function)
    return {
        "id": "document_evidence",
        "kind": "minions_transform_outputs",
        "function_name": "transform_outputs",
        "source": source,
        "output": "document_evidence",
    }


def _function_source(code: str, function: ast.FunctionDef) -> str:
    source = ast.get_source_segment(code, function)
    if isinstance(source, str) and source.strip():
        return source.strip()
    if function.lineno is None or function.end_lineno is None:
        raise DocumentPlanParseError(f"Unable to extract {function.name}() source")
    lines = code.splitlines()
    return "\n".join(lines[function.lineno - 1 : function.end_lineno]).strip()


class _PrepareJobsVisitor(ast.NodeVisitor):
    def __init__(self, source_info: _DocumentSourceInfo) -> None:
        self.source_info = source_info
        self.specs: list[_ManifestSpec] = []
        self.resource_views: list[_ResourceViewSpec] = []
        self._resource_view_by_key: dict[str, _ResourceViewSpec] = {}
        self._document_env_stack: list[dict[str, str]] = [{}]
        self._resource_view_env_stack: list[dict[str, _ResourceViewSpec]] = [{}]
        self._chunk_env_stack: list[dict[str, _ChunkSelector]] = [{}]
        self._literal_env_stack: list[dict[str, str]] = [{}]
        self._literal_list_env_stack: list[dict[str, list[str]]] = [{}]
        self._literal_tuple_list_env_stack: list[dict[str, list[tuple[str, ...]]]] = [
            {}
        ]
        self._manifest_env_stack: list[dict[str, _ManifestSpec]] = [{}]

    def visit_prepare_function(self, node: ast.FunctionDef) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        # Nested helpers are not part of the top-level extraction surface.
        return

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        binding_options = self._iteration_binding_options(node.target, node.iter)
        if binding_options:
            for bindings in binding_options:
                with self._scoped_bindings(bindings):
                    for statement in node.body:
                        self.visit(statement)
                    for statement in node.orelse:
                        self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        resource_view = self._resource_view_from_expr(node.value)
        if resource_view is not None:
            for target in node.targets:
                self._bind_resource_view_assignment(target, resource_view)
            return
        manifest_spec = self._manifest_spec_from_expr(node.value)
        if manifest_spec is not None:
            for target in node.targets:
                self._bind_manifest_assignment(target, manifest_spec)
            return
        literal = _literal_assignment_value(node.value)
        if literal is not None:
            for target in node.targets:
                self._bind_literal_assignment(target, literal)
            return
        literal_list = _literal_string_list(node.value)
        if literal_list is not None:
            for target in node.targets:
                self._bind_literal_list_assignment(target, literal_list)
            return
        literal_tuple_list = _literal_string_tuple_list(node.value)
        if literal_tuple_list is not None:
            for target in node.targets:
                self._bind_literal_tuple_list_assignment(target, literal_tuple_list)
            return
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is None:
            return
        resource_view = self._resource_view_from_expr(node.value)
        if resource_view is not None:
            self._bind_resource_view_assignment(node.target, resource_view)
            return
        manifest_spec = self._manifest_spec_from_expr(node.value)
        if manifest_spec is not None:
            self._bind_manifest_assignment(node.target, manifest_spec)
            return
        literal = _literal_assignment_value(node.value)
        if literal is not None:
            self._bind_literal_assignment(node.target, literal)
            return
        literal_list = _literal_string_list(node.value)
        if literal_list is not None:
            self._bind_literal_list_assignment(node.target, literal_list)
            return
        literal_tuple_list = _literal_string_tuple_list(node.value)
        if literal_tuple_list is not None:
            self._bind_literal_tuple_list_assignment(node.target, literal_tuple_list)
            return
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._visit_comprehension(node.elt, node.generators, index=0)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        self._visit_comprehension(node.elt, node.generators, index=0)

    def _visit_comprehension(
        self,
        elt: ast.AST,
        generators: list[ast.comprehension],
        *,
        index: int,
    ) -> None:
        if index >= len(generators):
            self.visit(elt)
            return
        generator = generators[index]
        binding_options = self._iteration_binding_options(
            generator.target,
            generator.iter,
        )
        if not binding_options:
            self._visit_comprehension(elt, generators, index=index + 1)
            return
        for bindings in binding_options:
            with self._scoped_bindings(bindings):
                self._visit_comprehension(elt, generators, index=index + 1)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        appended_spec = self._appended_manifest_spec_from_call(node)
        if appended_spec is not None:
            self.specs.append(appended_spec)
            return
        replicated_specs = self._replicated_manifest_specs_from_call(node)
        if replicated_specs is not None:
            self.specs.extend(replicated_specs)
            return
        manifest_spec = self._manifest_spec_from_expr(node)
        if manifest_spec is not None:
            self.specs.append(manifest_spec)
            return
        self.generic_visit(node)

    def _manifest_spec_from_expr(self, node: ast.AST) -> _ManifestSpec | None:
        if isinstance(node, ast.Name):
            return self._current_manifest_env().get(node.id)
        if isinstance(node, ast.Call) and _is_job_manifest_call(node):
            return _manifest_spec_from_call(
                node,
                chunk_env=self._current_chunk_env(),
                resource_view_env=self._current_resource_view_env(),
                literal_env=self._current_literal_env(),
                source_info=self.source_info,
                default_resource_view=self.default_resource_view,
            )
        return None

    def _appended_manifest_spec_from_call(
        self,
        node: ast.Call,
    ) -> _ManifestSpec | None:
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
            return None
        if len(node.args) != 1:
            return None
        return self._manifest_spec_from_expr(node.args[0])

    def _replicated_manifest_specs_from_call(
        self,
        node: ast.Call,
    ) -> list[_ManifestSpec] | None:
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "extend":
            return None
        if len(node.args) != 1:
            return None
        return self._replicated_manifest_specs_from_expr(node.args[0])

    def _replicated_manifest_specs_from_expr(
        self,
        node: ast.AST,
    ) -> list[_ManifestSpec] | None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            count = _literal_positive_int(node.right)
            elements = self._manifest_specs_from_sequence(node.left)
            if count is not None and elements is not None:
                return elements * count
            count = _literal_positive_int(node.left)
            elements = self._manifest_specs_from_sequence(node.right)
            if count is not None and elements is not None:
                return elements * count
        return self._manifest_specs_from_sequence(node)

    def _manifest_specs_from_sequence(
        self,
        node: ast.AST,
    ) -> list[_ManifestSpec] | None:
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        specs = []
        for item in node.elts:
            spec = self._manifest_spec_from_expr(item)
            if spec is None:
                return None
            specs.append(spec)
        return specs

    def _iteration_binding_options(
        self,
        target: ast.AST,
        iterator: ast.AST,
    ) -> list[dict[str, dict[str, Any]]]:
        chunk_selector = self._selector_from_chunk_iterator(iterator)
        if chunk_selector is not None:
            value_target = _enumerated_value_target(target, iterator)
            chunk_bindings = _bind_target_names(value_target, chunk_selector)
            if chunk_bindings:
                return [{"chunks": chunk_bindings}]

        document_source = self._source_from_document_iterator(iterator)
        if document_source is not None:
            value_target = _enumerated_value_target(target, iterator)
            document_bindings = _bind_target_names(value_target, document_source)
            if document_bindings:
                return [{"documents": document_bindings}]

        literal_values = self._literal_values_from_iterator(iterator)
        if literal_values is not None:
            value_target = _enumerated_value_target(target, iterator)
            return [
                {"literals": _bind_literal_target_names(value_target, value)}
                for value in literal_values
            ]
        return []

    def _selector_from_chunk_iterator(
        self,
        iterator: ast.AST,
    ) -> _ChunkSelector | None:
        value = _enumerated_iterator_value(iterator)
        resource_view = self._resource_view_from_expr(value)
        if resource_view is None:
            return None
        return _ChunkSelector(view_id=resource_view.id, chunks=CHUNKS_ALL)

    def _source_from_document_iterator(self, iterator: ast.AST) -> str | None:
        value = _enumerated_iterator_value(iterator)
        if _is_document_collection(value):
            return self.source_info.source_id
        return None

    def _literal_values_from_iterator(
        self,
        iterator: ast.AST,
    ) -> list[str | tuple[str, ...]] | None:
        value = _enumerated_iterator_value(iterator)
        if isinstance(value, ast.Name):
            literal_list = self._current_literal_list_env().get(value.id)
            if literal_list is not None:
                return literal_list
            return self._current_literal_tuple_list_env().get(value.id)
        literal_tuple_list = _literal_string_tuple_list(value)
        if literal_tuple_list is not None:
            return literal_tuple_list
        return _literal_string_list(value)

    def _resource_view_from_expr(self, node: ast.AST) -> _ResourceViewSpec | None:
        if isinstance(node, ast.Name):
            return self._current_resource_view_env().get(node.id)
        if isinstance(node, ast.Call) and _is_document_chunking_call(node):
            return self._resource_view_from_call(node)
        return None

    def _resource_view_from_call(self, call: ast.Call) -> _ResourceViewSpec:
        raw_op = _call_name(call.func)
        op = DOCUMENT_CHUNKING_ALIASES.get(raw_op, raw_op)
        if op not in DOCUMENT_CHUNKING_FUNCTIONS:
            raise DocumentPlanParseError(f"Unsupported chunking function: {raw_op}")
        if not call.args:
            raise DocumentPlanParseError(f"{op} requires a document argument")
        source_id = self._source_from_document_expr(call.args[0])
        params = _chunking_params_from_call(call, op)
        key = _resource_view_key(source_id=source_id, op=op, params=params)
        existing = self._resource_view_by_key.get(key)
        if existing is not None:
            return existing
        view = _ResourceViewSpec(
            id=f"CV{len(self.resource_views)}",
            source_id=source_id,
            op=op,
            params=params,
        )
        self.resource_views.append(view)
        self._resource_view_by_key[key] = view
        return view

    def _source_from_document_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            source_id = self._current_document_env().get(node.id)
            if source_id is not None:
                return source_id
            # A single-source worker may pass its document variable directly
            # to the chunker, e.g. chunk_by_section(ctx).
            return self.source_info.source_id
        if isinstance(node, ast.Subscript) and _is_document_collection(node.value):
            _constant_int(_subscript_slice(node))
            return self.source_info.source_id
        raise DocumentPlanParseError(
            "Unsupported chunking document expression at line "
            f"{getattr(node, 'lineno', '?')}"
        )

    def default_resource_view(self) -> _ResourceViewSpec:
        params = {
            "max_chunk_size": int(
                self.source_info.default_chunking.get("size", 3000) or 3000
            ),
            "overlap": int(
                self.source_info.default_chunking.get("overlap", 20) or 0
            ),
        }
        key = _resource_view_key(
            source_id=self.source_info.source_id,
            op="chunk_by_section",
            params=params,
        )
        existing = self._resource_view_by_key.get(key)
        if existing is not None:
            return existing
        view = _ResourceViewSpec(
            id=f"CV{len(self.resource_views)}",
            source_id=self.source_info.source_id,
            op="chunk_by_section",
            params=params,
        )
        self.resource_views.append(view)
        self._resource_view_by_key[key] = view
        return view

    @contextmanager
    def _scoped_bindings(
        self,
        bindings: dict[str, dict[str, Any]],
    ) -> Iterator[None]:
        with ExitStack() as stack:
            if bindings.get("documents"):
                stack.enter_context(self._document_scope(bindings["documents"]))
            if bindings.get("resource_views"):
                stack.enter_context(
                    self._resource_view_scope(bindings["resource_views"])
                )
            if bindings.get("chunks"):
                stack.enter_context(self._chunk_scope(bindings["chunks"]))
            if bindings.get("literals"):
                stack.enter_context(self._literal_scope(bindings["literals"]))
            yield

    @contextmanager
    def _document_scope(self, bindings: dict[str, str]) -> Iterator[None]:
        if not bindings:
            yield
            return
        next_env = dict(self._current_document_env())
        next_env.update(bindings)
        self._document_env_stack.append(next_env)
        try:
            yield
        finally:
            self._document_env_stack.pop()

    @contextmanager
    def _resource_view_scope(
        self,
        bindings: dict[str, _ResourceViewSpec],
    ) -> Iterator[None]:
        if not bindings:
            yield
            return
        next_env = dict(self._current_resource_view_env())
        next_env.update(bindings)
        self._resource_view_env_stack.append(next_env)
        try:
            yield
        finally:
            self._resource_view_env_stack.pop()

    @contextmanager
    def _chunk_scope(self, bindings: dict[str, _ChunkSelector]) -> Iterator[None]:
        if not bindings:
            yield
            return
        next_env = dict(self._current_chunk_env())
        next_env.update(bindings)
        self._chunk_env_stack.append(next_env)
        try:
            yield
        finally:
            self._chunk_env_stack.pop()

    @contextmanager
    def _literal_scope(self, bindings: dict[str, str]) -> Iterator[None]:
        if not bindings:
            yield
            return
        next_env = dict(self._current_literal_env())
        next_env.update(bindings)
        self._literal_env_stack.append(next_env)
        try:
            yield
        finally:
            self._literal_env_stack.pop()

    def _current_document_env(self) -> dict[str, str]:
        return self._document_env_stack[-1]

    def _current_resource_view_env(self) -> dict[str, _ResourceViewSpec]:
        return self._resource_view_env_stack[-1]

    def _current_chunk_env(self) -> dict[str, _ChunkSelector]:
        return self._chunk_env_stack[-1]

    def _current_literal_env(self) -> dict[str, str]:
        return self._literal_env_stack[-1]

    def _current_literal_list_env(self) -> dict[str, list[str]]:
        return self._literal_list_env_stack[-1]

    def _current_literal_tuple_list_env(self) -> dict[str, list[tuple[str, ...]]]:
        return self._literal_tuple_list_env_stack[-1]

    def _current_manifest_env(self) -> dict[str, _ManifestSpec]:
        return self._manifest_env_stack[-1]

    def _bind_literal_assignment(self, target: ast.AST, value: str) -> None:
        if isinstance(target, ast.Name):
            self._current_literal_env()[target.id] = value

    def _bind_literal_list_assignment(
        self,
        target: ast.AST,
        value: list[str],
    ) -> None:
        if isinstance(target, ast.Name):
            self._current_literal_list_env()[target.id] = list(value)

    def _bind_literal_tuple_list_assignment(
        self,
        target: ast.AST,
        value: list[tuple[str, ...]],
    ) -> None:
        if isinstance(target, ast.Name):
            self._current_literal_tuple_list_env()[target.id] = list(value)

    def _bind_resource_view_assignment(
        self,
        target: ast.AST,
        value: _ResourceViewSpec,
    ) -> None:
        if isinstance(target, ast.Name):
            self._current_resource_view_env()[target.id] = value

    def _bind_manifest_assignment(
        self,
        target: ast.AST,
        value: _ManifestSpec,
    ) -> None:
        if isinstance(target, ast.Name):
            self._current_manifest_env()[target.id] = value


def _manifest_spec_from_call(
    call: ast.Call,
    *,
    chunk_env: dict[str, _ChunkSelector],
    resource_view_env: dict[str, _ResourceViewSpec],
    literal_env: dict[str, str],
    source_info: _DocumentSourceInfo,
    default_resource_view: Any,
) -> _ManifestSpec:
    call_name = _call_name(call.func)
    if call_name == "Job":
        task_node = _call_argument(call, name="instruction", position=0)
        chunk_node = _call_argument(call, name="chunk", position=1)
    else:
        task_node = _call_argument(call, name="task", position=1)
        chunk_node = _call_argument(call, name="chunk", position=0)
    advice_node = _call_argument(call, name="advice", position=2)
    instruction = _literal_string(
        task_node,
        f"{call_name}.task",
        literal_env=literal_env,
    ).strip()
    if not instruction:
        raise DocumentPlanParseError(f"{call_name}.task must be a non-empty string")
    guidance = _literal_string(
        advice_node,
        f"{call_name}.advice",
        literal_env=literal_env,
        allow_none=True,
    ).strip()
    selector = _manifest_chunk_selector(
        call,
        chunk_node=chunk_node,
        chunk_env=chunk_env,
        resource_view_env=resource_view_env,
        source_info=source_info,
        default_resource_view=default_resource_view,
    )
    return _ManifestSpec(
        selector=selector,
        local_instruction=instruction,
        local_guidance=guidance,
    )


def _manifest_chunk_selector(
    call: ast.Call,
    *,
    chunk_node: ast.AST | None,
    chunk_env: dict[str, _ChunkSelector],
    resource_view_env: dict[str, _ResourceViewSpec],
    source_info: _DocumentSourceInfo,
    default_resource_view: Any,
) -> _ChunkSelector:
    chunk_id_node = _call_argument(call, name="chunk_id", position=3)
    if chunk_node is not None:
        return _selector_from_chunk_expr(
            chunk_node,
            chunk_env=chunk_env,
            resource_view_env=resource_view_env,
            source_info=source_info,
        )
    if chunk_id_node is not None:
        # Keep literal chunk aliases recoverable even when a remote response
        # omitted the chunk object itself.
        default_view = default_resource_view()
        return _ChunkSelector(
            view_id=default_view.id,
            chunks=(_selector_from_chunk_id(chunk_id_node),),
        )
    raise DocumentPlanParseError("JobManifest must include chunk or chunk_id")


def _selector_from_chunk_expr(
    node: ast.AST,
    *,
    chunk_env: dict[str, _ChunkSelector],
    resource_view_env: dict[str, _ResourceViewSpec],
    source_info: _DocumentSourceInfo,
) -> _ChunkSelector:
    if isinstance(node, ast.Name) and node.id in chunk_env:
        return chunk_env[node.id]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        resource_view = resource_view_env.get(node.value.id)
        if resource_view is None:
            raise DocumentPlanParseError(
                "Unsupported JobManifest.chunk collection at line "
                f"{getattr(node, 'lineno', '?')}"
            )
        index = _constant_int(_subscript_slice(node))
        return _ChunkSelector(
            view_id=resource_view.id,
            chunks=(_normalize_chunk_alias(index),),
        )
    raise DocumentPlanParseError(
        "Unsupported JobManifest.chunk expression at line "
        f"{getattr(node, 'lineno', '?')}"
    )


def _selector_from_chunk_id(
    node: ast.AST,
) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
        return _normalize_chunk_alias(node.value)
    raise DocumentPlanParseError("JobManifest.chunk_id must be a literal chunk alias")


def _bind_target_names(target: ast.AST, value: Any) -> dict[str, Any]:
    if isinstance(target, ast.Name):
        return {target.id: value}
    if isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            if isinstance(item, ast.Name):
                return {item.id: value}
    return {}


def _bind_literal_target_names(
    target: ast.AST,
    value: str | tuple[str, ...],
) -> dict[str, str]:
    if isinstance(target, ast.Name) and isinstance(value, str):
        return {target.id: value}
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, tuple)
        and len(target.elts) == len(value)
    ):
        bindings: dict[str, str] = {}
        for item, literal in zip(target.elts, value):
            if not isinstance(item, ast.Name):
                return {}
            bindings[item.id] = literal
        return bindings
    return _bind_target_names(target, value) if isinstance(value, str) else {}


def _enumerated_iterator_value(iterator: ast.AST) -> ast.AST:
    if (
        isinstance(iterator, ast.Call)
        and _is_name(iterator.func, "enumerate")
        and iterator.args
    ):
        return iterator.args[0]
    return iterator


def _enumerated_value_target(target: ast.AST, iterator: ast.AST) -> ast.AST:
    if (
        isinstance(iterator, ast.Call)
        and _is_name(iterator.func, "enumerate")
        and isinstance(target, (ast.Tuple, ast.List))
        and len(target.elts) >= 2
    ):
        return target.elts[1]
    return target


def _is_document_collection(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in DOCUMENT_COLLECTION_NAMES


def _is_job_manifest_call(node: ast.Call) -> bool:
    return _call_name(node.func) in {"JobManifest", "Job"}


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_document_chunking_call(node: ast.Call) -> bool:
    return _call_name(node.func) in DOCUMENT_CHUNKING_ALIASES


def _call_argument(call: ast.Call, *, name: str, position: int) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    if len(call.args) > position:
        return call.args[position]
    return None


def _chunking_params_from_call(call: ast.Call, op: str) -> dict[str, Any]:
    if op == "chunk_by_section":
        return {
            "max_chunk_size": _literal_int_argument(
                call,
                names=("max_chunk_size",),
                position=1,
                default=3000,
            ),
            "overlap": _literal_int_argument(
                call,
                names=("overlap",),
                position=2,
                default=20,
            ),
        }
    if op == "chunk_by_page":
        markers = _call_argument(call, name="page_markers", position=1)
        if markers is not None and not (
            isinstance(markers, ast.Constant) and markers.value is None
        ):
            raise DocumentPlanParseError(
                "chunk_by_page page_markers must be omitted or None"
            )
        return {}
    raise DocumentPlanParseError(f"Unsupported chunking function: {op}")


def _literal_int_argument(
    call: ast.Call,
    *,
    names: tuple[str, ...],
    position: int,
    default: int,
) -> int:
    value_node = None
    for name in names:
        value_node = _call_argument(call, name=name, position=position)
        if value_node is not None:
            break
    if value_node is None:
        return default
    return _constant_int(value_node)


def _resource_view_key(
    *,
    source_id: str,
    op: str,
    params: dict[str, Any],
) -> str:
    return f"{source_id}:{op}:{sorted(params.items())}"


def _literal_string(
    node: ast.AST | None,
    field_name: str,
    *,
    literal_env: dict[str, str],
    allow_none: bool = False,
) -> str:
    if node is None:
        if allow_none:
            return ""
        raise DocumentPlanParseError(f"{field_name} is required")
    if isinstance(node, ast.Name) and node.id in literal_env:
        return literal_env[node.id]
    if _is_literal_strip_call(node):
        return _literal_string(
            node.func.value,
            field_name,
            literal_env=literal_env,
            allow_none=allow_none,
        ).strip()
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if allow_none and node.value is None:
            return ""
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for item in node.values:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                raise DocumentPlanParseError(f"{field_name} must be a literal string")
            pieces.append(item.value)
        return "".join(pieces)
    raise DocumentPlanParseError(f"{field_name} must be a literal string")


def _is_literal_strip_call(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "strip"
        and not node.args
        and not node.keywords
    )


def _literal_assignment_value(node: ast.AST) -> str | None:
    try:
        return _literal_string(
            node,
            "literal assignment",
            literal_env={},
            allow_none=False,
        )
    except DocumentPlanParseError:
        return None


def _literal_string_list(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = []
    for item in node.elts:
        try:
            values.append(
                _literal_string(
                    item,
                    "literal list item",
                    literal_env={},
                    allow_none=False,
                )
            )
        except DocumentPlanParseError:
            return None
    return values


def _literal_string_tuple_list(node: ast.AST) -> list[tuple[str, ...]] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    rows: list[tuple[str, ...]] = []
    for item in node.elts:
        if not isinstance(item, (ast.List, ast.Tuple)):
            return None
        values: list[str] = []
        for child in item.elts:
            try:
                values.append(
                    _literal_string(
                        child,
                        "literal tuple item",
                        literal_env={},
                        allow_none=False,
                    )
                )
            except DocumentPlanParseError:
                return None
        rows.append(tuple(values))
    return rows


def _subscript_slice(node: ast.Subscript) -> ast.AST:
    return node.slice


def _constant_int(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    raise DocumentPlanParseError("Chunk subscript must use a literal integer index")


def _literal_positive_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value > 0:
        return node.value
    return None


def _normalize_chunk_alias(
    value: int | str,
) -> str:
    if isinstance(value, int):
        index = value
    else:
        text = value.strip()
        match = re.fullmatch(r"chunk_(\d+)", text)
        if match:
            index = int(match.group(1))
        elif text.isdigit():
            index = int(text)
        else:
            raise DocumentPlanParseError(f"Unsupported chunk alias: {value!r}")
    if index < 0:
        raise DocumentPlanParseError(f"Unsupported chunk alias: {value!r}")
    alias = f"chunk_{index}"
    return alias


def _sort_chunk_aliases(
    aliases: list[str],
) -> list[str]:
    return sorted(aliases, key=lambda alias: int(alias.split("_", 1)[1]))


def _extract_fenced_python_blocks(text: str) -> list[str]:
    blocks = []
    fence_pattern = re.compile(
        r"```(?:python|py)?\s*(.*?)```",
        flags=re.DOTALL | re.I,
    )
    for match in fence_pattern.finditer(text):
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks
