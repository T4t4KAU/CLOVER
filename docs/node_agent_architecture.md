# NodeAgent Architecture

## Scope

NodeAgent is the executor-side abstraction for running one physical DAG node.
It does not own the global task, the full DAG, or global resources. The
Executor schedules a ready node, builds a node-local context, and delegates the
operation to the task-specific NodeAgent.

Current implementation:

- Base abstraction: `clover/executor/agents/base.py`
- Table reasoning agent: `clover/executor/agents/table_reasoning.py`
- Sandbox wrapper: `clover/executor/sandbox/core.py`
- Table reasoning sandbox policy: `clover/executor/sandbox/table_reasoning.py`
- Agent prompt template: `clover/executor/agents/templates/table_reasoning/agent_loop.md`

## Execution Workflow

1. The Executor validates the physical plan and finds a ready node.
2. `ExecutionContext.node_context()` creates a `NodeResourceView` for that node.
3. The resource view pins only the node dependencies and source resources.
4. `build_node_agent()` instantiates the task-specific NodeAgent.
5. `BaseNodeAgent.run()` tries Fast Path first.
6. If Fast Path succeeds, the deterministic local tool result becomes the node
   output.
7. If Fast Path misses, or a recoverable local Fast Path execution error occurs,
   the Agent Loop is allowed to run.
8. The NodeAgent returns a `NodeExecutionRecord`.
9. The Executor stores the output in `ResourceStore`, releases consumed
   intermediates when possible, and schedules the next ready node.

The NodeAgent returns records instead of raising most node failures. This keeps
execution traces compact and lets the runtime route failures to Reporter or SQL
repair logic.

## Fast Path

For table reasoning, Fast Path is the default path and should handle almost all
normal nodes.

Fast Path is considered a hit only when all conditions hold:

- `task_type` is supported by `TableReasoningNodeAgent`.
- The node operation exists in `TABLE_REASONING_STATIC_TOOLS`.
- All dependency outputs are already available.
- All source resources are available.
- `build_static_tool_call()` can validate and normalize the node call.

Execution currently uses `PandasTableReasoningExecutor`. The backend is an
implementation detail of the first working path; the NodeAgent boundary keeps
room for future backends.

Fast Path should not repair cloud-side mistakes. For example, unknown columns,
unknown resources, invalid schema references, or bad SQL semantics belong to
Commander/Reporter retry. NodeAgent recovery is limited to local execution
problems where the node operation is still well-defined.

## Agent Loop

The Agent Loop is a local fallback for completing the current node operation
when deterministic tools cannot finish it. It is not a SQL repair loop and does
not modify global plans.

Current triggers:

- `fast_path_miss`
- `fast_path_execution_error` when the error is classified as locally
  recoverable

The table reasoning Agent Loop uses a compact ReAct-like protocol with a small
set of actions:

- `run_python`: execute Python code in the node-local workspace.
- `submit_result`: submit an existing workspace variable as the output.
- `abort`: stop the loop with a failure reason.

The prompt is completion-oriented: the model is asked to implement the current
local table operation, assign the output to `result`, and avoid rewriting the
broader task.

## Visible Context

The Agent Loop sees a node-local workspace, not the full system state. The view
contains:

- `task.operation`
- `task.params`
- `task.input_handles`
- `task.resource_handles`
- optional `task.operation_note`
- `task.output_contract`
- summaries of current dependency handles
- summaries of current source handles
- an optional `tool` reference for the exact current operation
- feedback from prior failed local attempts
- observations from previous Agent Loop steps

The Python workspace provides:

- `inputs`: dependency handles keyed as `dep_0`, `dep_1`, ...
- `resources`: source handles keyed as `source_0`, `source_1`, ...
- direct variables for each handle name, such as `dep_0`
- `task` and `params`
- `tool`, an optional local static-tool reference
- `helpers`, with small table reasoning helpers
- `pd`, `np`, `TableHandle`, and `PandasTable`

The prompt does not expose global DAG structure, table schemas unrelated to the
node, remote conversation state, filesystem paths, or source code paths.

## Output Contract

The sandbox policy validates and normalizes candidate outputs before the
NodeAgent accepts them:

- Table-producing operations must return a table-like value.
- `TableHandle`, `pandas.DataFrame`, and `PandasTable` are normalized to
  `PandasTable`.
- Group metadata is preserved through `TableHandle.group_keys`.
- `FormatAnswer` must return a JSON-ready answer value.

This validation is structural. Final semantic correctness remains Reporter's
responsibility.

## Sandbox Boundary

The current sandbox is a task-local workspace boundary, not an operating-system
security boundary. It limits what the model is shown and gives the Agent copies
or projected handles for its node-owned data. It also clears workspace objects
when the loop ends.

The implementation has restricted builtins and import roots, but it should not
be treated as a hostile-code isolation mechanism. Strong filesystem or process
isolation is intentionally deferred.

## Extension Points

To add a new task-specific NodeAgent:

1. Implement a `BaseNodeAgent` subclass.
2. Register it in `clover/executor/agents/registry.py`.
3. Provide Fast Path capability checks and execution.
4. Add a sandbox policy only if Agent Loop fallback is required.
5. Add task-specific prompt templates under `clover/executor/agents/templates`.

## Deferred To V3

- Node-level worker parallelism inside one Executor instance.
- OS-level sandboxing or subprocess isolation.
- Agent requests to Remote LLM for node-level guidance.
- More task-specific workspace policies beyond table reasoning.
- Stronger runtime limits for slow or memory-heavy Agent code.
- Richer trace export for Agent Loop debugging.
