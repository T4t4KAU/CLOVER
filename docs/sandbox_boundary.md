# Sandbox Boundary

## Scope

This document describes the current CLOVER Agent sandbox: what it is designed
to provide, what it can do today, and what it explicitly does not guarantee.

Current implementation:

- Sandbox wrapper: `clover/executor/sandbox/core.py`
- Table reasoning policy: `clover/executor/sandbox/table_reasoning.py`
- Resource projection: `clover/executor/handles/projector.py`
- Sandbox handles: `clover/executor/handles`
- Agent Loop prompt: `clover/executor/agents/templates/table_reasoning/agent_loop.md`

## Design Position

The sandbox is a node-local workspace boundary. It is designed to keep the
NodeAgent focused on the current physical node and to prevent accidental access
to global CLOVER state.

It is not an operating-system security boundary. It does not currently use
containers, chroot, seccomp, a separate restricted user, or filesystem-level
isolation. The current goal is controlled context and lifecycle management for
local task execution, not hostile-code containment.

## What The Sandbox Provides

The sandbox provides these guarantees at the CLOVER application level:

- The Agent sees only the current node task capsule.
- The Agent receives copied or projected data handles, not the global
  `ResourceStore`.
- The Agent does not receive the full DAG, full Task DSL, Remote LLM session,
  source code paths, or all table resources.
- The Agent cannot mutate executor-owned resource objects directly.
- The Agent's temporary workspace is cleared when the loop finishes.
- Candidate outputs are normalized and validated before they become node
  outputs.
- Agent actions are recorded as compact traces.

These properties are enough for the current table reasoning fallback, where the
Agent's job is to complete one local operation using the data already assigned
to that node.

## What The Sandbox Does Not Guarantee

The current sandbox should not be treated as a secure jail.

It does not guarantee:

- protection against malicious Python code
- filesystem isolation
- network isolation
- CPU or memory hard limits
- prevention of all Python introspection tricks
- protection if the model intentionally tries to escape the workspace
- safe execution of untrusted third-party code

The implementation restricts imports and builtins, but that is a lightweight
guardrail, not a complete Python security model.

## Visible Context

The sandbox view sent to the Local SLM contains a compact JSON-safe summary:

- `task.operation`
- `task.params`
- `task.input_handles`
- `task.resource_handles`
- optional `task.operation_note`
- `task.output_contract`
- summaries of dependency handles
- summaries of source handles
- optional local `tool` metadata
- feedback from failed automatic execution
- observations from previous Agent Loop steps

The Python workspace contains the actual projected values:

- `inputs`: dependency handles such as `dep_0`
- `resources`: source handles such as `source_0`
- direct handle variables such as `dep_0` and `source_0`
- `task`
- `params`
- `tool`
- `helpers`
- `pd`
- `np`
- `TableHandle`
- `PandasTable`

The Agent is not told about the sandbox as a system component. From the Agent's
perspective, this is simply the available local workspace for completing the
operation.

## Data Projection

Resource projection is the main mechanism that controls what data the Agent can
see.

`SandboxProjector` converts executor resources into sandbox-visible handles:

- table resources become `TableHandle`
- scalar or JSON-like values become `ValueHandle`

`TableHandle` keeps the dataframe and table metadata together:

- `.frame`
- `.group_keys`
- `resource_id`
- `role`
- `metadata`

The handle owns a copied dataframe. Mutating it does not mutate the executor's
stored resource. This preserves node-local freedom while protecting shared
runtime state.

## Action Surface

The current Agent Loop supports three actions:

```json
{"action": "run_python", "code": "..."}
```

Run Python code inside the node-local workspace. The code should compute the
current node output and assign it to `result`.

```json
{"action": "submit_result", "name": "result"}
```

Submit an existing workspace variable as the node output.

```json
{"action": "abort", "reason": "..."}
```

Stop the loop and return a node failure.

The design intentionally gives table reasoning code broad access to its own
projected node resources. It does not force the Agent to use a specific static
tool. The tool reference is optional: the Agent may call `tool.run()` or write
equivalent local code.

## Python Workspace

The workspace is implemented with separate globals and locals passed to
`exec()`. It provides:

- pandas and numpy
- small table reasoning helpers
- table/value handles
- an optional local tool reference
- a restricted builtins dictionary
- a restricted import hook

Allowed import roots currently include common local computation modules such as
`pandas`, `numpy`, `math`, `statistics`, `datetime`, `itertools`, `collections`,
`functools`, `operator`, `decimal`, `json`, and `re`.

Imports such as `os`, `sys`, and `subprocess` are not part of the allowed import
roots. This reduces accidental unsafe behavior but is not a hard security
guarantee against adversarial code.

## Output Validation

After `run_python` or `submit_result`, the sandbox validates the candidate
output against the node operation.

For table-producing operations, accepted outputs include:

- `TableHandle`
- `pandas.DataFrame`
- `PandasTable`
- table-like objects with a dataframe payload

They are normalized into `PandasTable`.

For `FormatAnswer`, the output must be a JSON-ready answer value, not a table.
List-like answer values are normalized when possible.

This validation is structural. It checks that the output can continue through
the executor. It does not prove that the answer is semantically correct for the
original user question. Reporter remains responsible for final answer review.

## Lifecycle

One sandbox state is created for one Agent Loop invocation.

Lifecycle:

```text
NodeAgent starts Agent Loop
  -> sandbox.start()
  -> project current node dependencies and resources
  -> create Python workspace
  -> repeat Local SLM action + sandbox observation
  -> accept output, abort, or reach iteration limit
  -> sandbox.close()
  -> clear handles, feedback, locals, globals, and traces
```

The accepted result is detached from the sandbox before cleanup and returned to
the Executor. Temporary variables and intermediate objects created by the Agent
are discarded at loop end.

## Error Feedback

When a local automatic attempt fails, the sandbox gives the Agent compact
feedback:

- a generic failure message
- compact error type and message
- later observations from failed Python actions
- workspace summaries after non-terminal attempts

If the Agent produces a valid output, the loop stops immediately. There is no
extra success observation sent back to the model.

## Relationship To Fast Path

Fast Path is the normal route for table reasoning nodes. The sandbox is only
used when:

- Fast Path misses, or
- Fast Path execution fails with a recoverable local error.

The sandbox does not repair cloud-side SQL mistakes. Examples of cloud-side
mistakes include unknown columns, unknown tables, invalid schema references, or
SQL that answers the wrong question. Those are routed to Reporter and Remote
LLM SQL repair.

## Current Capability Boundary

The table reasoning sandbox is capable of:

- inspecting all data assigned to the current node
- using pandas/numpy operations on copied table handles
- calling the optional local static tool reference
- writing equivalent custom local code
- preserving table metadata such as group keys
- returning table outputs or answer values
- recovering from many local data-format and local execution issues

It is not intended to:

- change the global plan
- edit static tool source code
- edit local files
- call Remote LLM
- inspect unrelated tables
- access prior or future DAG nodes
- decide final task correctness

## Why No Filesystem Isolation Yet

The current table reasoning Agent is expected to perform harmless local data
operations on copied resources. Full filesystem isolation would add complexity
before the system needs it for the current benchmark path.

For now, CLOVER relies on:

- limited context exposure
- copied projected handles
- restricted imports and builtins
- no shell action
- no explicit filesystem handles
- workspace cleanup after the loop

This is sufficient for controlled local SLM fallback experiments, but it should
not be described as secure sandboxing.

## Deferred To V3

- Subprocess-based Agent execution.
- Temporary working directory per Agent Loop.
- Optional filesystem isolation.
- CPU and memory limits.
- Timeout enforcement for Agent code.
- Stronger import and builtins policy.
- Structured IPC for returning only approved outputs.
- Per-task sandbox policies beyond table reasoning.
- More detailed sandbox audit traces.
