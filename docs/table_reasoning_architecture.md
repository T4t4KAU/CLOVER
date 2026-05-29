# Table Reasoning Architecture

## Scope

The public task type for table reasoning is `table_reasoning`. CLOVER routes
that task type to the latest table reasoning pipeline by default.

Internally, the implementation still keeps two pipeline modes:

- v1: one table reasoning question per Remote LLM request and one local
  execution flow.
- v2: multiple same-table questions batched in one Remote LLM conversation,
  lowered into multiple Logic DAGs, merged into one physical DAG when possible,
  and executed locally.

Both versions share the same lower-level Planner, Optimizer, Executor,
NodeAgent, static tools, resource store, and Reporter concepts. v2 mainly adds
batching, Remote session reuse, answer-key lifecycle management, DAG merging,
and partial retry.

Current implementation:

- v1 runtime: `clover/runtime/table_reasoning/v1.py`
- v2 runtime: `clover/runtime/table_reasoning/v2.py`
- SQL parser and Logic DAG compiler: `clover/planner/sql_parser.py`
- SQL list parser for v2 Commander output: `clover/planner/sql_list_parser.py`
- Optimizer and v2 DAG merge: `clover/optimizer/core.py`
- Executor and NodeAgent: `clover/executor`
- Static table tools: `clover/tools/table_reasoning`
- Commander prompts: `clover/commander/templates/table_reasoning`
- Reporter prompts: `clover/reporter/templates/table_reasoning`

## Internal Pipeline Matrix

| Aspect | v1 mode | v2 mode |
| --- | --- | --- |
| Input unit | One question | Multiple same-table questions |
| Remote Commander output | One SQL statement | JSON array of SQL strings |
| Answer naming | Usually `answer` | Globally unique `answer_1`, `answer_2`, ... |
| Remote session | Per case in eval | Reused per source file |
| Planner input | One SQL | Each SQL parsed as v1 |
| Logic DAG shape | `nodes` + `edges` | `subtasks`, each with one v1 Logic DAG |
| Optimizer output | One physical plan | Merged physical plan |
| DAG reuse | None across cases | Equivalent nodes reused inside one merged batch |
| Reporter decision | One answer or one retry SQL | Per-answer accept/retry with keyed SQL |
| Retry scope | Whole case | Only rejected or failed answer keys |

## Shared High-Level Flow

Both versions follow the same conceptual path:

```text
Task DSL
  -> Preprocess
  -> Remote Commander SQL
  -> Local Planner Logic DAG
  -> Optimizer Physical DAG
  -> Executor NodeAgent execution
  -> Reporter final answer or retry SQL
```

Important boundary:

- Remote Commander returns SQL only.
- Local Planner generates the Logic DAG.
- Optimizer generates the Physical DAG.
- Executor runs Physical DAG nodes.
- Reporter can accept answers or request new SQL.

Reporter retry always returns SQL, not Logic DAG JSON. The new SQL goes back
through Local Planner before reaching Optimizer and Executor.

## Preprocess Outputs

Preprocessing converts the original task DSL into three payloads:

- `remote_dsl`: schema-safe payload for Remote Commander. It includes table
  ids, table schema, question text, and answer type. It does not include local
  filesystem paths.
- `local_dsl`: local execution payload. It includes resolved local source
  information and answer metadata.
- `context`: local runtime context, including `base_dir` and `source_map`.

For v2, individual v1-style cases are converted into `TaskItem` records and
then grouped by resolved source file. The runtime builds a v2 remote DSL for a
Commander batch:

```json
{
  "task_type": "table_reasoning_v2",
  "questions": [
    "Is the person with the highest net worth self-made?",
    "What is the country of the person with the highest net worth?"
  ],
  "sources": [
    {
      "id": "table_1",
      "type": "table",
      "schema": {
        "columns": ["finalWorth", "selfMade", "country"]
      }
    }
  ],
  "answers": [
    {"name": "answer_1", "type": "boolean"},
    {"name": "answer_2", "type": "string"}
  ]
}
```

## Commander Contract

### v1

Commander receives one table reasoning task and must return exactly one SQL
statement:

```sql
SELECT COUNT(*) AS "answer" FROM "table_1";
```

The prompt requires:

- use only provided table ids
- use table ids as SQL table names
- quote table and column identifiers
- use only schema columns
- return one SQL statement and no extra text

### v2

Commander receives a batch of same-table questions and must return exactly one
JSON array of SQL strings:

```json
[
  "SELECT \"selfMade\" AS \"answer_1\" FROM \"table_1\" ORDER BY \"finalWorth\" DESC LIMIT 1;",
  "SELECT \"country\" AS \"answer_2\" FROM \"table_1\" ORDER BY \"finalWorth\" DESC LIMIT 1;"
]
```

The array length must equal the number of questions. SQL at index `i` must
answer `questions[i]` and return the answer using `answers[i].name` as the
quoted output alias.

The first v2 Commander batch for a table sends schema and constraints. Later
batches reuse the same Remote LLM session and send only the new batch payload.
The follow-up prompt tells the model to reuse the existing schema and SQL
constraints from the conversation.

## Logic DAG

The Planner parses Commander SQL using `sqlglot`, validates that it is a
read-only SELECT-like query, checks table references against the provided DSL,
and lowers it into CLOVER's atomic table operation set.

Supported logical operations include:

- `Scan`
- `Filter`
- `Project`
- `Derive`
- `Aggregate`
- `Group`
- `Sort`
- `Limit`
- `Distinct`
- `Join`
- `SetOp`
- `RepeatUnion`
- `FormatAnswer`

Each Logic DAG node has the same core fields:

```json
{
  "id": "N1",
  "op": "Sort",
  "dependency": ["T0"],
  "input": [],
  "params": {
    "keys": [
      {
        "expr": {"type": "column", "name": "finalWorth"},
        "direction": "DESC",
        "nulls": "LAST"
      }
    ]
  },
  "output": "T1"
}
```

Field meanings:

- `id`: node id inside this DAG.
- `op`: atomic operation name.
- `dependency`: upstream intermediate output names.
- `input`: external source ids, normally used by `Scan`.
- `params`: operation-specific parameters.
- `output`: intermediate or answer output name.

Dependencies name intermediate outputs, not upstream node ids. `edges` are
derived from node dependencies:

```json
[
  {"from": "N0", "to": "N1"},
  {"from": "N1", "to": "N2"}
]
```

## v1 Logic DAG Example

SQL:

```sql
SELECT "selfMade" AS "answer"
FROM "table_1"
ORDER BY "finalWorth" DESC
LIMIT 1;
```

Logic DAG:

```json
{
  "task_type": "table_reasoning_v1",
  "nodes": [
    {
      "id": "N0",
      "op": "Scan",
      "dependency": [],
      "input": ["table_1"],
      "params": {"source": "table_1"},
      "output": "T0"
    },
    {
      "id": "N1",
      "op": "Sort",
      "dependency": ["T0"],
      "input": [],
      "params": {
        "keys": [
          {
            "expr": {"type": "column", "name": "finalWorth"},
            "direction": "DESC",
            "nulls": "LAST"
          }
        ]
      },
      "output": "T1"
    },
    {
      "id": "N2",
      "op": "Limit",
      "dependency": ["T1"],
      "input": [],
      "params": {"count": 1},
      "output": "T2"
    },
    {
      "id": "N3",
      "op": "Project",
      "dependency": ["T2"],
      "input": [],
      "params": {
        "expressions": [
          {"expr": {"type": "column", "name": "selfMade"}}
        ]
      },
      "output": "T3"
    },
    {
      "id": "N4",
      "op": "FormatAnswer",
      "dependency": ["T3"],
      "input": [],
      "params": {
        "answer": {"name": "answer", "type": "boolean"}
      },
      "output": "answer"
    }
  ],
  "edges": [
    {"from": "N0", "to": "N1"},
    {"from": "N1", "to": "N2"},
    {"from": "N2", "to": "N3"},
    {"from": "N3", "to": "N4"}
  ]
}
```

`FormatAnswer` is always the final semantic boundary between local table
operations and the answer value expected by Reporter.

## Physical DAG

The Optimizer turns a Logic DAG into a Physical DAG by adding local execution
details while preserving logical dependencies.

Main optimizer steps:

1. `ResourceBindingStrategy` binds source ids to local absolute paths and
   schema metadata from `context` and `local_dsl`.
2. `NodeAnnotationStrategy` annotates nodes with `output_type` and
   task-specific `instruction`.
3. v2 additionally lowers each subtask separately and then merges physical
   plans.

Physical plan shape:

```json
{
  "task_type": "table_reasoning_v1",
  "resources": [
    {
      "id": "table_1",
      "type": "table",
      "path": "/abs/path/table.csv",
      "format": "csv",
      "schema": {
        "columns": ["finalWorth", "selfMade", "country"]
      }
    }
  ],
  "nodes": [
    {
      "id": "N0",
      "op": "Scan",
      "dependency": [],
      "input": ["table_1"],
      "params": {"source": "table_1"},
      "output": "T0",
      "output_type": "table",
      "instruction": ""
    }
  ],
  "edges": []
}
```

The Executor only sees physical plans. It does not call Commander or Planner.

## v1 Runtime

v1 is the simplest complete loop:

```text
initial SQL
  -> Planner
  -> Optimizer
  -> Executor
  -> Reporter
  -> done or retry SQL
```

`run_reporter_retry_loop()` keeps the Remote LLM session created for Commander
and reuses it for Reporter. This means Reporter sees the original SQL
constraints from the same conversation.

When Reporter returns `retry=true`, it must return:

```json
{
  "answer": null,
  "retry": true,
  "new_sql": {
    "sql": "SELECT ..."
  }
}
```

The runtime parses `new_sql.sql` into a fresh Logic DAG and starts the next
local round. If retry budget is exhausted, the case fails.

## v2 Logic DAG Wrapper

v2 does not invent a new per-node operation set. Each SQL statement is parsed as
a normal v1 Logic DAG. v2 wraps a batch of v1 DAGs:

```json
{
  "task_type": "table_reasoning_v2",
  "subtasks": [
    {
      "id": "answer_1",
      "index": 0,
      "question": "Is the person with the highest net worth self-made?",
      "answer": {"name": "answer_1", "type": "boolean"},
      "sql": "SELECT ... AS \"answer_1\" ...",
      "logic_dag": {
        "task_type": "table_reasoning_v1",
        "nodes": [],
        "edges": []
      }
    },
    {
      "id": "answer_2",
      "index": 1,
      "question": "What is the country of the person with the highest net worth?",
      "answer": {"name": "answer_2", "type": "string"},
      "sql": "SELECT ... AS \"answer_2\" ...",
      "logic_dag": {
        "task_type": "table_reasoning_v1",
        "nodes": [],
        "edges": []
      }
    }
  ]
}
```

The wrapper lets the Optimizer reason over multiple DAGs while keeping the
single-node execution semantics identical to v1.

## v2 Physical DAG Merge

The v2 Optimizer first produces one v1 physical plan per subtask. It then
aggregates those plans into one merged physical plan.

Node reuse is deliberately conservative. A node is reusable when:

- the operation is the same
- external input ids are the same
- mapped upstream dependency outputs are the same
- canonicalized params are the same
- the node is not `FormatAnswer`

The output name is ignored for equivalence, because equivalent nodes may have
different local names in different subplans. If the node is reused, the old
subplan output is remapped to the already-created merged output.

Example:

```text
answer_1 SQL:
Scan -> Sort(finalWorth DESC) -> Limit(1) -> Project(selfMade) -> FormatAnswer

answer_2 SQL:
Scan -> Sort(finalWorth DESC) -> Limit(1) -> Project(country) -> FormatAnswer
```

The shared prefix is reused:

```text
N0 Scan(table_1)                  -> T0
N1 Sort(T0, finalWorth DESC)      -> T1
N2 Limit(T1, 1)                   -> T2
N3 Project(T2, selfMade)          -> T3
N4 FormatAnswer(T3, answer_1)     -> answer_1
N5 Project(T2, country)           -> T4
N6 FormatAnswer(T4, answer_2)     -> answer_2
```

Merged edges:

```json
[
  {"from": "N0", "to": "N1"},
  {"from": "N1", "to": "N2"},
  {"from": "N2", "to": "N3"},
  {"from": "N3", "to": "N4"},
  {"from": "N2", "to": "N5"},
  {"from": "N5", "to": "N6"}
]
```

The merged physical plan also carries:

- `answers`: answer metadata for the batch
- `subtask_outputs`: mapping from subtask answer names to merged outputs
- `merge_stats`: subplan count, answer count, final node count, and reused node
  count

## v2 Runtime

v2 runtime pipeline:

```text
case specs
  -> TaskItem lifecycle records
  -> same-source Commander batch
  -> SQL list
  -> per-SQL v1 Planner
  -> grouped priority DAG queue
  -> local-batch DAG merge
  -> Executor
  -> Reporter review
  -> success or retry SQL
```

Batch sizes:

- `remote_batch_size`: maximum same-table questions sent to Commander.
- `local_batch_size`: maximum same-table DAGs merged into one Executor
  submission.

The runtime pops up to the batch size; it does not wait for a full batch.

Priority:

- DAG items are grouped by resolved source file.
- Lower retry count has higher priority.
- FIFO order is preserved within the same priority.

## v2 Reporter And Partial Retry

Reporter receives a compact local result report for the current answer batch.
It must decide per answer:

```json
{
  "answer": {
    "answer_1": true,
    "answer_2": null
  },
  "retry": true,
  "new_sql": {
    "answer_2": "SELECT \"country\" AS \"answer_2\" FROM \"table_1\" ORDER BY \"finalWorth\" DESC LIMIT 1;"
  }
}
```

Rules:

- Accepted answers are finalized immediately.
- Rejected answers must have `new_sql[answer_name]`.
- Retry SQL goes back to Local Planner.
- Retry budget is tracked per `TaskItem`.
- Retry exhaustion marks only that answer failed.

Execution failures use a SQL repair prompt. The report includes the original
question, answer name, answer type, current SQL, and local error. Reporter
returns corrected SQL keyed by failed answer names.

## Execution Failure Routing In v2

The Executor fails fast inside a merged physical plan. v2 maps the failure back
to affected answer keys:

1. Find the failing node output.
2. Follow downstream dependency closure.
3. Identify `FormatAnswer` outputs and `subtask_outputs` affected by that
   closure.

Then:

- affected answers go to SQL repair if retry budget remains
- unaffected answers with available outputs go to Reporter review
- independent answers that did not execute because of fail-fast scheduling are
  requeued without consuming retry budget

This keeps partial failure local to the answers that actually depend on the
failed node.

## Executor Semantics For Table Reasoning Nodes

v1 and v2 use the same node execution logic. There is no separate v2 NodeAgent.

For each physical node:

1. The Executor creates a node-local resource view.
2. `TableReasoningNodeAgent` checks Fast Path.
3. Fast Path calls static tools backed by pandas.
4. If a recoverable local Fast Path error occurs, Agent Loop may run.
5. The result is normalized and stored as a resource object.
6. Intermediates no longer needed downstream are released.

Most table reasoning nodes should hit Fast Path. Agent Loop exists for local
execution recovery, not cloud-side SQL repair.

## DAG Design Principles

The table reasoning DAGs follow these principles:

- Logic DAGs are backend-neutral and operation-oriented.
- Physical DAGs bind local resources and execution annotations.
- `dependency` references intermediate outputs, not node ids.
- `input` references external source ids, not intermediate outputs.
- `FormatAnswer` is the answer boundary and is not reused.
- v2 reuse happens at the physical-plan merge layer, not in Commander.
- Retry SQL is always lowered through Planner again.

## Current Limits

- v2 batching currently groups by one resolved source file.
- Multi-table and cross-source reuse are not optimized.
- Commander still controls SQL quality; NodeAgent does not fix bad SQL.
- v2 DAG reuse is limited to one merged local batch.
- Executor node execution is sequential inside one physical plan.
- Reporter context can grow across long table sessions.

## Deferred To V3

- True async pipeline buffers between Commander, Planner, Optimizer, Executor,
  and Reporter.
- Executor worker parallelism for independent ready nodes.
- Cross-batch DAG reuse and cache-aware scheduling.
- Better tail-batch policy for rare same-table cases.
- Session compaction for long table conversations.
- Multi-source v2 grouping and merge semantics.
- More advanced semantic DAG equivalence beyond exact op/input/dependency/params
  matching.
