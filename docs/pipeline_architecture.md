# Pipeline Architecture

## Scope

The current pipeline document describes the latest internal table reasoning
runtime. Externally, table reasoning is exposed as `table_reasoning`; v1 and v2
are implementation modes. The design is table-reasoning-first, but the queue,
profiling, and case result primitives live in a generic runtime layer so other
task families can reuse the pattern later.

Current implementation:

- Generic pipeline helpers: `clover/runtime/pipeline.py`
- Table reasoning v2 runtime: `clover/runtime/table_reasoning/v2.py`
- v1 retry runtime: `clover/runtime/table_reasoning/v1.py`
- DataBench eval integration: `benchmarks/databench/eval.py`

## Main Objects

`TableReasoningCaseSpec` is the external case input. It carries the original
task DSL, base directory, optional preprocess result, and metadata.

`TaskItem` is the internal lifecycle record for one question. It owns:

- `case_id`
- globally unique `answer_key`, currently `answer_1`, `answer_2`, ...
- question text and answer type
- source table identity
- local and remote DSLs
- current SQL
- retry count
- status
- callback for streaming final case results to eval

`LogicDagItem` binds one TaskItem, one SQL statement, and its parsed Logic DAG.

`TableSessionState` owns one Remote LLM conversation for one source file.
Commander and Reporter reuse that same conversation.

## Lifecycle States

Current task statuses:

- `pending_remote`
- `sql_ready`
- `dag_ready`
- `executing`
- `reporter_review`
- `sql_repair`
- `retrying`
- `success`
- `failed`

The lifecycle is answer-key based. One case produces one answer in the current
DataBench path.

## Queue Topology

The v2 runtime is synchronous but structured as a pipeline:

```text
case specs
  -> pending_remote
  -> Commander batch
  -> sql_items
  -> Local Planner
  -> grouped DAG priority queue
  -> Optimizer merge
  -> Executor
  -> Reporter review or SQL repair
  -> success / retry / failed
```

`GroupedPriorityQueue` groups DAG items by source file and orders work by retry
count first. This means lower-retry answers are handled before answers that have
already consumed more retry budget.

## Batching Semantics

There are two system batch sizes:

- `remote_batch_size`: maximum questions sent to Remote Commander in one call
  for the same source file.
- `local_batch_size`: maximum same-table Logic DAGs merged into one physical
  plan and sent to the Executor.

Local Planner does not need an LLM and has no model batch limit. Optimizer
passes that do not merge DAGs also do not need a model batch limit. The merge
step uses `local_batch_size` because one merged plan becomes one Executor
submission.

The runtime pops up to the configured batch size. It does not wait for a full
batch.

The eval layer can run multiple v2 system instances in parallel, one per
source-file group. Internal node workers inside one Executor are intentionally
not part of v2.

## Remote Session Reuse

The runtime creates at most one Remote LLM session per source file in a system
instance.

First Commander batch:

- Sends schema and SQL constraints.
- Marks `schema_sent = True`.

Later Commander batches for the same source file:

- Reuse the same session.
- Do not resend the schema.
- Ask for SQL for the new questions only.

Reporter:

- Reuses the same table session.
- Sends the Reporter instruction only once per table session.
- On normal review, asks for final answers or new SQL keyed by insufficient
  answer names.
- On local execution failure, asks for SQL repair for affected answers.

## Local Planning And DAG Merge

Remote Commander returns SQL only. Local Planner parses each SQL statement into
a v1 Logic DAG. The v2 runtime wraps these per-question DAGs into a v2 Logic DAG
with `subtasks`.

Optimizer lowers each subtask to a v1 physical plan, then merges the plans.

Current conservative node reuse key:

```text
operation
external input ids
mapped dependency outputs
canonical params
```

The output name is not part of the equivalence key. If two nodes have the same
operation, the same source inputs, equivalent upstream results, and identical
parameters, they are treated as equivalent and computed once.

`FormatAnswer` nodes are not reused because each answer key must remain
separate. Non-answer intermediate outputs are renamed to `T0`, `T1`, ... in the
merged physical plan. Final answer outputs use their global answer keys.

## Execution And Partial Failure Routing

The Executor currently fails fast inside one physical plan. When execution
fails, v2 computes affected answer keys by following the dependency closure from
the failing output.

- Affected answers go to SQL repair if retry budget remains.
- Unaffected answers with available outputs still go to Reporter review.
- Independent answers that were not executed because of fail-fast scheduling are
  requeued without consuming retry budget.

Reporter can partially accept a batch:

- Accepted answers are finalized immediately.
- Rejected answers receive `new_sql[answer_key]` and re-enter the local Planner.
- Retry exhaustion finalizes the answer as failed.

Execution errors and Reporter-rejected answers both become new SQL processing
when retry budget remains.

## Eval Integration

`benchmarks/databench/eval.py` converts selected DataBench cases into
`TableReasoningCaseSpec` objects. v2 streams `CaseResult` objects through a
callback, so eval can update progress and per-case artifacts as soon as answers
finish.

`benchmarks/run_databench_eval.sh` runs the latest table reasoning pipeline and exposes
the model config paths plus batch sizes through arguments and environment
variables.

## Profiling

`PipelineProfiler` records stage timing and counters:

- Commander
- Planner
- Optimizer
- Executor
- Reporter review
- Reporter SQL repair

The runtime also records merge counters, reused node count, remote call count,
and optional baseline timing when baseline profiling is enabled.

## Current Limits

- The v2 pipeline is synchronous inside one source-file group.
- There are no internal module workers or asynchronous buffers.
- Context growth in reused Remote LLM conversations is not compacted.
- Same-table grouping currently means same resolved source file.
- Multi-source table reasoning and cross-table grouping are not optimized.
- Local execution is still one ready node at a time inside the Executor.

## Deferred To V3

- True asynchronous pipeline stages with bounded buffers between modules.
- Separate concurrency and rate limits for Remote Commander, Local SLM,
  Optimizer, Executor, and Reporter.
- Executor worker parallelism for independent ready nodes.
- Resource-aware scheduling and memory-pressure-aware batching.
- Session context compaction or session rollover for long table conversations.
- More advanced DAG reuse across batches, not only inside one merged batch.
- Better tail and small-group batching policies.
- Multi-source and cross-table queue grouping.
- More detailed tracing for partial failures, retries, and per-stage latency.
