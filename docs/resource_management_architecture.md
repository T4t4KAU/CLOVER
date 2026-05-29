# Resource Management Architecture

## Scope

Resource management gives the Executor one unified lifecycle model for external
files, in-memory intermediate data, and executor-owned spilled files. Physical
nodes should access data through a resource view instead of directly owning
global tables or intermediate values.

Current implementation:

- Resource objects: `clover/executor/resources/objects.py`
- Resource store and node views: `clover/executor/resources/store.py`
- Execution context integration: `clover/executor/context.py`
- Sandbox handles: `clover/executor/handles`

## Resource Types

`ResourceObject` is the common base. It exposes:

- `materialize(target=...)`
- `summary()`
- `estimate_size()`
- `pin()` and `unpin()`
- `release_materialized()`
- `close()`

Current concrete resources:

- `FileExternalResourceObject`: source data owned outside the executor. Closing
  releases cached materialized values but never deletes the original file.
- `MemoryResourceObject`: executor-owned output whose value is currently in
  memory.
- `FileSpilledResourceObject`: executor-owned output stored in a temporary spill
  file under `/tmp`.

## Store Lifecycle

Each `Executor.execute()` call creates one `ResourceStore`.

1. External resources from the physical plan are registered as source resources.
2. A ready node receives a `NodeResourceView` for exactly its dependencies and
   source inputs.
3. The view pins those resources during node execution.
4. The node output is written with `ResourceStore.put_output()`.
5. Consumed dependency outputs are released when no downstream node still needs
   them, unless they are retained final outputs.
6. At the end of execution, `close_all()` releases all resources and removes the
   run spill directory.

This keeps intermediate data lifetime tied to DAG dependency usage instead of
the whole run.

## Spill Policy

`ResourceLimits` controls capacity:

- `memory_budget_bytes`: approximate memory budget for executor-owned outputs.
- `spill_threshold_bytes`: values larger than this are written directly to
  spill storage.
- `spill_root`: defaults to `/tmp/clover_spill`.

Spilled resources are written under:

```text
/tmp/clover_spill/<run_id>/
  resources/<resource_id>.pkl
  metadata/<resource_id>.json
  manifest.jsonl
```

When the memory budget would be exceeded, the store spills the least recently
accessed unpinned memory output. Pinned resources are never chosen as spill
candidates during the active node execution.

Spill files are executor-owned and deleted by `FileSpilledResourceObject.close()`
or by `ResourceStore.close_all()`.

## Unified Access Interface

The rest of the executor should not need to know whether a value lives in
memory, an external file, or a spill file.

Important access methods:

- `ResourceStore.node_view(...)`: returns a node-local resource view.
- `NodeResourceView.materialize_dependencies(target="pandas")`
- `NodeResourceView.materialize_sources(target="resource_spec")`
- `NodeResourceView.project_dependencies(projector)`
- `NodeResourceView.project_sources(projector)`
- `ResourceObject.materialize(target="python" | "pandas" | "resource_spec")`

Fast Path uses the same resource view as Agent Loop. It receives source specs
and materialized dependency values derived from the view, so storage placement
is transparent to the static tools.

## Sandbox Projection

Sandbox projection is generic and lives above task-specific agents:

- `SandboxProjector` converts a `ResourceObject` into a sandbox-visible handle.
- Table resources become `TableHandle`.
- Scalar or JSON-like values become `ValueHandle`.

`TableHandle` keeps a copied dataframe together with table metadata such as
`group_keys`. This avoids the lossy `PandasTable -> DataFrame` conversion path
that would drop grouping metadata needed by later table reasoning operations.

The Agent Loop sees projected handles, not the `ResourceStore`.

## Summaries

Resources cache compact summaries for tracing and reporting:

- table row count, columns, preview rows, and optional group keys
- list length and preview
- scalar preview

Summaries should be used for diagnostics and LLM prompts. Full outputs should
not be serialized into Reporter prompts by default.

## Current Limits

- Size estimation is approximate for non-table Python objects.
- Spill recovery is run-local only; CLOVER does not reload a previous run's
  spill manifest.
- File spill uses pickle, which is convenient but not a stable cross-version
  storage format.
- The current Executor is single-worker inside one physical plan.
- The sandbox boundary is not an OS security boundary.

## Deferred To V3

- Resource-aware node scheduling and true executor worker parallelism.
- Stronger spill codecs and schema-aware persisted formats.
- Cross-batch or cross-run cache reuse.
- Better memory accounting for cached external resources.
- Context compaction for large resource summaries.
- Optional process-level sandbox projection for stronger isolation.
