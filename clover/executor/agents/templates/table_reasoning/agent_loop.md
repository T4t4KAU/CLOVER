Complete only the current local table operation.

Strictly implement `task.operation` and `params`; do not infer or rewrite any broader task.

Follow `task.operation_note` when present.

Write Python code that computes the required output and assigns it to `result`.

Table handles expose `.frame` for pandas operations and metadata such as `.group_keys`.

If `tool.available` is true, prefer `result = tool.run()` for the exact current operation.

Prefer direct completion code. Inspect only if needed to repair a failed attempt.

Return exactly one JSON action and no extra text.

Actions:

{"action":"run_python","code":"..."}

{"action":"submit_result","name":"result"}

{"action":"abort","reason":"..."}

Iteration: {{ iteration }}

Context:

{{ SANDBOX_VIEW }}
