Evidence payload:

{{OBSERVATION_PAYLOAD}}

Return exactly one JSON object.

Choose one mutually exclusive action form:

Terminal answer:
{"op":"answer","a":null|string|number|boolean|array|object}

Evidence plan:
{"acts":[{"op":"sql","q":"SELECT ..."},{"op":"analyze","kind":"statistical|correlation","seed":"SELECT ..."}]}

Rules: JSON only. Answer if `ev` supports the answer; else ask a small action group.
`acts` may contain only `sql` or `analyze`; never put `{"op":"answer"}` inside `acts`.
Use sql for exact data. Use analyze only when SQL cannot express the statistic.
`ty` is answer type. If `ty` is number, `a` must be one number. If `ty` is string, `a` must be one concise string.
Return final answers only: no explanations, full sentences, or extra values.
When `repair` is present, treat it as the complete compact failure report:
- `fault` names the diagnosed failure class.
- `sql` is the failed SQL.
- `failure` identifies the first useful failure.
- `schema` contains only relevant columns.
- `join_candidates` and `join_paths`, when present, contain locally supported join edges and bridge-table paths.
- `evidence` contains locally verified values and Edge repair results.
- `prior` contains compact previous attempts.
- `requirements` are mandatory.
Generate one materially corrected SQL action. Never repeat `repair.sql` or SQL in `repair.prior`.
Do not change columns based on a guessed candidate. A candidate column is usable only when `repair.evidence` shows the predicate literal in its `matches` or samples.
For `join_semantic_error`, change the join path or join keys using `repair.join_candidates` and `repair.join_paths`; include an intermediate bridge table when the candidates connect the needed tables only through that table.
For `relative_row_semantic_error`, implement the previous/next relation; do not return the anchor row.
For `sql_execution_error`, replace unsupported syntax with SQLite-compatible SQL.
For `predicate_mismatch`, make the smallest predicate change supported by actual values.
Preserve the requested answer projection and answer type.
{% if force_final_answer %}
No more execution rounds are available. Return `{"op":"answer","a":...}` using the current evidence. If it is insufficient, return `{"op":"answer","a":null}`. Do not return `acts`.
{% endif %}

Supervisor JSON:
