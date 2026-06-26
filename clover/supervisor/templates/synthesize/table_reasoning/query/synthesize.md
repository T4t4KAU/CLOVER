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
For WikiTableQuestions-style string answers:
- yes/no questions must answer "yes" or "no", not true/false, unless `ty` is
  explicitly boolean.
- "A or B" comparison questions must answer the selected option text, not a
  boolean.
- If SQL returns an object with the exact requested values (for example school
  and year), return a flat array of those values rather than a keyed object.
For `who`/`which` questions, answer with the requested entity value, not the
numeric/date evidence used to choose it.
If `ty` is boolean, `a` must be true or false:
- If `ev` contains a boolean/0/1 answer column, copy that value directly.
- If SQL returned rows specifically checking all constraints in the statement,
  non-empty matching rows entail true and empty matching rows entail false.
- If rows or metrics show every required value/condition is present, answer true.
- If rows or metrics contradict any required value/condition, answer false.
- Do not invert supported evidence; e.g. evidence containing both requested
  values means true for a "both values are listed" statement.
Return final answers only: no explanations, full sentences, or extra values.
When `repair` is present, treat it as the complete compact failure report:
- `goal` restates the original question and required answer type/projection.
- `fault` names the diagnosed failure class.
- `sql` is the failed SQL.
- `failure` identifies the first useful failure.
- `schema` contains only relevant columns.
- `samples` contains row-level examples that preserve relationships between relevant columns.
- `evidence` contains locally verified values and Edge repair results.
- `prior` contains compact previous attempts.
- `requirements` are mandatory.
Generate one materially corrected SQL action. Never repeat `repair.sql` or SQL in `repair.prior`.
Use `repair.goal` to preserve the requested projection; most repairs should change filters, joins, grouping, ordering, or value normalization, not the selected answer column.
Use `repair.samples` and `repair.evidence.column_values` as ground truth for actual cell wording and row relationships.
Do not change columns based on a guessed candidate. A candidate column is usable only when `repair.evidence` shows the predicate literal in its `matches`, samples, or row-level `repair.samples`.
For `relative_row_semantic_error`, implement the previous/next relation; do not return the anchor row.
For `sql_execution_error`, replace unsupported syntax with SQLite-compatible SQL.
For `predicate_mismatch`, make the smallest predicate change supported by actual values: prefer LOWER/LIKE, punctuation/date normalization, or a literal seen in `column_values` before changing columns.
Preserve the requested answer projection and answer type.
{% if force_final_answer %}
No more execution rounds are available. Return `{"op":"answer","a":...}` using the current evidence. If it is insufficient, return `{"op":"answer","a":null}`. Do not return `acts`.
{% endif %}

Supervisor JSON:
