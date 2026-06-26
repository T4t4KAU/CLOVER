Evidence payload:

{{OBSERVATION_PAYLOAD}}

Return exactly one JSON object.

Choose one mutually exclusive action form:

Terminal answer:
{"op":"answer","a":null|string|number|boolean|array|object}

Evidence SQL object:
`op` must be `"sql"` and `q` must be one complete SQLite SELECT using real
source ids and columns.

Evidence analyze object:
`op` must be `"analyze"`, `kind` must be `"statistical"` or `"correlation"`,
and `seed` must be one complete SQLite SELECT using real source ids and columns.

Rules: JSON only. Answer if `ev` supports the answer; else ask exactly one action.
Never return `acts`, `actions`, `sqls`, arrays, or multiple SQL statements.
Never output placeholders, angle-bracket text, or ellipsis characters; output
executable SQL only.
If several evidence checks are needed, fold them into one SELECT using joins,
CTEs, derived tables, aggregates, or `EXISTS`.
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
If evidence contains helper columns used only for ranking/sorting/filtering,
omit those helper values from the final answer unless the question explicitly
asks for them.
If `ty` is boolean, `a` must be true or false:
- If `ev` contains a boolean/0/1 answer column, copy that value directly.
- If SQL returned rows specifically checking all constraints in the statement,
  non-empty matching rows entail true and empty matching rows entail false.
- If rows or metrics show every required value/condition is present, answer true.
- If rows or metrics contradict any required value/condition, answer false.
- Do not invert supported evidence; e.g. evidence containing both requested
  values means true for a "both values are listed" statement.
Return final answers only: no explanations, full sentences, or extra values.
Never answer with a sentence such as "the answer is ..."; return only the
atomic value(s) requested by the question.
For approximate integer-count questions such as "approximately how many",
round the final numeric answer to the nearest integer.
For list answers, return a JSON array of atomic answer values. Do not collapse multiple rows into one sentence.
For multi-column answers, return an array of row arrays, for example `[["name", 3], ["other", 4]]`, not `"name, 3; other, 4"`.
If the question asks for a single entity plus a numeric difference/percentage,
return both requested values and omit unrequested helper values.
For date, numeric, and boolean values, preserve the table value exactly unless the question asks for a normalized form.
If the evidence contains the requested values exactly, copy those values instead of paraphrasing them.
When `repair` is present, treat it as the complete compact failure report:
- `goal` restates the original question and required answer type/projection.
- `fault` names the diagnosed failure class.
- `sql` is the failed SQL.
- `failure` identifies the first useful failure.
- `schema` contains only relevant columns.
- `join_candidates` and `join_paths`, when present, contain locally supported join edges and bridge-table paths.
- `samples` contains row-level examples that preserve relationships between relevant columns.
- `evidence` contains locally verified values and Edge repair results.
- `prior` contains compact previous attempts.
- `requirements` are mandatory.
Generate one materially corrected SQL action. Never repeat `repair.sql` or SQL in `repair.prior`.
Return the repair as exactly one SQL action object with an executable `q` SQL string.
For `join_semantic_error`, change the join path or join keys using `repair.join_candidates` and `repair.join_paths`; include an intermediate bridge table when the candidates connect the needed tables only through that table.
Use `repair.goal` to preserve the requested projection; most repairs should change filters, joins, grouping, ordering, or value normalization, not the selected answer column.
Use `repair.samples` and `repair.evidence.column_values` as ground truth for actual cell wording and row relationships.
Do not change columns based on a guessed candidate. A candidate column is usable only when `repair.evidence` shows the predicate literal in its `matches`, samples, or row-level `repair.samples`.
For `relative_row_semantic_error`, implement the previous/next relation; do not return the anchor row.
For `sql_execution_error`, replace unsupported syntax with SQLite-compatible SQL.
For `predicate_mismatch`, make the smallest predicate change supported by actual values: prefer LOWER/LIKE, punctuation/date normalization, or a literal seen in `column_values` before changing columns.
For temporal/range repairs, preserve arithmetic direction: `from A to B` means
value_at_B - value_at_A; `increase compared to previous` means current -
previous; `decrease compared to previous` means the most negative current -
previous change. For top/bottom-N aggregate repairs, select the N rows in a CTE
or derived table before aggregating them.
For ranking repairs, preserve ties unless exactly one row is requested. For
comparison repairs between two named entities, compute each entity's scalar
value separately before subtracting/comparing.
Preserve the requested answer projection and answer type.
{% if force_final_answer %}
No more execution rounds are available. Return an answer object using the current evidence. If it is insufficient, return `{"op":"answer","a":null}`. Do not return an action.
{% endif %}

Supervisor JSON:
