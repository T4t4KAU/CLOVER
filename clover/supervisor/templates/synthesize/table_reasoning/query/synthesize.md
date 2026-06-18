Evidence payload:

{{OBSERVATION_PAYLOAD}}

Return exactly one JSON object.

Choose one mutually exclusive action form:

Terminal answer:
{"op":"answer","a":null|string|number|boolean|array|object}

Evidence plan:
{"acts":[{"op":"sql","q":"SELECT ..."},{"op":"inspect","q":"what to inspect","seed":"optional SELECT ..."},{"op":"analyze","kind":"statistical|correlation","seed":"SELECT ..."}]}

Rules: JSON only. Answer if `ev` supports the answer; else ask a small action group.
`acts` may contain only `sql`, `inspect`, or `analyze`; never put `{"op":"answer"}` inside `acts`.
Use sql for exact data. Use inspect for open evidence. Use analyze only when SQL cannot express the statistic.
`ty` is answer type. If `ty` is number, `a` must be one number. If `ty` is string, `a` must be one concise string.
Return final answers only: no explanations, full sentences, or extra values.
Use `ctx` only when `ev` is missing or failed.
When an action's `ev` contains `column_values`, those are the actual top values in the WHERE-equality columns (collected locally because the SQL returned 0 rows). Compare them against the SQL literals to spot value-format mismatches (case, spacing, suffixes), then either answer directly or rewrite the SQL with a relaxed match (LIKE, casefold, substring).

Supervisor JSON:
