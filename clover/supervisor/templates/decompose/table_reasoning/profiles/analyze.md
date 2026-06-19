Profile: analyze.

{% if task_dsl.get("hints") -%}
Hints: {{ task_dsl.get("hints") | tojson }}
{% endif -%}

Return exactly one JSON object. No ids/wrapper.

Choose one mutually exclusive action form:

Evidence plan:
{"acts":[{"op":"sql","q":"SELECT ..."},{"op":"analyze","kind":"statistical|correlation","seed":"SELECT ..."}]}

Terminal answer:
{"op":"answer","a":null|string|number|boolean|array|object}

Rules:
- `acts` may contain only `sql` or `analyze` actions.
- Never put `{"op":"answer"}` inside `acts`.
- SQL actions retrieve evidence only; they do not encode final answers.
- Prefer SQL for exact deterministic work: totals, averages, differences, percentages, ranks, and ties.
- Use analyze only when the requested statistic cannot be expressed directly in SQL.
- Exclude summary rows such as total, all, overall, or all ages unless explicitly requested.
- Parenthesize mixed AND/OR filters.
- Use terminal answer only when no table evidence is needed.
