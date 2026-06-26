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
- Exclude summary rows such as total, totals, all, overall, subtotal, rank total,
  or all ages unless the question explicitly asks for the overall total.
- For `who`/`which` questions, project the requested entity column (person,
  team, country, title, event, location, etc.), not the column used only for
  filtering, ranking, or comparison.
- For "which X has the most/least ..." questions, filter out summary rows
  before ranking and return the X value, not the numeric max/min value.
- For yes/no questions, answer yes/no (or boolean only when the declared answer
  type is boolean). For "A or B" comparison questions, return the winning option
  text (A or B), not true/false.
- For "came first", "earlier", "last", "before", and "after", compare actual
  dates/row order, not raw strings.
- Parenthesize mixed AND/OR filters.
- For text predicates, prefer case-insensitive matching with `LOWER(...) LIKE`
  when the question wording may differ in case or spacing from cell text.
- Use terminal answer only when no table evidence is needed.
