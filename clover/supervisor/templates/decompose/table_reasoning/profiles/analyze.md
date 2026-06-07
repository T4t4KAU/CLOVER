Profile: analyze.

{% if task_dsl.get("hints") -%}
Hints: {{ task_dsl.get("hints") | tojson }}
{% endif -%}

Return exactly one JSON object. No ids/wrapper.

Choose one mutually exclusive action form:

Evidence plan:
{"acts":[{"op":"sql","q":"SELECT ..."},{"op":"inspect","q":"what to inspect","seed":"optional SELECT ..."},{"op":"analyze","kind":"statistical|correlation","seed":"SELECT ..."}]}

Terminal answer:
{"op":"answer","a":null|string|number|boolean|array|object}

Rules:
- `acts` may contain only `sql`, `inspect`, or `analyze` actions.
- Never put `{"op":"answer"}` inside `acts`.
- SQL actions retrieve evidence only; they do not encode final answers.
- Use sql for exact data. Use inspect for open evidence.
- Use analyze for deterministic numeric statistics or correlation over selected rows.
- Use terminal answer only when no table evidence is needed.
