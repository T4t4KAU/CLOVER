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
Use sql for exact data. Use inspect for open evidence. Use analyze for deterministic numeric statistics or correlation.
`ty` is answer type. If `ty` is number, `a` must be one number. If `ty` is string, `a` must be one concise string.
Use `ctx` only when `ev` is missing or failed.

Supervisor JSON:
