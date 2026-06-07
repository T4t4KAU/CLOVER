Evidence payload:

{{OBSERVATION_PAYLOAD}}

Return one JSON action:
{"op":"answer","a":{"answer_name":null|string|number|boolean|array|object}}

Rules:
- Copy each supported value from `ev` into `a[answer_name]`.
- Use null only when the observation does not support that answer.
- No markdown or extra text.

Supervisor JSON:
