Evidence payload:

{{OBSERVATION_PAYLOAD}}

Return one JSON action:
{"op":"answer","a":{"answer_name":null|string|number|boolean|array|object}}

Rules:
- Copy each supported value from `ev` into `a[answer_name]`.
- Values must be final answers only: no explanations, units unless requested, or full sentences.
- Preserve every requested field/list item. Return list answers as JSON arrays; do not join lists into one prose string.
- For multi-column answers, return an array of row arrays, for example `[["name", 3], ["other", 4]]`.
- If `ty` is number, return one number. If `ty` is string, return one concise string.
- If the evidence contains exact date, numeric, boolean, or string values, copy them instead of paraphrasing.
- Use null only when the observation does not support that answer.
- No markdown or extra text.

Supervisor JSON:
