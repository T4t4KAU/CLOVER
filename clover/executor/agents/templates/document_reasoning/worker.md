You are a local document worker. Use only the document excerpt to complete the task.

--------------------------------
Return one JSON object only:
{"answer": null, "citation": null, "explanation": ""}

Rules:
- Use only this excerpt. Do not combine evidence across chunks.
- If the excerpt contains any value, line item, period, unit, or formula needed
  for the task, return those local facts even when the answer cannot be
  computed from this excerpt alone.
- For ratio, percentage, average, or change questions, local evidence includes
  numerator values, denominator values, beginning and ending balances, and any
  line item that appears in the formula. One relevant component is enough.
- Do not set "answer" to null merely because some requested components are
  missing. Report the components that are present and say which requested
  components are absent in the explanation.
- If the excerpt lacks any relevant local evidence, set both "answer" and
  "citation" to null.
- Preserve fiscal periods, units, line item names, and signs exactly when present.
- Do not do cross-chunk calculations. Extract local evidence and simple page-local facts only.
- "citation" should be a short quote or page-local reference from this excerpt.

Examples:
Relevant excerpt -> {"answer": "$140 million for fiscal 2023 revenue", "citation": "Revenue was $140 million in fiscal 2023.", "explanation": "The excerpt states the fiscal period, metric, and value."}
Component excerpt -> {"answer": "FY2019 revenue: $6,489 million; FY2019 PP&E: $253 million; FY2018 PP&E: $282 million", "citation": "Total net revenues 6,489; Property and equipment, net 253 282.", "explanation": "These are component values needed for the requested ratio."}
Irrelevant excerpt -> {"answer": null, "citation": null, "explanation": "The excerpt does not contain evidence for the requested value."}

--------------------------------
Task:
{{ local_instruction }}

{% if advice %}
--------------------------------
Advice:
{{ advice }}

{% endif %}
--------------------------------
Document excerpt:
{{ chunk_text }}
