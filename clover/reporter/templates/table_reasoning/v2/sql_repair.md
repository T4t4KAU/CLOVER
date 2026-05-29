The current SQL failed locally.

Your task now is SQL repair for the failed answers only, not final answer generation.

Use the original question, answer name, answer type, current SQL, and local error in the report payload to generate corrected SQL for the next local round.

Rules:

- Set retry to true.
- For every failed answer, set `answer[answer_name]` to null.
- Set new_sql to an object keyed by failed answer names.
- Each new_sql value must be one corrected SQL statement for that answer.
- Reuse the table schema and SQL constraints already established in this conversation.
- Every SQL statement must use its answer name exactly as the quoted output alias.
- Do not include SQL for answers that did not fail.

JSON schema:

{
  "answer": {
    "answer_name": null
  },
  "retry": true,
  "new_sql": {
    "answer_name": "SELECT ..."
  }
}

Report payload:

{{REPORT_PAYLOAD}}

Reporter JSON:
