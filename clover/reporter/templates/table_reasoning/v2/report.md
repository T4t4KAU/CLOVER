Review the following local result report in the existing CLOVER conversation.

Multiple answers may be reviewed together.

Decision rules:

- If an answer is sufficient, copy its final value into `answer[answer_name]`.
- If an answer is insufficient, set `answer[answer_name]` to null.
- If all answers are sufficient, set retry to false and new_sql to null.
- If any answer is insufficient, set retry to true.
- When retry is true, new_sql must be an object keyed only by insufficient answer names.
- Each new_sql value must be one corrected SQL statement for that answer.
- Reuse the table schema and SQL constraints already established in this conversation.
- Every SQL statement must use its answer name exactly as the quoted output alias.
- Do not request retry for answers that are already sufficient.

JSON schema:

{
  "answer": {
    "answer_1": null | string | number | boolean | array | object,
    "answer_2": null | string | number | boolean | array | object
  },
  "retry": boolean,
  "new_sql": null | {
    "answer_name": "SELECT ..."
  }
}

Report payload:

{{REPORT_PAYLOAD}}

Reporter JSON:
