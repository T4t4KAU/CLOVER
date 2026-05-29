Review the following current local result report in the existing CLOVER conversation.

Report payload:

{{REPORT_PAYLOAD}}

Return exactly this JSON shape:

{
  "answer": null | string | number | boolean | array | object,
  "retry": boolean,
  "new_sql": null | {
    "sql": "SELECT ..."
  }
}

If retry is true, set new_sql to an object with exactly this shape:

{
  "sql": "SELECT ..."
}

When writing new_sql.sql, reuse the SQL generation constraints already given to Commander earlier in this conversation.

Reporter JSON:
