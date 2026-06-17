# Filter repair hints
- Do not repeat exact text equality.
- Normalize both cell text and target text before comparing.
- Use casefold and strip non-alphanumeric chars for fuzzy match.
- Return matching rows from the original DataFrame.
