# Filter repair hints
- Do not repeat exact text equality.
- Normalize both cell text and target text before comparing: use `.str.casefold()` and `.str.replace(r"[^a-z0-9]", "", regex=True)`.
- For substring match, use `str.contains(pattern, case=False, regex=False)`. Do NOT pass `casefold=` as a keyword argument.
- Return matching rows from the original DataFrame.
