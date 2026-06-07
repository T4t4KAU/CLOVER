Return only JSON: {"s":"def solve(...):\n    ...\n    return result"}.
"s" must be exactly one top-level def solve function.
No code outside solve. No markdown. No prose.
Use only function arguments plus pd, np, helpers, print.
Repair with the smallest change.
For text matching, normalize both cell text and target text:
def norm_text(x): return ''.join(ch for ch in str(x).casefold() if ch.isalnum())
Use map(norm_text) on Series; return matching rows from the original DataFrame.

Case:
{{ CASE_JSON }}
