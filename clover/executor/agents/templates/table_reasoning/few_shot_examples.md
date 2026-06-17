# Few-shot hints

When the file payload includes a `few_shot_hint` string, it shows a one-line pattern for the current operation. Follow that pattern.

## Common patterns

- Text-to-number: `"$1.2M"` → `1_200_000`. Strip `$`, parse number, multiply by suffix (K=1e3, M=1e6, B=1e9).
- Fuzzy text match: normalize both sides (`casefold`, keep alnum only), then `str.contains`.
- Single value answer: `pd.DataFrame({"answer": [value]})`.
