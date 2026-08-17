# Report Writer — Examples

## Example 1: Research query

User query: "What is the latest news about AI?"

After answering (using `web_search`), write to
`/reports/2026-08-17T0930-latest-ai-news.md`:

```markdown
# Report: Latest AI news summary

**Date:** 2026-08-17T09:30Z
**Query:** What is the latest news about AI?

## Approach

Ran a web search via the `web_search` tool (Tavily) for recent AI news,
reviewed the top 5 results, and summarized the common themes for the user.

## Result

Summarized 3 headline stories: [topic A], [topic B], [topic C], with a
one-sentence takeaway on why each matters.

## Sources

- https://example.com/article-a
- https://example.com/article-b
- https://example.com/article-c
```

## Example 2: Code fix query

User query: "Fix the bug where `average_column` crashes on an empty CSV."

After fixing and verifying, write to
`/reports/2026-08-17T1105-fix-average-column-empty-csv.md`:

```markdown
# Report: Fix average_column crash on empty CSV

**Date:** 2026-08-17T11:05Z
**Query:** Fix the bug where average_column crashes on an empty CSV.

## Approach

Reproduced the crash (`ZeroDivisionError`) by running the function against
an empty CSV. Added a guard that raises a clear `ValueError` instead, and
re-ran the function against both an empty and a populated CSV to confirm.

## Result

`average_column` now raises `ValueError("No rows found in <path>")` for an
empty file instead of crashing with an unhandled `ZeroDivisionError`, and
still returns the correct mean for non-empty files.

## Files touched

- `mod.py` — added the empty-input guard in `average_column`.

## Open questions / follow-ups

None — behavior confirmed with both the empty-file and normal-file cases.
```

## Example 3: When to skip

User query: "What's 12 * 7?"

No report is written — this is a trivial one-line answer with no research,
files, or multi-step process behind it.
