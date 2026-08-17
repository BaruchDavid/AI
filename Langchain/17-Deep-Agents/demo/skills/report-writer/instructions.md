# Report Writer — Instructions

## Trigger rule

After producing the final answer for a user query, decide whether the turn
warrants a report using the criteria in `skill.md`. If yes, write the report
as the last step of the turn — after the answer content is finalized, not
before (the report summarizes what was actually done, so it must be written
last).

## Where to save it

Use the deep agent's filesystem tool (`write_file` / equivalent) to save
under a `/reports/` directory, regardless of which backend is active
(`StateBackend`, `FilesystemBackend`, or `StoreBackend` — the report is just
another file written through the same tool as any other agent output).

**File name:** `/reports/<UTC-timestamp>-<short-slug>.md`

- `<UTC-timestamp>` — `YYYY-MM-DDTHHMM` (minute precision is enough).
- `<short-slug>` — 3-6 words from the query, lowercase, hyphenated (e.g.
  `latest-ai-news`, `fix-csv-average-bug`).

Example: `/reports/2026-08-17T1420-fix-csv-average-bug.md`

## Report structure (template)

```markdown
# Report: <one-line title derived from the query>

**Date:** <ISO date/time>
**Query:** <the user's original request, verbatim or lightly trimmed>

## Approach

<2-5 sentences: what steps were taken — tools used, files read/written,
searches run, subagents delegated to. Enough for someone to understand
*how* the answer was produced, not a full transcript.>

## Result

<The substance of the final answer — the actual finding, code change
summary, or conclusion. This is the part someone would actually want to
re-read later.>

## Files touched

<Bullet list of files created/modified/read, with paths. Omit this section
if none.>

## Sources

<Bullet list of URLs/references consulted, if any web search or external
lookup was involved. Omit this section if none.>

## Open questions / follow-ups

<Anything left unresolved, assumptions made, or suggested next steps.
Omit this section if there are none — don't invent follow-ups just to fill
it in.>
```

Omit any section that has nothing to say — don't pad the template. Keep the
whole report short: a few short paragraphs plus lists, not a full essay.

## What NOT to do

- Don't write a report for every single tool call inside a turn — one
  report per user-facing answer, at most.
- Don't duplicate the full answer verbatim if it was already long (e.g. a
  full code diff) — summarize it in "Result" and point to the file(s)
  instead of repeating hundreds of lines.
- Don't fabricate sources or files that weren't actually used.
- Mentioning that a report was saved to the user is optional and should be
  brief (one line, with the path) — it should never replace or overshadow
  the actual answer to their query.
