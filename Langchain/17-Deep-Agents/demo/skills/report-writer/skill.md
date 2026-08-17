---
name: report-writer
description: Use this skill after finishing any substantive answer to a user query — once the deep agent has completed research, analysis, code changes, or a multi-step task — to write a short structured Markdown report summarizing what was asked, what was done, and the outcome. Not needed for trivial one-line answers or simple clarifying exchanges.
---

# Report Writer Skill

Standardizes what happens *after* the deep agent finishes answering: it
writes a concise Markdown report capturing the query, the approach taken,
the result, and any sources/files touched, and saves it via the agent's
filesystem tools so there's a durable, reviewable record of the work.

## When to use this skill

Apply this skill at the end of a turn whenever the deep agent's answer
involved real work, for example:

- Multi-step research (e.g. web searches, comparing sources).
- Writing, modifying, or debugging code.
- Any task that used the planning tool or delegated to a subagent.
- Any task whose result the user would plausibly want to revisit later.

Skip it for trivial exchanges: a one-line factual answer, a clarifying
question, a greeting, or "yes/no" confirmations — writing a report for
these adds noise, not value.

## How this skill is organized

- `instructions.md` — report structure, naming/location convention, and
  the trigger rule in detail.
- `examples.md` — full worked example reports.

Read `instructions.md` to see the exact template and file-naming
convention before writing a report.
