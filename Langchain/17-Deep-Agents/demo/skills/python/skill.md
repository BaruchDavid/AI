---
name: python
description: Use this skill whenever the user asks to write, debug, refactor, review, test, or explain Python code, or asks about Python language features, standard library, packaging (uv/pip), virtual environments, or performance. Covers general-purpose Python 3.10+ development.
---

# Python Skill

Gives the deep agent a consistent, opinionated approach to Python work: clean,
idiomatic, type-hinted code; sane project/dependency conventions (this repo
uses `uv`); and a habit of validating code (run it / run tests) before
declaring a task done.

## When to use this skill

- Writing new Python scripts, modules, CLIs, or small services.
- Debugging a traceback or unexpected behavior in existing Python code.
- Refactoring or reviewing Python code for clarity, correctness, or
  performance.
- Explaining a Python language feature, stdlib module, or error message.
- Setting up or modifying dependencies (`uv add`, `pyproject.toml`).

## How this skill is organized

- `instructions.md` — coding conventions, tooling rules, and the workflow to
  follow (write → run → fix) before answering.
- `examples.md` — worked examples showing the expected style and workflow.

Read `instructions.md` before writing any non-trivial Python code, and check
`examples.md` when unsure how much detail or which pattern to use.
