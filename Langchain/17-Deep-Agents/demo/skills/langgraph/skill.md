---
name: langgraph
description: Use this skill whenever the user asks about building, debugging, or explaining LangGraph graphs, LangChain agents built on LangGraph, the `deepagents` library (create_deep_agent, backends, middleware, subagents), state schemas, nodes/edges, checkpointing, or streaming. Applies to work in this repo's LangChain/deepagents demos.
---

# LangGraph Skill

Gives the deep agent domain knowledge about LangGraph and the `deepagents`
library (used throughout this repo, see `demo/1-basicdeepagent.ipynb` and
`demo/2-backend.ipynb`) so it can build, extend, and debug graph-based
agents correctly instead of guessing at APIs.

## When to use this skill

- Building or modifying a LangGraph `StateGraph` (nodes, edges, conditional
  routing, state schema).
- Working with `deepagents.create_deep_agent`, its `tools`, `system_prompt`,
  `backend`, or `subagents` arguments.
- Choosing/configuring a backend: `StateBackend`, `FilesystemBackend`,
  `StoreBackend`.
- Debugging streaming, checkpointing, threads, or persistence issues.
- Explaining how a deep agent's planning/file-system/subagent tools work.

## How this skill is organized

- `instructions.md` — core concepts, this-repo conventions, and API usage
  rules.
- `examples.md` — worked examples mirroring patterns already used in this
  repo's notebooks.

Read `instructions.md` before writing or modifying any graph/agent code.
Check the actual installed version's API (e.g.
`.venv/Lib/site-packages/deepagents`) if unsure — the library evolves
quickly and instructions here may lag behind a newly upgraded version.
