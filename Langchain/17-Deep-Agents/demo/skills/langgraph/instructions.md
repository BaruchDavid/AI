# LangGraph / deepagents — Instructions

## Core concepts

- **StateGraph**: a graph of nodes (functions that take/return partial
  state) connected by edges (fixed) or conditional edges (routing
  functions). State is a typed dict/`TypedDict` or Pydantic model merged
  across node outputs via reducers (e.g. `add_messages` for message lists).
- **Deep agents** (`deepagents.create_deep_agent`) wrap a normal LangChain
  agent with extra built-in capabilities: a planning tool, a virtual
  filesystem (read/write/edit/ls tools), and the ability to delegate to
  subagents for context isolation on complex sub-tasks. Use
  `create_deep_agent` instead of the plain `langchain.agents.create_agent`
  whenever the task needs multi-step planning, large context handled via
  files, or delegation — use the plain agent for simple single-shot
  tool-calling tasks.
- **Backends** control where the deep agent's virtual filesystem actually
  lives:
  - `StateBackend()` — files live in LangGraph state, in RAM, scoped to the
    current run/thread. Nothing persists across threads.
  - `FilesystemBackend(root_dir=..., virtual_mode=True)` — files are
    written to real disk under `root_dir`. Use for demos/scripts where you
    want files to survive and be inspectable on disk.
  - `StoreBackend(namespace=..., store_name=...)` — files live in a
    LangGraph `Store` (e.g. `InMemoryStore` or a persistent store),
    addressable by namespace, so they can be read back from a **different**
    thread. Use when multiple conversations/threads need to share files, or
    when you need cross-thread persistence without hitting real disk.
  - Choose the backend based on the persistence requirement the user
    states — don't default to `FilesystemBackend` if they only need
    per-run scratch state (`StateBackend` is simpler and has no I/O side
    effects).

## API usage rules

- Import the deep agent constructor from `deepagents`:
  `from deepagents import create_deep_agent`.
- Import backends from `deepagents.backends`:
  `from deepagents.backends import StateBackend, FilesystemBackend, StoreBackend`.
- `create_deep_agent(model=..., tools=[...], system_prompt=..., backend=...)`
  — `model` can be a string like `"groq:openai/gpt-oss-20b"` (resolved via
  `init_chat_model` under the hood) or an already-initialized chat model
  object.
- Tools passed to `create_deep_agent` are plain Python functions with type
  hints and a docstring — LangChain infers the tool schema from these, so
  keep the docstring accurate and the signature fully typed.
- To invoke: `agent.invoke({"messages": [{"role": "user", "content": ...}]})`
  and read `result["messages"][-1]["content"]` for the final answer, or
  iterate `result["messages"]` to inspect the full trace.
- For thread-scoped persistence (`StoreBackend`, or any checkpointed graph),
  pass `config={"configurable": {"thread_id": <uuid>}}` on `invoke`. A new
  `thread_id` means a fresh conversation/session.
- Before assuming an API shape (argument names, return structure), check the
  installed package source rather than relying on memory — the library
  moves fast and this repo's `.venv` is the source of truth for the exact
  installed version.

## This repo's conventions

- Environment variables (`OPENAI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`)
  are loaded via `python-dotenv`'s `load_dotenv()` at the top of each
  notebook — don't hardcode keys.
- The demo notebooks use Tavily (`tavily-python`) as the web search tool,
  wrapped in a `web_search(query, max_results, topic, include_raw_content)`
  function passed into `tools=[...]`.
- New experiments should generally start as a notebook cell in
  `demo/*.ipynb` following the existing pattern (imports → tools →
  agent construction → invoke → print results), unless the user asks for a
  standalone `.py` module.

## Debugging tips

- If a node isn't firing, check the conditional edge routing function's
  return value matches one of the declared edge keys exactly.
- If state isn't merging as expected, check the reducer on that state key
  (default is "last write wins" unless annotated with a reducer like
  `add_messages`).
- If `StateBackend` files "disappear," that's expected — they're
  thread/run-scoped RAM only, not a bug.
