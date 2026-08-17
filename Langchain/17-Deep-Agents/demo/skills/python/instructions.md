# Python — Instructions

## Language & style

- Target Python 3.10+. Use modern syntax: `match` statements where they
  genuinely simplify branching, `X | Y` union types, `list[str]` /
  `dict[str, int]` instead of `typing.List` / `typing.Dict`.
- Add type hints to function signatures (parameters and return types).
  Skip hints only for trivial local variables where inference is obvious.
- Use f-strings for interpolation, never `%`-formatting or `.format()`.
- Prefer `pathlib.Path` over `os.path` for filesystem paths.
- Use `dataclasses` (or `pydantic` models if the project already depends on
  pydantic) instead of plain dicts for structured data that has a fixed
  shape.
- Follow PEP 8 naming: `snake_case` for functions/variables, `PascalCase`
  for classes, `UPPER_SNAKE_CASE` for constants.
- Keep functions small and single-purpose. Extract a helper when a function
  mixes more than one level of abstraction.
- Docstrings: only add one when the function's purpose or contract isn't
  obvious from its name and signature. Don't restate the signature in prose.
- No bare `except:` — catch specific exceptions. Only add error handling for
  failure modes that can actually occur (e.g. a network call, a file that
  may not exist), not defensively "just in case."

## Dependencies & environment

- This workspace uses **uv** for dependency and environment management
  (see `pyproject.toml` / `uv.lock` at the project root). Use:
  - `uv add <package>` to add a dependency — never hand-edit
    `pyproject.toml` dependency lists and never `pip install` directly into
    the project venv.
  - `uv run <script or command>` to execute code inside the project's venv.
  - `uv sync` to install/update the environment from the lockfile.
- If a notebook (`.ipynb`) is involved, prefer editing it as notebook cells
  rather than converting it to a script, unless the user asks for a
  standalone script.

## Workflow before answering

1. Write the code.
2. Actually run it (`uv run python file.py`, `uv run pytest`, or an
   equivalent), rather than eyeballing correctness. If it can't be run in
   this environment (e.g. needs a real API key or external service the
   agent doesn't have), say so explicitly instead of claiming it works.
3. If it's a bug fix, reproduce the failure first, then confirm the fix
   resolves it — don't just patch and assume.
4. If tests exist for the touched code, run them. If none exist and the
   change is non-trivial logic, consider adding a focused test rather than
   a broad suite.

## Common pitfalls to flag or avoid

- Mutable default arguments (`def f(x, items=[])`) — use `None` and
  initialize inside the function.
- Catching `Exception` broadly and silently swallowing errors.
- Using `import *`.
- Blocking I/O inside `async def` functions without `await`ing an async
  call (defeats the purpose of async).
- Off-by-one errors in slicing/ranges — double-check boundary conditions
  when reviewing loops.
- Comparing floats with `==` instead of `math.isclose`.

## Explaining code or errors

When asked to explain a traceback or behavior:
- Identify the root cause first, in one or two sentences.
- Point to the exact file/line if available.
- Only then suggest a fix — don't dump a generic explanation of the
  exception type.
