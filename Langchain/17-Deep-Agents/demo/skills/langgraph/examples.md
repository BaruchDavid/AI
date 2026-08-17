# LangGraph / deepagents — Examples

## Example 1: Basic deep agent with a tool (matches `1-basicdeepagent.ipynb`)

```python
from deepagents import create_deep_agent

def web_search(query: str, max_results: int = 5) -> dict:
    """Run a web search and return the results."""
    return tavily_client.search(query, max_results=max_results)

deep_agent = create_deep_agent(
    model="groq:openai/gpt-oss-20b",
    tools=[web_search],
    system_prompt="You are a helpful assistant that can search the web.",
)

results = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "What is the latest news about AI?"}]}
)
print(results["messages"][-1]["content"])
```

## Example 2: In-memory (RAM) filesystem via `StateBackend`

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

agent = create_deep_agent(model="groq:openai/gpt-oss-20b", backend=StateBackend())
```
Use this when files are just scratch space for a single run — nothing
needs to survive after `invoke` returns.

## Example 3: Real disk persistence via `FilesystemBackend`

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="groq:openai/gpt-oss-20b",
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Create a file at /notes/todo.txt with the shopping list, then confirm.",
    }]
})
print(result["messages"][-1]["content"])
```
Files actually land on disk under `root_dir` — use when the user wants
inspectable output files.

## Example 4: Cross-thread persistence via `StoreBackend`

```python
import uuid
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend

agent = create_deep_agent(
    model="groq:openai/gpt-oss-20b",
    backend=StoreBackend(namespace=lambda rt: ("demo-user",)),
    store_name="my-store",
)

thread_1 = {"configurable": {"thread_id": str(uuid.uuid4())}}
agent.invoke({"messages": [{"role": "user", "content": "Write /notes/todo.txt"}]}, config=thread_1)

thread_2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
followup = agent.invoke(
    {"messages": [{"role": "user", "content": "Read /notes/todo.txt and tell me its content"}]},
    config=thread_2,
)
```
Note the file written on `thread_1` is readable from `thread_2` — this is
the key difference from `StateBackend`, which is why `StoreBackend` is the
right choice when multiple sessions must share files.

## Example 5: Choosing between the plain agent and a deep agent

User: "I just need the model to call one calculator tool and answer."

Good response: use `langchain.agents.create_agent`, not
`create_deep_agent` — no planning, filesystem, or subagents are needed, and
the deep agent's extra scaffolding would add overhead without benefit.
