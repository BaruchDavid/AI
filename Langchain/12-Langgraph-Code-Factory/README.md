# Coding Agent - Refactored Structure

Clean architecture following Single Responsibility Principle.

## Project Structure

```
code_generator/
├── tools/
│   ├── __init__.py
│   └── dev_tools.py          
├── agents/
│   ├── __init__.py
│   └── code_agent.py         
├── graph/
│   ├── __init__.py
│   └── graph_builder.py     
├── main.py                   
└── README.md                  
```

## Responsibilities

### tools/dev_tools.py
- Define individual tools (@tool decorator)
- Each tool has ONE specific purpose
- Provide `get_all_tools()` for easy tool collection

### agents/code_agent.py
- Define `AgentState` (TypedDict)
- Manage LLM configuration
- Bind tools to LLM
- Process state through LLM (the agent node logic)

### graph/graph_builder.py
- Build the StateGraph
- Add nodes and edges
- Configure conditional routing (tools_condition)
- Compile the graph

### main.py
- Environment setup (API keys)
- Orchestrate all components
- Run the agent with tasks
- Display results

## Usage

```python
python main.py
```

## Adding New Tools

1. Add tool function in `tools/dev_tools.py` with `@tool` decorator
2. Add to `get_all_tools()` return list
3. The graph will automatically use it!

## Adding New Agents

1. Create new agent class in `agents/` (inherit pattern from CodeAgent)
2. Import in main.py
3. Use GraphBuilder to build workflow

## Benefits

✅ **Single Responsibility**: Each module has ONE clear purpose
✅ **Testability**: Easy to unit test each component
✅ **Scalability**: Easy to add new tools/agents/workflows
✅ **Maintainability**: Changes isolated to specific modules
✅ **Reusability**: Components can be reused in different configurations