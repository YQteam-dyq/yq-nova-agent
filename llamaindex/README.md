# llama-index-yq-nova

A [LlamaIndex](https://www.llamaindex.ai) memory store and tool set for the
[yq-nova agent memory service](..). It gives LlamaIndex agents a persistent,
semantic memory backed by the yq-nova HTTP API.

## Features

- `YqNovaMemory` - a `BaseMemory` implementation that stores each `ChatMessage`
  in yq-nova and recalls the most relevant messages for the current input.
  Works with `AgentRunner`, `ChatMemoryBuffer`-style memory slots and
  `ReActAgent` / `FunctionAgent`.
- `create_memory_tools()` - ready-to-use `nova_remember`, `nova_recall` and
  `nova_forget` `FunctionTool`s to register on an agent.

## Install

```bash
pip install llama-index-yq-nova
```

Requires a running `yq-nova` server (default `http://127.0.0.1:7999`).

## Usage

### Memory store

```python
from yq_nova import Client
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index_yq_nova import YqNovaMemory

client = Client("http://127.0.0.1:7999")
memory = YqNovaMemory(client, namespace="alice")

memory.put(ChatMessage(role=MessageRole.USER, content="I like Rust"))
history = memory.get("What does Alice like?")
print(history)

memory.reset()
```

`YqNovaMemory` plugs into a LlamaIndex agent via the standard memory protocol:

```python
from llama_index.core.agent import AgentRunner
from llama_index.core.agent.runner.base import AgentState

state = AgentState(chat_history=memory.get_all())
runner = AgentRunner(agent=..., chat_history=memory)
```

### Agent tools

```python
from llama_index_yq_nova import create_memory_tools

tools = create_memory_tools(client)
```

Pass `tools` to a `FunctionAgent` or `ReActAgent` so the model can store, search
and forget memories at runtime.

The package reuses the zero-dependency `yq_nova.Client`, so no extra HTTP
library is required. See `examples/` for a full walkthrough.

## License

Business Source License 1.1.