# langchain-yq-nova

A [LangChain](https://python.langchain.com) memory provider and tool set for the
[yq-nova agent memory service](..). It lets LangChain agents persist, retrieve
and forget memories through the yq-nova HTTP API.

## Features

- `YqNovaChatMessageHistory` - a `BaseChatMessageHistory` backed by yq-nova,
  so `ConversationBufferMemory` and similar memories work unchanged.
- `YqNovaMemory` - a `BaseMemory` that recalls relevant history for the current
  input and saves each exchange.
- `create_memory_tools()` - ready-to-use `nova_remember`, `nova_recall` and
  `nova_forget` tools to register on an `AgentExecutor` or
  `create_openai_tools_agent`.

## Install

```bash
pip install langchain-yq-nova
```

Requires a running `yq-nova` server (default `http://127.0.0.1:7999`).

## Usage

### Chat message history

```python
from yq_nova import Client
from langchain.memory import ConversationBufferMemory
from langchain_yq_nova import YqNovaChatMessageHistory

client = Client("http://127.0.0.1:7999")
history = YqNovaChatMessageHistory(client, namespace="alice")

history.add_message(HumanMessage(content="Hi, I like Rust"))
print(history.messages)

memory = ConversationBufferMemory(chat_memory=history, return_messages=True)
```

### BaseMemory provider

```python
from langchain_yq_nova import YqNovaMemory

provider = YqNovaMemory(client, namespace="alice", top_k=10)
provider.save_context({"input": "My name is Alice"}, {"output": "Nice to meet you Alice"})
vars = provider.load_memory_variables({"input": "What is my name?"})
print(vars["history"])
```

### Agent tools

```python
from langchain_yq_nova import create_memory_tools

tools = create_memory_tools(client)
```

Register `tools` on your agent, then the model can store, search and forget
memories at runtime.

The package reuses the zero-dependency `yq_nova.Client`, so no extra HTTP
library is required. See `examples/` for a full walkthrough.

## License

Business Source License 1.1.