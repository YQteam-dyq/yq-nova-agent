# yq-nova Python Client SDK

A minimal, **zero-dependency** HTTP client for the [yq-nova](..) Agent memory
service. It uses only the Python standard library (`urllib.request`), so no
`pip install` is required.

It talks to a **running `yq_nova` server** — start one first, then point this
client at its HTTP base URL (default `http://127.0.0.1:7999`).

## Installation

There is nothing to install — just put `yq_nova/` on your `PYTHONPATH`, or run
from this directory:

```bash
PYTHONPATH=. python your_script.py
```

Requires Python 3.8+.

## Usage

```python
from yq_nova import Client

client = Client("http://127.0.0.1:7999")
# Optionally authenticate:
# client = Client("http://127.0.0.1:7999", api_key="<your-key>")

# Health + stats
print(client.health())
print(client.stats())

# Remember a memory
result = client.remember(
    "yq-nova stores agent memory in SQLite",
    importance=0.8,
    tags=["nova", "storage"],
)
uuid = result["uuid"]
print("stored:", uuid)

# Recall relevant memories
for hit in client.recall("SQLite memory", top_k=5)["hits"]:
    print(hit["memory"]["content"], hit["final_score"])

# Forget (archive) the memory again
print(client.forget(uuid=uuid, mode="archive"))

# Fetch / hard-delete a single memory
print(client.get_memory(uuid))
client.delete_memory(uuid)
```

## Graph operations

```python
# Upsert an entity
client.upsert_entity("Alice", entity_type="person", description="Rust engineer")

# BFS-traverse the graph from an entity uuid
client.traverse("<entity-uuid>", max_depth=2)

# Auto-extract entities and relations from text
client.extract_and_link(
    "I love [[Rust]] and [[Tokio]] async runtime",
    opts={"enabled": True, "upsert_entities": True, "create_relations": False},
)
```

## Errors

Non-2xx responses raise `NovaApiError` with `code`, `message`, `status` and an
optional `trace_id`:

```python
from yq_nova import NovaApiError

try:
    client.get_memory("does-not-exist")
except NovaApiError as e:
    print(e.status, e.code, e.message)  # e.g. 404 not_found ...
```

## Running the tests

```bash
python3 -m py_compile yq_nova/client.py yq_nova/__init__.py   # syntax check
python3 -m pytest tests/ -q                                   # optional smoke test
```