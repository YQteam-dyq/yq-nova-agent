<p align="center">
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/sqlite-3.x-blue?logo=sqlite" alt="SQLite">
  <img src="https://img.shields.io/badge/license-BSL--1.1-red" alt="License">
  <img src="https://img.shields.io/badge/status-beta-green" alt="Status">
  <img src="https://github.com/YQteam-dyq/yq-nova-agent/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/crates/v/yq-nova-core?logo=rust" alt="yq-nova-core">
  <img src="https://img.shields.io/crates/v/yq-nova-sdk?logo=rust" alt="yq-nova-sdk">
  <img src="https://img.shields.io/crates/v/yq-nova-server?logo=rust" alt="yq-nova-server">
  <img src="https://img.shields.io/github/stars/YQteam-dyq/yq-nova-agent?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
</p>

<h1 align="center">yq-nova-agent</h1>
<p align="center"><b>Lightweight, single-file Agent memory & state layer</b><br>
Semantic search · Graph traversal · Remember / Recall / Forget API</p>

---

## Who is this for?

**Agent developers and AI engineers** who need a **local, embeddable memory layer** for their agents — without spinning up a vector database, without a cloud dependency, without complexity.

You're building an agent that needs to:
- Remember what it learned across conversations
- Recall semantically relevant information from past sessions
- Track entities and relationships over time
- Forget stale or unimportant data automatically

**yq-nova-agent** gives you all of this in a **single SQLite file**, zero external services at runtime.

---

## Why yq-nova?

| Problem | yq-nova solution |
|---------|-----------------|
| Vector DBs are heavy (Pinecone, Qdrant, Weaviate…) | **Zero external dependencies** — just SQLite |
| Mem0 is cloud-only | **Fully local**, single binary, no telemetry |
| Agent memory is ephemeral | **Persistent**, survives restarts, TTL-aware GC |
| No graph = no relational context | **Entity-relation graph** with BFS traversal |
| Hybrid search is hard to integrate | **Built-in**: semantic + keyword (FTS5) + graph ranking |

---

## Features

| Capability | Description |
|-----------|-------------|
| **remember / recall / forget** | Three core operations, HTTP API or Rust SDK |
| **Semantic search** | Pluggable embedding providers: OpenAI-compatible, Zhipu, Qwen, Baichuan, Jina, and mock |
| **Graph state** | Entity-relation graph with recursive BFS traversal |
| **Multi-tenant namespaces** | Isolated, per-namespace data + API keys via the `x-namespace` header |
| **Hybrid ranking** | RRF fusion of semantic + keyword (FTS5) + graph signals |
| **Memory quality** | Deduplication, summarization, importance rebalancing and association discovery |
| **SQLite-backed** | WAL mode, composite indexes, production PRAGMAs |
| **Background GC** | TTL expiry, importance-based forgetting, graceful shutdown |
| **CLI subcommands** | `yq-nova remember`, `recall`, `forget`, `stats` — no server needed |
| **Embedded SDK** | `yq-nova-core` as a Rust library, `yq-nova-sdk` as HTTP client |

---

## Installation

### From crates.io

```bash
# Install the CLI + HTTP server binary
cargo install yq-nova-server --locked

# Or add as a library dependency
cargo add yq-nova-core    # core memory & graph operations
cargo add yq-nova-sdk     # HTTP client SDK
```

### From source

```bash
git clone https://github.com/YQteam-dyq/yq-nova-agent.git
cd yq-nova-agent
cargo build --release -p yq-nova-server --bin yq-nova
```

---

## Docker

### Docker Compose (recommended)

```bash
docker compose up -d
```

### Build and run directly

```bash
docker build -t yq-nova .
docker run -p 7999:7999 \
  -v yq-nova-data:/data \
  -e YQ_NOVA_EMBEDDING__DEFAULT_PROVIDER=mock \
  yq-nova serve
```

> The `yq-nova-data` volume is mounted at `/data` inside the container and the SQLite database is persisted at `/data/yq-nova.db`, so the data survives container recreation.

---

## Quick Start

```bash
# 1. Build the binary
cargo build --release -p yq-nova-server --bin yq-nova

# 2. Configure
cat > yq-nova.toml << 'EOF'
[storage]
db_path = "nova.db"

[embedding]
default_provider = "mock"
EOF

# 3. Start the server
YQ_NOVA_EMBEDDING__DEFAULT_PROVIDER=mock ./target/release/yq-nova serve

# 4. Remember something
curl -X POST http://127.0.0.1:7999/v1/memory/remember \
  -H 'Content-Type: application/json' \
  -d '{"content": "yq-nova stores memory in SQLite", "importance": 0.8, "tags": ["nova", "storage"]}'

# 5. Recall
curl -X POST http://127.0.0.1:7999/v1/memory/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "SQLite memory", "top_k": 5}'

# 6. Forget
curl -X DELETE http://127.0.0.1:7999/v1/memory/1a2b3c4d
```

---

## CLI (no HTTP server)

```bash
# Direct core operations — no server needed
yq-nova remember "Your content here" --tag rust --importance 0.9
yq-nova recall "query text" --top-k 10 --mode hybrid --graph
yq-nova forget --uuid <uuid>
yq-nova stats

# Browse memories with filters, pagination and sorting
yq-nova list --limit 20 --sort importance_desc --tag rust
yq-nova list --status active,archived --importance-min 0.5 --json

# Inspect tags and how many memories carry each one
yq-nova tags --limit 50 --json
```

---

## List, Batch Ingest and Tag Management

### HTTP API

```bash
# List memories with filters, pagination and sorting
curl -X POST http://127.0.0.1:7999/v1/memory/list \
  -H 'Content-Type: application/json' \
  -d '{"filter": {"tags_all": ["nova"], "importance_min": 0.5}, "limit": 20, "offset": 0, "sort": "importance_desc"}'

# Ingest up to 200 memories in a single round trip
curl -X POST http://127.0.0.1:7999/v1/memory/remember-batch \
  -H 'Content-Type: application/json' \
  -d '{"items": [{"content": "first", "tags": ["batch"]}, {"content": "second", "tags": ["batch"]}], "continue_on_error": true}'

# Inspect tags together with their memory counts
curl 'http://127.0.0.1:7999/v1/tags?limit=100&offset=0'

# Rename a tag while every memory association stays intact
curl -X PATCH http://127.0.0.1:7999/v1/tags/nova \
  -H 'Content-Type: application/json' \
  -d '{"new_name": "yq-nova"}'

# Delete a tag: memories are kept, the association is dropped
curl -X DELETE http://127.0.0.1:7999/v1/tags/yq-nova
```

`POST /v1/memory/list` accepts `sort` values of `created_desc`, `created_asc`, `importance_desc`, `importance_asc` and `accessed_desc`. It returns `total` alongside the current page, so a client can paginate without issuing a second count request. `POST /v1/memory/remember-batch` reports one result per item: each entry carries its `index`, the resulting `uuid`, whether it was a `duplicate`, and an `error` string when the item was rejected.

### Rust SDK

```rust
use yq_nova_sdk::{EmbeddedNova, ListInput, MemorySortOrder, MemoryFilter};

# async fn demo(nova: &EmbeddedNova) -> yq_nova_sdk::NovaResult<()> {
let page = nova
    .list_memories(ListInput {
        filter: MemoryFilter {
            tags_all: Some(vec!["nova".into()]),
            ..Default::default()
        },
        limit: 20,
        offset: 0,
        sort: MemorySortOrder::ImportanceDesc,
    })
    .await?;
println!("total={} count={}", page.total, page.count);

let tags = nova.list_tags(100, 0).await?;
for tag in tags.items {
    println!("{} -> {}", tag.name, tag.memory_count);
}
# Ok(())
# }
```

The same methods exist on the HTTP client, plus `remember_batch`, `rename_tag` and `delete_tag`.

### Python client

```python
from yq_nova import Client

client = Client("http://127.0.0.1:7999")
page = client.list_memories(filter={"tags_all": ["nova"]}, limit=20, sort="importance_desc")
client.remember_batch([{"content": "first"}, {"content": "second"}])
client.list_tags(limit=100)
client.rename_tag("nova", "yq-nova")
client.delete_tag("yq-nova")
```

### MCP tools

The MCP server exposes two additional tools: `nova_list` lists memories with tag, status, pagination and sorting options, and `nova_tags` lists tags together with their memory counts.

---

## Multi-tenant Namespaces

Since v0.4.0 every memory, tag, entity and relation is scoped to a namespace. A default `default` namespace is always present, so existing single-tenant deployments keep working without changes.

### HTTP API

```bash
# Create a namespace (admin only)
curl -X POST http://127.0.0.1:7999/v1/namespaces \
  -H 'Content-Type: application/json' \
  -d '{"name": "team-a", "description": "Customer A data"}'

# List namespaces with pagination (admin only)
curl 'http://127.0.0.1:7999/v1/namespaces?limit=100&offset=0'
```

### Scoping requests

- Pass the `x-namespace` header to target a namespace: `curl -H 'x-namespace: team-a' ...`.
- When `x-namespace` is absent, requests target the `default` namespace.
- Bind per-namespace API keys in `[server].namespace_keys` (see Configuration). Requests carrying a valid namespace key are treated as tenant-scoped and restricted to that namespace; requests with the global `auth_token` are admin-scoped and may access all namespaces.

---

## Memory Quality

The core library ships a `memory::quality` module for keeping memory stores clean and relevant. The methods are available on the Rust SDK (`EmbeddedNova` / HTTP client):

- **Deduplication** — `dedup::detect(...)` finds near-duplicate memories so stale copies can be archived.
- **Summarization** — `summary::summarize_group(...)` condenses a group of related memories into a single summary entry.
- **Importance rebalancing** — `importance::rebalance(...)` re-scales importance scores across a namespace.
- **Association discovery** — `associate::discover(...)` surfaces latent semantic associations between memories.

These primitives complement the recall-time `rebalance_importance` flag and give agents a toolkit for automatic memory hygiene.

---

## LLM Entity-Relation Extraction

Set `graph.extract_llm` to an available chat provider to let your agent extract entities and relations from memory content with an LLM:

```bash
curl -X POST http://127.0.0.1:7999/v1/graph/extract-and-link \
  -H 'Content-Type: application/json' \
  -d '{"text": "yq-nova supports Zhipu and Qwen embedding providers", "opts": {"enabled": true}}'
```

Entities become graph nodes and relations are linked between them, feeding the existing `traverse` and `entities` / `relations` endpoints so recall can walk the extracted knowledge graph.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                     Client                       │
│  (HTTP / Rust SDK embedded / CLI subcommands)   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              yq-nova-server                      │
│  axum HTTP · DTO validation · middleware stack   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              yq-nova-core                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Memory   │  │ Graph    │  │ Embedding     │  │
│  │  recall  │  │ entities │  │ OpenAI compat │  │
│  │  remember│  │ relations│  │ Mock provider │  │
│  │  forget  │  │ BFS      │  │ Retry + batch │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       │             │               │            │
│  ┌────▼─────────────▼───────────────▼────────┐   │
│  │           SQLite (sqlx)                   │   │
│  │  memory_items · entities · relations      │   │
│  │  embeddings · tags · FTS5 · migrations    │   │
│  └───────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
crates/
├── yq-nova-core/      # Core library: storage, memory ops, embedding, graph
│   └── migrations/    # SQLite schema migrations
├── yq-nova-server/    # HTTP server (axum) + CLI binary
├── yq-nova-sdk/       # Rust HTTP client SDK with builder API
└── yq-nova-mcp/       # MCP server exposing the memory API to AI clients
```

## Ecosystem SDKs

```
python/                # Zero-dependency Python client SDK (full v1 API)
js/                    # TypeScript SDK for Node.js and browsers (npm)
langchain/             # langchain-yq-nova: LangChain memory provider + tools
llamaindex/            # llama-index-yq-nova: LlamaIndex memory store + tools
```

The `python/` client covers every v1 endpoint. The `js/` SDK mirrors the same
23 endpoints with typed methods for Node.js and browsers. `langchain-yq-nova`
and `llama-index-yq-nova` wrap `remember / recall / forget` as native memory
providers and agent tools for the two frameworks.

---

## Configuration

All settings via TOML file or `YQ_NOVA_*` environment variables:

```toml
[server]
bind = "127.0.0.1:7999"

[storage]
db_path = "./nova.db"
wal_mode = true

[embedding]
default_provider = "openai"
[embedding.openai_compatible.default]
api_key = "${OPENAI_API_KEY}"
base_url = "https://api.openai.com/v1"
model = "text-embedding-3-small"
dimensions = 1536

# Zhipu, Qwen, Baichuan and Jina embedding providers (#9)
# [embedding.zhipu.default]
# api_key = "${ZHIPU_API_KEY}"
# model = "embedding-3"
# dimensions = 1024
# [embedding.qwen.default]
# api_key = "${DASHSCOPE_API_KEY}"
# model = "text-embedding-v3"
# [embedding.baichuan.default]
# api_key = "${BAICHUAN_API_KEY}"
# [embedding.jina.default]
# api_key = "${JINA_API_KEY}"
# model = "jina-embeddings-v3"

# Multi-tenant namespaces (#12): bind per-namespace API keys for tenant isolation.
[server]
namespace_keys = { "team-a" = "${TEAM_A_NAMESPACE_KEY}" }

[graph]
# Set graph.extract_llm to a chat provider name to enable LLM entity-relation extraction.
extract_llm = "default"
[graph.openai_compatible_chat.default]
api_key = "${OPENAI_API_KEY}"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
```

---

## CI

All workflows live in [`.github/workflows`](.github/workflows) and run on GitHub Actions:

- [`ci.yml`](.github/workflows/ci.yml) - formatting, Clippy, build and unit tests for the whole workspace.
- [`english-only.yml`](.github/workflows/english-only.yml) - enforces the [English-only policy](CONTRIBUTING.md#language-policy-mandatory) on pull request titles, pull request descriptions and added diff lines.

Run the same checks locally before opening a pull request:

```bash
cargo +nightly fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
```

---

## License

**Business Source License 1.1** — see [LICENSE](LICENSE) for details.

Non-production and personal use are **free**. Commercial and production use require a separate license. Contact the licensor for commercial licensing inquiries.

---

## Agent Integration

### MCP Quick Start

yq-nova ships an MCP (Model Context Protocol) server, so the yq-nova memory API can be used directly from any MCP-capable AI client.

```bash
# Start the MCP server (the yq-nova-mcp binary must be built first)
./yq-nova-mcp --db-path ./nova.db
```

Then add the following to the configuration file of **Claude Desktop** or another MCP client:

```json
{
  "mcpServers": {
    "yq-nova": {
      "command": "./yq-nova-mcp",
      "args": ["--db-path", "./nova.db"]
    }
  }
}
```

The MCP server exposes six tools: `nova_remember`, `nova_recall`, `nova_forget`, `nova_memory_update`, `nova_stats` and `nova_traverse`. AI clients discover and call them automatically.

### OpenAI tool-calling schema example

The JSON schema snippets below can be used directly for OpenAI function calling:

```json
[
  {
    "type": "function",
    "function": {
      "name": "nova_remember",
      "description": "Store a memory",
      "parameters": {
        "type": "object",
        "properties": {
          "content": {"type": "string"},
          "importance": {"type": "number", "default": 0.5},
          "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "nova_recall",
      "description": "Retrieve relevant memories",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "top_k": {"type": "integer", "default": 5}
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "nova_forget",
      "description": "Archive or delete a memory",
      "parameters": {
        "type": "object",
        "properties": {
          "uuid": {"type": "string"},
          "mode": {"type": "string", "enum": ["soft", "archive", "hard"], "default": "soft"}
        },
        "required": ["uuid"]
      }
    }
  }
]
```

A typical OpenAI function-calling flow:

```
1. Pass the schemas above as the tools argument of chat.completions.create()
2. When the model returns tool_calls, parse name and arguments
3. Call the yq-nova HTTP API (remember / recall / forget) according to name
4. Return the result to the model as a tool message and continue the conversation
```

A complete runnable example is available at [`python/examples/openai_memory_tools.py`](python/examples/openai_memory_tools.py).

### LangChain / LlamaIndex integration

yq-nova integrates easily into LangChain or LlamaIndex agent workflows:

- **LangChain**: call the yq-nova HTTP API through `httpx` or the standard `urllib`, wrap `remember` / `recall` / `forget` as `Tool` instances, then register them on an `AgentExecutor` or `create_openai_tools_agent`.
- **LlamaIndex**: wrap the yq-nova operations as `ToolMetadata` through `FunctionTool`, define the matching JSON schema, and use them as tools of an `OpenAIAgent` or `ReActAgent`.
- **General principle**: whichever framework is used, the core step is the same - map the three yq-nova operations (remember / recall / forget) to function-calling tool schemas and call the yq-nova server API through an HTTP client.
