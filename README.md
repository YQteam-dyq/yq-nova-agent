<p align="center">
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/sqlite-3.x-blue?logo=sqlite" alt="SQLite">
  <img src="https://img.shields.io/badge/license-BSL--1.1-red" alt="License">
  <img src="https://img.shields.io/badge/status-alpha-yellow" alt="Status">
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
| **Semantic search** | Pluggable embedding providers (OpenAI-compatible, mock) |
| **Graph state** | Entity-relation graph with recursive BFS traversal |
| **Hybrid ranking** | RRF fusion of semantic + keyword (FTS5) + graph signals |
| **SQLite-backed** | WAL mode, composite indexes, production PRAGMAs |
| **Background GC** | TTL expiry, importance-based forgetting, graceful shutdown |
| **CLI subcommands** | `yq-nova remember`, `recall`, `forget`, `stats` — no server needed |
| **Embedded SDK** | `yq-nova-core` as a Rust library, `yq-nova-sdk` as HTTP client |

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
```

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
├── yq-nova-core/    # Core library: storage, memory ops, embedding, graph
├── yq-nova-server/  # HTTP server (axum) + CLI binary
└── yq-nova-sdk/     # Rust HTTP client SDK with builder API
migrations/          # SQLite schema migrations
```

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
```

---

## License

**Business Source License 1.1** — see [LICENSE](LICENSE) for details.

Non-production and personal use are **free**. Commercial and production use require a separate license. Contact the licensor for commercial licensing inquiries.