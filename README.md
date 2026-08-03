# yq-nova-agent

**Lightweight, single-file Agent memory & state layer.**

Store, recall, and forget agent memories with semantic search, graph traversal, and a simple API — all in a single SQLite file, zero external dependencies at runtime.

## Features

- **remember / recall / forget** — three core operations, simple HTTP API or Rust SDK
- **Semantic search** — pluggable embedding providers (OpenAI-compatible, mock)
- **Graph state** — entity-relation graph with recursive BFS traversal
- **Hybrid ranking** — combines semantic + keyword (FTS5) + graph signals via RRF
- **SQLite-backed** — WAL mode, composite indexes, configurable PRAGMAs for production
- **Background GC** — TTL expiry, importance-based forgetting, graceful shutdown
- **CLI ready** — `yq-nova remember`, `yq-nova recall`, `yq-nova forget`, `yq-nova stats` without HTTP server

## Quick Start

```bash
# 1. Build
cargo build --release -p yq-nova-server --bin yq-nova

# 2. Configure
cat > yq-nova.toml << 'EOF'
[storage]
db_path = "nova.db"

[embedding]
default_provider = "mock"
EOF

# 3. Start server
YQ_NOVA_EMBEDDING__DEFAULT_PROVIDER=mock ./target/release/yq-nova serve

# 4. Remember
curl -X POST http://127.0.0.1:7999/v1/memory/remember \
  -H 'Content-Type: application/json' \
  -d '{"content": "yq-nova stores memory in SQLite", "importance": 0.8, "tags": ["nova", "storage"]}'

# 5. Recall
curl -X POST http://127.0.0.1:7999/v1/memory/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "SQLite memory", "top_k": 5}'

# 6. Forget
curl -X DELETE http://127.0.0.1:7999/v1/memory/<uuid>
```

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
│  │ remember │  │ entities │  │ OpenAI compat │  │
│  │ recall   │  │ relations│  │ Mock provider │  │
│  │ forget   │  │ BFS      │  │ Retry + batch │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       │             │               │            │
│  ┌────▼─────────────▼───────────────▼────────┐   │
│  │           SQLite (sqlx)                   │   │
│  │  memory_items · entities · relations      │   │
│  │  embeddings · tags · FTS5 · migrations    │   │
│  └───────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## Project Structure

```
crates/
├── yq-nova-core/    # Core library: storage, memory ops, embedding, graph
├── yq-nova-server/  # HTTP server (axum) + CLI binary
└── yq-nova-sdk/     # Rust HTTP client SDK with builder API
migrations/          # SQLite schema migrations
```

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

## CLI

```bash
# Direct core operations (no HTTP server needed)
yq-nova remember "Your content here" --tag rust --importance 0.9
yq-nova recall "query text" --top-k 10 --mode hybrid --graph
yq-nova forget --uuid <uuid>
yq-nova stats
```

## License

Business Source License 1.1 — see [LICENSE](LICENSE) for details.

Non-production and personal use are free. **Commercial and production use require a separate license.** Contact the licensor for commercial licensing inquiries.