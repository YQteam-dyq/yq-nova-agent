<p align="center">
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/sqlite-3.x-blue?logo=sqlite" alt="SQLite">
  <img src="https://img.shields.io/badge/license-BSL--1.1-red" alt="License">
  <img src="https://img.shields.io/badge/status-beta-green" alt="Status">
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
| **Semantic search** | Pluggable embedding providers (OpenAI-compatible, mock) |
| **Graph state** | Entity-relation graph with recursive BFS traversal |
| **Hybrid ranking** | RRF fusion of semantic + keyword (FTS5) + graph signals |
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

### Compose（推荐）

```bash
docker compose up -d
```

### 直接构建并运行

```bash
docker build -t yq-nova .
docker run -p 7999:7999 \
  -v yq-nova-data:/data \
  -e YQ_NOVA_EMBEDDING__DEFAULT_PROVIDER=mock \
  yq-nova serve
```

> 数据卷 `yq-nova-data` 挂载到容器内 `/data`，SQLite 数据库持久化在 `/data/yq-nova.db`，容器重建后数据不丢失。

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

---

## Agent 集成

### MCP 快速开始

yq-nova 提供 MCP (Model Context Protocol) 服务，让你可以直接在支持 MCP 的 AI 客户端中使用 yq-nova 的 memory 能力。

```bash
# 启动 MCP 服务（假设 yq-nova-mcp 二进制已构建）
./yq-nova-mcp serve --db-path ./nova.db
```

启动后，在 **Claude Desktop** 或其他 MCP 客户端的配置文件中添加：

```json
{
  "mcpServers": {
    "yq-nova": {
      "command": "./yq-nova-mcp",
      "args": ["serve", "--db-path", "./nova.db"]
    }
  }
}
```

MCP 服务暴露三个核心工具：`nova_remember`、`nova_recall`、`nova_forget`，AI 客户端会自动发现并调用。

### OpenAI 工具调用 schema 示例

以下 JSON schema 片段可直接用于 OpenAI function-calling：

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

典型的 OpenAI function-calling 调用流程：

```
1. 将上述 schema 作为 tools 参数传给 chat.completions.create()
2. 当模型返回 tool_calls 时，解析 name 和 arguments
3. 根据 name 调用 yq-nova HTTP API（remember / recall / forget）
4. 将结果作为 tool 消息返回给模型继续对话
```

完整可运行示例见 [`python/examples/openai_memory_tools.py`](python/examples/openai_memory_tools.py)。

### LangChain / LlamaIndex 对接思路

yq-nova 可以轻松集成到 LangChain 或 LlamaIndex 的 agent 工作流中：

- **LangChain**: 通过 `httpx` 或标准 `urllib` 调用 yq-nova HTTP API，将 `remember` / `recall` / `forget` 封装为 `Tool` 实例，然后注册到 `AgentExecutor` 或 `create_openai_tools_agent`。
- **LlamaIndex**: 通过 `FunctionTool` 将 yq-nova 的操作包装成 `ToolMetadata`，定义对应的 JSON schema 后即可作为 `OpenAIAgent` 或 `ReActAgent` 的工具使用。
- **通用原则**: 无论使用哪种框架，核心都是将 yq-nova 的三个操作（remember / recall / forget）映射为 function-calling 工具 schema，然后通过 HTTP 客户端调用 yq-nova 服务端 API。
