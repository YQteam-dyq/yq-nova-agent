# Changelog

All notable changes to **yq-nova-agent** are documented in this file. The project follows [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-09-14

### Added

#### Multi-tenant namespaces (#12)
- Every memory, tag, entity and relation is now scoped to a namespace.
- New admin API `POST|GET /v1/namespaces` to create and list namespaces; a default `default` namespace is always present so single-tenant deployments are unaffected.
- Requests scope data through the `x-namespace` header; per-namespace API keys can be bound in `[server].namespace_keys` for tenant-scoped isolation.

#### Additional embedding providers (#9)
- New first-class embedding providers: **Zhipu** (`embedding.zhipu`), **Qwen / DashScope** (`embedding.qwen`), **Baichuan** (`embedding.baichuan`) and **Jina** (`embedding.jina`), alongside the existing OpenAI-compatible and mock providers.
- Each provider supports configurable model, dimensions, batch size, retry and request timeout.

#### Ecosystem SDKs & framework integrations (#10)
- New TypeScript SDK (`js/`) for Node.js and browsers.
- New Python client additions (`python/`), plus first-party **LangChain** (`langchain/langchain-yq-nova`) and **LlamaIndex** (`llamaindex/llama-index-yq-nova`) memory providers and tools.

#### Memory quality toolkit (#11)
- New `memory::quality` module exposed through the SDK / core service:
  - Deduplication (`dedup::detect`) to find near-duplicate memories.
  - Summarization (`summary::summarize_group`) to condense groups of related memories.
  - Importance rebalancing (`importance::rebalance`).
  - Association discovery (`associate::discover`).
- Recall now supports `rebalance_importance`.

#### LLM entity-relation auto extraction (#8)
- New `POST /v1/graph/extract-and-link` endpoint that uses a chat provider (`graph.extract_llm`) to extract entities and relations from text and link them into the graph.

#### Memory operations & tags (#6)
- New `POST /v1/memory/list` with filters, pagination and multiple sort orders.
- New `POST /v1/memory/remember-batch` for batch ingest (up to 200 items per round trip).
- Full tag management: `GET|PATCH|DELETE /v1/tags` with counting, rename and delete.

### Changed
- Workspace and crate versions bumped to `0.4.0` (`yq-nova-core`, `yq-nova-sdk`, `yq-nova-server`).
- Ecosystem SDK package versions bumped to `0.4.0` (`js`, `python`, `langchain`, `llamaindex`).
- Documentation unified to English and the README repaired (#5); internal clippy warnings cleared (#4).

## [0.3.0] - 2026-09-07

- Memory operations: PATCH update, merge, export/import, chunking, group chunks in recall.
- Graph memory: entity_focus recall, enforced metadata_match and traverse predicates, entity merge, hard-forget relation cleanup.
- New `yq-nova-mcp` crate exposing an MCP server; Agent integration docs and Python OpenAI-tools example.

## [0.2.0]

- Complete optimization suite; BSL-1.0 license change.

## [0.1.0]

- Initial beta release of yq-nova-agent.