# yq-nova JS/TS SDK

TypeScript client for the [yq-nova agent memory service](..). It covers every
HTTP endpoint (20+ APIs) with typed methods, and runs in both **Node.js (18+)**
and **browsers** without any runtime dependency.

## Install

```bash
npm install yq-nova
```

## Usage

```ts
import { NovaClient } from "yq-nova";

const client = new NovaClient("http://127.0.0.1:7999", {
  apiKey: "optional-key",
  timeoutMs: 30000,
});

await client.remember({
  content: "yq-nova stores agent memory in SQLite",
  importance: 0.8,
  tags: ["nova", "storage"],
});

const { hits } = await client.recall({ query: "SQLite memory", top_k: 5 });
for (const hit of hits) {
  console.log(hit.memory.content, hit.final_score);
}

const page = await client.listMemories({
  filter: { tags_all: ["nova"] },
  sort: "importance_desc",
  limit: 20,
});
console.log(page.total, page.items.length);
```

In a browser the global `fetch` is used automatically. You can inject a custom
fetch (for example one that adds auth headers):

```ts
const client = new NovaClient("http://127.0.0.1:7999", {
  fetch: myCoolFetch,
});
```

## API coverage

| Method | HTTP endpoint |
|--------|---------------|
| `health()` | `GET /v1/health` |
| `stats()` | `GET /v1/stats` |
| `remember()` | `POST /v1/memory/remember` |
| `rememberBatch()` | `POST /v1/memory/remember-batch` |
| `recall()` | `POST /v1/memory/recall` |
| `listMemories()` | `POST /v1/memory/list` |
| `forget()` | `POST /v1/memory/forget` |
| `getMemory()` | `GET /v1/memory/:uuid` |
| `updateMemory()` | `PATCH /v1/memory/:uuid` |
| `deleteMemory()` | `DELETE /v1/memory/:uuid` |
| `exportMemories()` | `POST /v1/memory/export` |
| `importMemories()` | `POST /v1/memory/import` |
| `mergeMemories()` | `POST /v1/memory/merge` |
| `listTags()` | `GET /v1/tags` |
| `renameTag()` | `PATCH /v1/tags/:name` |
| `deleteTag()` | `DELETE /v1/tags/:name` |
| `extractAndLink()` | `POST /v1/graph/extract-and-link` |
| `upsertEntity()` | `POST /v1/graph/entities` |
| `listEntities()` | `GET /v1/graph/entities` |
| `mergeEntities()` | `POST /v1/graph/entities/merge` |
| `upsertRelation()` | `POST /v1/graph/relations` |
| `listRelations()` | `GET /v1/graph/relations` |
| `traverse()` | `POST /v1/graph/traverse` |

## Errors

Every non-2xx response throws a `NovaApiError` carrying `code`, `message`,
`status` and an optional `traceId`. Network and timeout failures throw a
`NovaApiError` with status `0` and code `network` or `timeout`.

## Building from source

```bash
npm install
npm run build
```

## License

Business Source License 1.1.