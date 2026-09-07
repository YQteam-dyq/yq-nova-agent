use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tracing_subscriber::EnvFilter;
use yq_nova_core::config::StorageConfig;
use yq_nova_core::embedding::MockEmbeddingProvider;
use yq_nova_core::graph::{GraphService, TraverseOpts};
use yq_nova_core::memory::{
    ForgetInput, ForgetMode, MemoryService, ops_forget::ForgetTarget, ops_recall::RecallInput,
    ops_remember::RememberInput, ops_update::UpdateInput,
};
use yq_nova_core::storage::{Database, MemoryFilter, MemoryRepository, MemoryStatus, SqliteMemoryRepository};
use yq_nova_core::Uuid;

#[derive(Parser)]
#[command(name = "yq-nova-mcp", version = "0.3.0")]
struct Cli {
    #[arg(long, default_value = "./nova.db")]
    db_path: String,
}

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    #[serde(default)]
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let config = StorageConfig {
        db_path: cli.db_path.into(),
        ..Default::default()
    };
    let database = Database::open(config).await?;
    let embedding: Arc<MockEmbeddingProvider> = Arc::new(MockEmbeddingProvider::new(64));
    let memory = MemoryService::new(database.clone(), embedding);
    let graph = GraphService::new(database.clone());

    let tools = build_tool_list();

    let stdin = tokio::io::stdin();
    let reader = tokio::io::BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        let req: JsonRpcRequest = match serde_json::from_str(&trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError { code: -32700, message: e.to_string() }),
                };
                println!("{}", serde_json::to_string(&resp)?);
                continue;
            }
        };

        let id = req.id.clone().unwrap_or(serde_json::Value::Null);
        let is_notification = req.id.is_none();
        let resp = handle_request(req, &memory, &graph, &tools).await;
        let id = if id.is_null() { None } else { Some(id) };

        if is_notification {
            continue;
        }

        let output = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: resp.result,
            error: resp.error,
        };
        println!("{}", serde_json::to_string(&output)?);
    }

    let _ = database.close().await;
    Ok(())
}

struct HandlerResult {
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

async fn handle_request(
    req: JsonRpcRequest,
    memory: &MemoryService,
    graph: &GraphService,
    tools: &[ToolDef],
) -> HandlerResult {
    match req.method.as_str() {
        "initialize" => {
            let result = serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "yq-nova-mcp", "version": "0.3.0" }
            });
            HandlerResult { result: Some(result), error: None }
        }
        "notifications/initialized" => HandlerResult { result: None, error: None },
        "tools/list" => {
            let result = serde_json::json!({ "tools": tools });
            HandlerResult { result: Some(result), error: None }
        }
        "tools/call" => {
            let params = req.params.unwrap_or(serde_json::Value::Null);
            let name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
            handle_tool_call(name, &args, memory, graph).await
        }
        _ => HandlerResult {
            result: None,
            error: Some(JsonRpcError { code: -32601, message: format!("Method not found: {}", req.method) }),
        },
    }
}

async fn handle_tool_call(
    name: &str,
    args: &serde_json::Value,
    memory: &MemoryService,
    graph: &GraphService,
) -> HandlerResult {
    let result = match name {
        "nova_remember" => tool_remember(args, memory).await,
        "nova_recall" => tool_recall(args, memory).await,
        "nova_forget" => tool_forget(args, memory).await,
        "nova_memory_update" => tool_update(args, memory).await,
        "nova_stats" => tool_stats(memory, graph).await,
        "nova_traverse" => tool_traverse(args, graph).await,
        _ => {
            return HandlerResult {
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Unknown tool: {}", name),
                }),
            }
        }
    };

    match result {
        Ok(value) => {
            let text = serde_json::to_string(&value).unwrap_or_default();
            HandlerResult {
                result: Some(serde_json::json!({
                    "content": [{"type": "text", "text": text}]
                })),
                error: None,
            }
        }
        Err(e) => HandlerResult {
            result: Some(serde_json::json!({
                "isError": true,
                "content": [{"type": "text", "text": e.to_string()}]
            })),
            error: None,
        },
    }
}

async fn tool_remember(args: &serde_json::Value, memory: &MemoryService) -> Result<serde_json::Value> {
    let content = args
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: content"))?;
    let importance = args.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
    let tags: Vec<String> = args
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let metadata = args.get("metadata").cloned();

    let input = RememberInput {
        content,
        importance,
        tags: &tags,
        metadata: metadata.as_ref(),
        ..Default::default()
    };
    let out = memory.remember(input).await?;
    Ok(serde_json::json!({
        "uuid": out.uuid.to_string(),
        "duplicate": out.duplicate,
        "embedding_stored": out.embedding_stored,
        "entities_extracted": out.entities_extracted,
        "relations_extracted": out.relations_extracted,
        "tags": out.tags,
        "chunks": out.chunks,
    }))
}

async fn tool_recall(args: &serde_json::Value, memory: &MemoryService) -> anyhow::Result<serde_json::Value> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: query"))?;
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let entity_focus: Vec<String> = args
        .get("entity_focus")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let input = RecallInput {
        query,
        top_k,
        entity_focus,
        score_threshold: -1.0,
        similarity_threshold: -1.0,
        ..Default::default()
    };
    let out = memory.recall(input).await?;
    Ok(serde_json::to_value(out)?)
}

async fn tool_forget(args: &serde_json::Value, memory: &MemoryService) -> anyhow::Result<serde_json::Value> {
    let uuid_str = args
        .get("uuid")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: uuid"))?;
    let uuid = Uuid::parse_str(uuid_str)?;
    let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("soft");
    let forget_mode = match mode {
        "hard" => ForgetMode::Hard,
        "archive" => ForgetMode::Archive,
        _ => ForgetMode::Soft,
    };

    let input = ForgetInput {
        target: ForgetTarget::One(uuid),
        mode: forget_mode,
        ..Default::default()
    };
    let out = memory.forget(input).await?;
    Ok(serde_json::to_value(out)?)
}

async fn tool_update(args: &serde_json::Value, memory: &MemoryService) -> anyhow::Result<serde_json::Value> {
    let uuid_str = args
        .get("uuid")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: uuid"))?;
    let uuid = Uuid::parse_str(uuid_str)?;
    let content = args.get("content").and_then(|v| v.as_str()).map(String::from);
    let importance = args.get("importance").and_then(|v| v.as_f64()).map(|v| v as f32);
    let metadata = args.get("metadata").cloned();
    let tags: Vec<String> = args
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let input = UpdateInput {
        content,
        importance,
        metadata,
        tags: Some(tags),
        ..Default::default()
    };
    let out = memory.update(uuid, input).await?;
    Ok(serde_json::to_value(out)?)
}

async fn tool_stats(memory: &MemoryService, graph: &GraphService) -> Result<serde_json::Value> {
    let repo = SqliteMemoryRepository::new();
    let active = repo
        .count(
            &memory.database,
            &MemoryFilter {
                status_in: Some(vec![MemoryStatus::Active]),
                ..Default::default()
            },
        )
        .await
        .unwrap_or(0);
    let archived = repo
        .count(
            &memory.database,
            &MemoryFilter {
                status_in: Some(vec![MemoryStatus::Archived]),
                ..Default::default()
            },
        )
        .await
        .unwrap_or(0);
    let total_memories = active + archived;

    let entities: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entities")
        .fetch_one(&graph.database.pool)
        .await
        .unwrap_or(0);
    let relations: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM relations")
        .fetch_one(&graph.database.pool)
        .await
        .unwrap_or(0);

    Ok(serde_json::json!({
        "total_memories": total_memories,
        "active_memories": active,
        "archived_memories": archived,
        "total_entities": entities,
        "total_relations": relations,
    }))
}

async fn tool_traverse(args: &serde_json::Value, graph: &GraphService) -> anyhow::Result<serde_json::Value> {
    let uuid_str = args
        .get("start_uuid")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: start_uuid"))?;
    let uuid = Uuid::parse_str(uuid_str)?;
    let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(2) as u8;

    let opts = TraverseOpts {
        max_depth,
        ..Default::default()
    };
    let nodes = graph.traverse_graph(uuid, opts).await?;
    Ok(serde_json::to_value(nodes)?)
}

#[derive(Serialize)]
struct ToolDef {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

fn build_tool_list() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "nova_remember".into(),
            description: "Store a new memory with optional importance, tags, and metadata".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Memory content text"},
                    "importance": {"type": "number", "description": "Importance score 0.0-1.0", "default": 0.5},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                    "metadata": {"type": "object", "description": "Optional metadata key-value pairs"}
                },
                "required": ["content"]
            }),
        },
        ToolDef {
            name: "nova_recall".into(),
            description: "Search memories by semantic query with optional entity focus and mode".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "top_k": {"type": "number", "description": "Max results to return", "default": 10},
                    "entity_focus": {"type": "array", "items": {"type": "string"}, "description": "Entity names to anchor recall"},
                    "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "description": "Recall mode", "default": "semantic"}
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "nova_forget".into(),
            description: "Forget a memory by UUID with soft/archive/hard mode".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "uuid": {"type": "string", "description": "UUID of the memory to forget"},
                    "mode": {"type": "string", "enum": ["soft", "archive", "hard"], "description": "Forget mode", "default": "soft"}
                },
                "required": ["uuid"]
            }),
        },
        ToolDef {
            name: "nova_memory_update".into(),
            description: "Update an existing memory's content, importance, metadata, or tags".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "uuid": {"type": "string", "description": "UUID of the memory to update"},
                    "content": {"type": "string", "description": "New content text"},
                    "importance": {"type": "number", "description": "New importance score 0.0-1.0"},
                    "metadata": {"type": "object", "description": "New metadata key-value pairs"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Replace tag set"}
                },
                "required": ["uuid"]
            }),
        },
        ToolDef {
            name: "nova_stats".into(),
            description: "Get memory and graph statistics".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDef {
            name: "nova_traverse".into(),
            description: "Traverse the knowledge graph from a starting entity".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "start_uuid": {"type": "string", "description": "UUID of the starting entity"},
                    "max_depth": {"type": "number", "description": "Max traversal depth", "default": 2}
                },
                "required": ["start_uuid"]
            }),
        },
    ]
}
