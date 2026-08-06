//! Crate-level integration tests for the embedded-mode SDK (`EmbeddedNova`).
//!
//! These exercise the full remember → recall → forget pipeline and the graph
//! extract-and-link path against a real (temp) SQLite database, with no
//! network involved.

use std::{path::PathBuf, sync::Arc};

use yq_nova_core::{
    Uuid,
    config::StorageConfig,
    embedding::{MockEmbeddingProvider, SharedEmbeddingProvider},
    error::ErrorCode,
    graph::{GraphExtractOpts, GraphService},
    graph::extractor::RegexWikiExtractor,
    memory::MemoryService,
    storage::Database,
};
use yq_nova_sdk::{EmbeddedNova, http_client};

/// Make a unique temp DB path under `std::env::temp_dir()`.
fn tmp_db(tag: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut p = std::env::temp_dir();
    p.push(format!("yqnova-embedded-{tag}-{}.db", n ^ 0x9e3779b97f4a7c15));
    p
}

#[tokio::test]
async fn remember_recall_forget_flow() {
    let nova = EmbeddedNova::open(tmp_db("rr")).await.expect("open embedded nova");

    // --- remember ---
    let out = nova
        .remember(http_client::RememberRequest {
            content: "Rust memory: impl Deref for MyBox via Box-like layout".into(),
            importance: 0.9,
            tags: vec!["rust".into()],
            ..Default::default()
        })
        .await
        .expect("remember");
    assert!(!out.uuid.is_nil(), "new memory must get a non-nil uuid");

    // --- get_memory ---
    let rec = nova.get_memory(out.uuid).await.expect("get_memory");
    assert_eq!(rec.content, "Rust memory: impl Deref for MyBox via Box-like layout");

    // --- recall (semantic mode, mock embedder) ---
    let recall = nova
        .recall(http_client::RecallRequest {
            query: "rust deref".into(),
            top_k: 5,
            ..Default::default()
        })
        .await
        .expect("recall");
    assert!(
        recall.hits.iter().any(|h| h.memory.uuid == out.uuid),
        "recall should return the remembered uuid"
    );

    // --- delete (hard) then verify gone ---
    nova.delete_memory(out.uuid).await.expect("delete_memory");
    let err = nova.get_memory(out.uuid).await.expect_err("get after delete should fail");
    assert_eq!(err.code(), ErrorCode::NotFound);
}

#[tokio::test]
async fn stats_reflect_counts() {
    let nova = EmbeddedNova::open(tmp_db("stats")).await.expect("open");

    let s0 = nova.stats().await.expect("stats before");
    assert_eq!(s0.memory_active, 0);
    assert_eq!(s0.memory_total, 0);

    let out = nova
        .remember(http_client::RememberRequest {
            content: "stats counting note".into(),
            tags: vec!["counted".into()],
            ..Default::default()
        })
        .await
        .expect("remember");

    let s1 = nova.stats().await.expect("stats after");
    assert_eq!(s1.memory_active, 1);
    assert_eq!(s1.memory_total, 1);
    assert!(s1.tag_count >= 1);
    let _ = out;
}

#[tokio::test]
async fn graph_extract_and_link_works() {
    // `EmbeddedNova::open` wires a NoopExtractor, so for a real extraction to
    // happen we build a graph service with the RegexWikiExtractor and hand it
    // in via `from_services`.
    let db_path = tmp_db("graph");
    let db = Database::open(StorageConfig { db_path, ..Default::default() })
        .await
        .expect("open db");
    let embedding: SharedEmbeddingProvider = Arc::new(MockEmbeddingProvider::new(64));
    let memory = MemoryService::new(db.clone(), embedding);
    let graph = GraphService::with_parts(db.clone(), Arc::new(RegexWikiExtractor::new()));
    let nova = EmbeddedNova::from_services(db, memory, graph);

    let out = nova
        .extract_and_link(http_client::ExtractAndLinkRequest {
            text: "[[Alice]] works with [[Bob]]".into(),
            opts: GraphExtractOpts { enabled: true, ..Default::default() },
        })
        .await
        .expect("extract_and_link");
    assert!(out.entities_upserted >= 2, "expected >= 2 entities, got {}", out.entities_upserted);
    assert!(out.entities.iter().any(|(e, _)| e.name == "Alice"));
    assert!(out.entities.iter().any(|(e, _)| e.name == "Bob"));
}

#[tokio::test]
async fn health_returns_ok() {
    let nova = EmbeddedNova::open(tmp_db("health")).await.expect("open");
    let h = nova.health().await.expect("health");
    assert_eq!(h.status, "ok");
    assert_eq!(h.version, yq_nova_core::VERSION);
    let _ = Uuid::new_v4(); // exercise the re-exported Uuid import
}