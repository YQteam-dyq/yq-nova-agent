//! Task 5: performance benchmarks (criterion).
//!
//! Three groups:
//!   a. `vector_knn`   — KNN linear-scan scaling vs #vectors (SqliteVectorStore)
//!   b. `graph_bfs`    — GraphService BFS traversal latency vs #nodes
//!   c. `embedding`    — MockEmbeddingProvider::embed_batch throughput
//!
//! criterion is synchronous, so each benchmark closure drives the async
//! calls through a single current-thread tokio runtime via `block_on`.

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tokio::runtime::Builder;

use yq_nova_core::config::StorageConfig;
use yq_nova_core::embedding::{EmbeddingProvider, MockEmbeddingProvider};
use yq_nova_core::graph::{GraphService, TraverseOpts};
use yq_nova_core::storage::{
    Database,
    entity::{EntityRepository, SqliteEntityRepository, UpsertEntityInput},
    memory::{InsertMemoryInput, MemoryRepository, SqliteMemoryRepository},
    relation::{InsertRelationInput, RelationRepository, SqliteRelationRepository},
    vector::{SqliteVectorStore, VectorStore},
};
use yq_nova_core::Uuid;

/// A single current-thread runtime reused across all async calls in a bench.
fn runtime() -> tokio::runtime::Runtime {
    Builder::new_current_thread().enable_all().build().expect("build tokio runtime")
}

/// Open a fresh temp-file SQLite DB (applies migrations), mirroring the
/// `temp_db()` pattern used in the storage tests.
async fn temp_db(tag: &str) -> Database {
    let dir = std::env::temp_dir().join(format!("yq-nova-bench-{tag}-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = StorageConfig {
        db_path: dir.join("bench.db"),
        pool_max_connections: 2,
        pool_min_connections: 0,
        ..StorageConfig::default()
    };
    Database::open(cfg).await.expect("open temp db")
}

// -----------------------------------------------------------------------------
// a. Vector KNN scaling
// -----------------------------------------------------------------------------

fn bench_vector_knn(c: &mut Criterion) {
    const DIMS: usize = 64;
    const K: usize = 10;
    let rt = runtime();
    let mut group = c.benchmark_group("vector_knn_linear_scan");

    for n in [1_000usize, 10_000, 100_000] {
        // Build + populate the store once per scale so setup cost is excluded
        // from the timed measurement.
        let store = rt.block_on(async {
            let db = temp_db("knn").await;
            let mem_repo = SqliteMemoryRepository::new();
            let store = SqliteVectorStore::with_db(&db, DIMS);
            for i in 0..n {
                let mem_uuid = mem_repo
                    .insert(
                        &db,
                        InsertMemoryInput { content: &format!("memory-{i}"), ..Default::default() },
                    )
                    .await
                    .expect("insert memory")
                    .uuid();
                let v = yq_nova_core::embedding::deterministic_pseudo_embedding(
                    &format!("memory-{i}"),
                    DIMS,
                );
                store
                    .insert_vector(mem_uuid, "bench", "mock-64d", &v)
                    .await
                    .expect("insert vector");
            }
            store
        });

        let query = yq_nova_core::embedding::deterministic_pseudo_embedding("query", DIMS);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let hits = rt.block_on(store.knn_search(black_box(&query), K, -1.0));
                black_box(hits)
            });
        });
    }
    group.finish();
}

// -----------------------------------------------------------------------------
// b. Graph BFS traversal
// -----------------------------------------------------------------------------

fn bench_graph_bfs(c: &mut Criterion) {
    let rt = runtime();
    let mut group = c.benchmark_group("graph_bfs_traverse");

    for n in [100usize, 300, 1000] {
        // Build a graph where each node connects to the next few nodes, so a
        // BFS rooted at node 0 fans out broadly. Setup is done once per scale.
        let (svc, start) = rt.block_on(async {
            let db = temp_db("bfs").await;
            let svc = GraphService::new(db);
            let er = SqliteEntityRepository::new();
            let rr = SqliteRelationRepository::new();
            let mut uuids: Vec<Uuid> = Vec::with_capacity(n);
            for i in 0..n {
                let u = er
                    .upsert(
                        &svc.database,
                        UpsertEntityInput {
                            name: &format!("node-{i}"),
                            r#type: "bench",
                            description: None,
                            metadata: None,
                        },
                    )
                    .await
                    .expect("upsert entity")
                    .uuid();
                uuids.push(u);
            }
            for i in 0..n {
                for j in (i + 1)..(i + 5).min(n) {
                    let (src, tgt) = (uuids[i], uuids[j]);
                    rr.insert(
                        &svc.database,
                        InsertRelationInput {
                            source_uuid: src,
                            target_uuid: tgt,
                            predicate: "links_to",
                            confidence: 0.9,
                            memory_uuid: None,
                            metadata: None,
                            idempotent: true,
                        },
                    )
                    .await
                    .expect("insert relation");
                }
            }
            (svc, uuids[0])
        });

        let opts = TraverseOpts {
            max_depth: 8,
            max_nodes: n,
            predicate_whitelist: Vec::new(),
            min_confidence: 0.0,
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let nodes =
                    rt.block_on(svc.traverse_graph(black_box(start), opts.clone()));
                black_box(nodes)
            });
        });
    }
    group.finish();
}

// -----------------------------------------------------------------------------
// c. Embedding throughput (mock provider)
// -----------------------------------------------------------------------------

fn bench_embedding(c: &mut Criterion) {
    let rt = runtime();
    let provider = Arc::new(MockEmbeddingProvider::new(64));
    let mut group = c.benchmark_group("embedding_batch_throughput");

    for n in [100usize, 1000] {
        // Build the text corpus once per scale.
        let texts: Vec<String> = (0..n)
            .map(|i| {
                format!(
                    "This is a synthetic text chunk number {i} used to exercise the embedding \
                     provider batch path without any network dependency."
                )
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || refs.clone(),
                |batch| {
                    let out = rt.block_on(provider.embed_batch(black_box(&batch)));
                    black_box(out)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    name = perf;
    config = Criterion::default().sample_size(10).warm_up_time(std::time::Duration::from_millis(300));
    targets = bench_vector_knn, bench_graph_bfs, bench_embedding
);
criterion_main!(perf);