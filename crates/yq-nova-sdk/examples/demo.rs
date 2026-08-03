//! End-to-end yq-nova demo using `yq-nova-sdk`'s HTTP client.
//!
//! Run in two terminals:
//!   Terminal 1: YQ_NOVA_EMBEDDING__DEFAULT_PROVIDER=mock cargo run -p yq-nova -- serve
//!   Terminal 2: cargo run -p yq-nova-sdk-demo
//!
//! By default it connects to http://127.0.0.1:7999; set YQ_NOVA_URL to override.

use std::time::Duration;

use yq_nova_core::{
    error::NovaResult,
    graph::GraphExtractOpts,
    memory::{ForgetMode, SearchMode, ops_forget},
    storage::{MemoryFilter, MemorySource},
};
use yq_nova_sdk::{
    Uuid,
    http_client::{
        ExtractAndLinkRequest, HttpClient, TraverseRequest, UpsertEntityRequest,
        UpsertRelationRequest,
    },
};

#[tokio::main]
async fn main() -> NovaResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let base_url = std::env::var("YQ_NOVA_URL").unwrap_or_else(|_| "http://127.0.0.1:7999".into());
    let client = HttpClient::with_timeout(base_url, Duration::from_secs(5))?;
    println!("==> connecting to {}", client.base_url());

    // 0. health / stats
    let health = client.health().await?;
    println!(
        "[health]  status={} version={} git_sha={} uptime={}s",
        health.status, health.version, health.git_sha, health.uptime_secs
    );
    let stats = client.stats().await?;
    println!(
        "[stats]   active={active} archived={archived} entities={ent} relations={rel} db={db}b",
        active = stats.memory_active,
        archived = stats.memory_archived,
        ent = stats.entity_count,
        rel = stats.relation_count,
        db = stats.database_size_bytes,
    );

    // 1. remember some facts about a project
    println!("\n==> remember 3 facts");
    let facts = [
        (
            "yq-nova-agent stores memory locally in SQLite with optional embeddings",
            0.85,
            vec!["yq-nova".into(), "rust".into(), "storage".into()],
        ),
        (
            "We use graph extraction with [[WikiLinks]] and #hashtags for light NLP",
            0.70,
            vec!["graph".into(), "nlp".into()],
        ),
        (
            "The HTTP surface exposes /v1/memory/remember /recall /forget and /v1/graph/*",
            0.92,
            vec!["api".into(), "http".into()],
        ),
    ];
    let mut uuids: Vec<Uuid> = Vec::with_capacity(facts.len());
    for (content, importance, tags) in facts {
        let out = client
            .remember_builder()
            .content(content)
            .source(MemorySource::User)
            .importance(importance)
            .tags(tags)
            .embed(true)
            .extract_graph(true)
            .send()
            .await?;
        println!(
            "  remember uuid={uuid} duplicate={dup} emb_store={emb} entities={ents} tags={tags:?}",
            uuid = out.uuid,
            dup = out.duplicate,
            emb = out.embedding_stored,
            ents = out.entities_extracted,
            tags = out.tags,
        );
        uuids.push(out.uuid);
    }

    // 2. recall with a fuzzy query
    println!("\n==> recall 'SQLite memory graph api'");
    let recall = client
        .recall_builder()
        .query("SQLite memory graph api")
        .top_k(5)
        .mode(SearchMode::Semantic)
        .score_threshold(0.0)
        .send()
        .await?;
    println!(
        "  returned {count} hits (total_candidates={total})",
        count = recall.hits.len(),
        total = recall.total_candidates
    );
    for (i, h) in recall.hits.iter().enumerate() {
        let s = &h.memory;
        let sim = h.raw_similarity.unwrap_or(0.0);
        println!(
            "    #{i}: score={score:.3} sim={sim:.3} imp={imp:.2} acc={acc}: {c}",
            score = h.final_score,
            imp = s.importance,
            acc = s.access_count,
            c = s.content.chars().take(70).collect::<String>(),
        );
    }

    // 3. graph: upsert 2 entities + link + BFS traverse
    println!("\n==> graph: Alice --reports_to--> Bob");
    let alice = client
        .upsert_entity(UpsertEntityRequest {
            name: "Alice".into(),
            entity_type: "person".into(),
            description: Some("Principal engineer".into()),
            ..Default::default()
        })
        .await?;
    let bob = client
        .upsert_entity(UpsertEntityRequest {
            name: "Bob".into(),
            entity_type: "person".into(),
            description: Some("Engineering manager".into()),
            ..Default::default()
        })
        .await?;
    let rel = client
        .upsert_relation(UpsertRelationRequest {
            source_uuid: alice.outcome.uuid(),
            target_uuid: bob.outcome.uuid(),
            predicate: "reports_to".into(),
            confidence: 0.95,
            idempotent: true,
            ..Default::default()
        })
        .await?;
    println!(
        "  inserted={ins} updated={upd} predicate={pred}",
        ins = rel.inserted,
        upd = rel.updated,
        pred = rel.relation.predicate
    );

    let nodes = client
        .traverse(TraverseRequest {
            start: alice.outcome.uuid(),
            max_depth: 3,
            max_nodes: 20,
            ..Default::default()
        })
        .await?;
    println!("  BFS from Alice returned {} nodes:", nodes.len());
    for n in &nodes {
        println!(
            "    * {name} ({t}) depth={d}",
            name = n.entity.name,
            t = n.entity.r#type,
            d = n.depth,
        );
    }

    // 4. extract_and_link on ad-hoc text (doesn't go into memory)
    println!("\n==> extract_and_link on text");
    let opts = GraphExtractOpts {
        enabled: true,
        upsert_entities: true,
        create_relations: true,
        min_confidence: 0.2,
    };
    let out = client
        .extract_and_link(ExtractAndLinkRequest {
            text: "Lachlan reviewed the [[SVD]] factorization patch for \
                   #recommender systems and commented on #rustlang's matrix-ops \
                   package."
                .into(),
            opts,
        })
        .await?;
    println!(
        "  entities={ents} upserted={up} relations_created={rel}",
        ents = out.entities.len(),
        up = out.entities_upserted,
        rel = out.relations_created,
    );
    for (cand, _uuid) in &out.entities {
        println!("    * {name} (type={t})", name = cand.name, t = cand.entity_type,);
    }

    // 5. forget via filter: tag='api' → Hard delete
    println!("\n==> forget(tag=api, mode=Hard)");
    let f = MemoryFilter { tags_all: Some(vec!["api".to_string()]), ..Default::default() };
    let forgotten = client
        .forget(ops_forget::ForgetInput {
            target: ops_forget::ForgetTarget::Filter(f),
            mode: ForgetMode::Hard,
            gc_graph: false,
            batch_limit: 100,
        })
        .await?;
    println!(
        "  affected={aff} cascade_embeddings={casc}",
        aff = forgotten.affected_memories,
        casc = forgotten.cascade_embeddings
    );

    // 5b. DELETE the UUID corresponding to the graph-fact memory directly.
    if let Some(uuid) = uuids.get(1).copied() {
        let del = client.delete_memory(uuid).await?;
        println!("  direct DELETE uuid={uuid}: affected={}", del.affected_memories);
    }

    // 6. final stats
    let stats = client.stats().await?;
    println!(
        "\n==> final stats active={active} archived={archived} entities={ent} relations={rel}",
        active = stats.memory_active,
        archived = stats.memory_archived,
        ent = stats.entity_count,
        rel = stats.relation_count,
    );

    Ok(())
}
