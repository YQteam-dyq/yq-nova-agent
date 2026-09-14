use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use super::{
    GraphExtractOpts, GraphService,
    extractor::EntityExtractor,
    policy::{ExtractionPolicy, FilteringExtractor},
};
use crate::{
    error::{NovaError, NovaResult},
    storage::{Database, MemoryFilter, MemoryRepository, MemorySortOrder, SqliteMemoryRepository},
};

const PAGE_SIZE: usize = 500;
const MAX_ERROR_LOGS: usize = 20;
const DEFAULT_MAX_RETAINED: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchExtractOptions {
    pub filter: MemoryFilter,
    pub limit: usize,
    pub min_confidence: f32,
    pub entity_type_whitelist: Vec<String>,
    pub relation_type_whitelist: Vec<String>,
    pub upsert_entities: bool,
    pub create_relations: bool,
}

impl Default for BatchExtractOptions {
    fn default() -> Self {
        Self {
            filter: MemoryFilter::default(),
            limit: 1000,
            min_confidence: 0.0,
            entity_type_whitelist: Vec::new(),
            relation_type_whitelist: Vec::new(),
            upsert_entities: true,
            create_relations: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchExtractSummary {
    pub scanned: u64,
    pub extracted: u64,
    pub entities_upserted: u64,
    pub relations_created: u64,
    pub errors: u64,
    pub recent_errors: Vec<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum BatchJobState {
    Queued { enqueued_at: DateTime<Utc> },
    Running { started_at: DateTime<Utc> },
    Done { summary: BatchExtractSummary },
    Failed { error: String, failed_at: DateTime<Utc> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJobView {
    pub job_id: Uuid,
    #[serde(flatten)]
    pub state: BatchJobState,
}

pub async fn run_batch_extract(
    db: &Database,
    extractor: Arc<dyn EntityExtractor>,
    options: &BatchExtractOptions,
) -> NovaResult<BatchExtractSummary> {
    let started_at = Utc::now();
    let started_inst = std::time::Instant::now();
    let policy = ExtractionPolicy {
        min_confidence: options.min_confidence,
        entity_type_whitelist: options.entity_type_whitelist.clone(),
        relation_type_whitelist: options.relation_type_whitelist.clone(),
        ..Default::default()
    };
    let filtered: Arc<dyn EntityExtractor> = Arc::new(FilteringExtractor::new(extractor, policy));
    let graph = GraphService::with_parts(db.clone(), filtered);
    let graph_opts = GraphExtractOpts {
        enabled: true,
        upsert_entities: options.upsert_entities,
        create_relations: options.create_relations,
        min_confidence: options.min_confidence,
    };
    let repo = SqliteMemoryRepository::new();
    let mut summary = BatchExtractSummary {
        started_at: Some(started_at),
        ..Default::default()
    };
    let mut offset = 0usize;
    loop {
        if options.limit > 0 && summary.scanned >= options.limit as u64 {
            break;
        }
        let remaining = if options.limit > 0 {
            (options.limit as u64 - summary.scanned) as usize
        } else {
            PAGE_SIZE
        };
        let take = remaining.min(PAGE_SIZE);
        let rows = repo
            .list_ordered(db, &options.filter, take, offset, MemorySortOrder::CreatedAsc)
            .await?;
        if rows.is_empty() {
            break;
        }
        for memory in &rows {
            summary.scanned += 1;
            match graph.extract_and_link(&memory.content, &graph_opts).await {
                Ok(result) => {
                    summary.extracted += 1;
                    summary.entities_upserted += result.entities_upserted as u64;
                    summary.relations_created += result.relations_created as u64;
                },
                Err(error) => {
                    summary.errors += 1;
                    if summary.recent_errors.len() < MAX_ERROR_LOGS {
                        let mut detail = error.to_string();
                        if detail.len() > 300 {
                            detail.truncate(300);
                            detail.push_str("...");
                        }
                        summary.recent_errors.push(detail);
                    }
                },
            }
        }
        offset += rows.len();
        if rows.len() < take {
            break;
        }
    }
    summary.finished_at = Some(Utc::now());
    summary.duration_ms = started_inst.elapsed().as_millis() as u64;
    Ok(summary)
}

enum BatchJobMessage {
    Run(Uuid, Box<BatchExtractOptions>),
    Shutdown,
}

#[derive(Clone)]
pub struct BatchExtractQueue {
    tx: mpsc::Sender<BatchJobMessage>,
    registry: Arc<Mutex<Vec<(Uuid, BatchJobState)>>>,
    max_retained: usize,
    worker: Arc<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for BatchExtractQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchExtractQueue").finish_non_exhaustive()
    }
}

impl BatchExtractQueue {
    pub fn spawn(db: Database, extractor: Arc<dyn EntityExtractor>, capacity: usize) -> Self {
        Self::spawn_with_retention(db, extractor, capacity, DEFAULT_MAX_RETAINED)
    }

    pub fn spawn_with_retention(
        db: Database,
        extractor: Arc<dyn EntityExtractor>,
        capacity: usize,
        max_retained: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel(capacity.max(1));
        let registry = Arc::new(Mutex::new(Vec::new()));
        let worker_registry = registry.clone();
        let worker = tokio::spawn(worker_loop(rx, db, extractor, worker_registry, max_retained));
        Self {
            tx,
            registry,
            max_retained,
            worker: Arc::new(worker),
        }
    }

    pub async fn enqueue(&self, options: BatchExtractOptions) -> NovaResult<Uuid> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        {
            let mut guard = self
                .registry
                .lock()
                .map_err(|_| NovaError::internal("batch queue registry poisoned"))?;
            prune_registry(&mut guard, self.max_retained);
            guard.push((
                id,
                BatchJobState::Queued {
                    enqueued_at: now,
                },
            ));
        }
        self.tx
            .send(BatchJobMessage::Run(id, Box::new(options)))
            .await
            .map_err(|_| NovaError::internal("batch queue worker has exited"))?;
        Ok(id)
    }

    pub fn get(&self, id: Uuid) -> Option<BatchJobState> {
        let guard = self.registry.lock().ok()?;
        guard.iter().find(|(job_id, _)| *job_id == id).map(|(_, state)| state.clone())
    }

    pub fn list(&self) -> Vec<BatchJobView> {
        self.registry
            .lock()
            .map(|guard| {
                guard
                    .iter()
                    .map(|(job_id, state)| BatchJobView {
                        job_id: *job_id,
                        state: state.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.registry.lock().map(|guard| guard.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub async fn shutdown(&self) {
        let _ = self.tx.send(BatchJobMessage::Shutdown).await;
    }

    pub fn abort(&self) {
        self.worker.abort();
    }
}

async fn worker_loop(
    mut rx: mpsc::Receiver<BatchJobMessage>,
    db: Database,
    extractor: Arc<dyn EntityExtractor>,
    registry: Arc<Mutex<Vec<(Uuid, BatchJobState)>>>,
    max_retained: usize,
) {
    while let Some(message) = rx.recv().await {
        match message {
            BatchJobMessage::Shutdown => break,
            BatchJobMessage::Run(id, options) => {
                if let Ok(mut guard) = registry.lock() {
                    if let Some((_, state)) = guard.iter_mut().find(|(job_id, _)| *job_id == id) {
                        *state = BatchJobState::Running {
                            started_at: Utc::now(),
                        };
                    }
                }
                match run_batch_extract(&db, extractor.clone(), &options).await {
                    Ok(summary) => {
                        if let Ok(mut guard) = registry.lock() {
                            if let Some((_, state)) =
                                guard.iter_mut().find(|(job_id, _)| *job_id == id)
                            {
                                *state = BatchJobState::Done {
                                    summary,
                                };
                            }
                            prune_registry(&mut guard, max_retained);
                        }
                    },
                    Err(error) => {
                        if let Ok(mut guard) = registry.lock() {
                            if let Some((_, state)) =
                                guard.iter_mut().find(|(job_id, _)| *job_id == id)
                            {
                                *state = BatchJobState::Failed {
                                    error: error.to_string(),
                                    failed_at: Utc::now(),
                                };
                            }
                            prune_registry(&mut guard, max_retained);
                        }
                    },
                }
            },
        }
    }
}

fn prune_registry(entries: &mut Vec<(Uuid, BatchJobState)>, max_retained: usize) {
    if max_retained == 0 || entries.len() <= max_retained {
        return;
    }
    let mut overflow = entries.len() - max_retained;
    let mut i = 0;
    while i < entries.len() && overflow > 0 {
        if matches!(entries[i].1, BatchJobState::Done { .. } | BatchJobState::Failed { .. }) {
            entries.remove(i);
            overflow -= 1;
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{
        config::StorageConfig,
        graph::extractor::RegexWikiExtractor,
        storage::{InsertMemoryInput, MemoryRepository, MemoryStatus, SqliteMemoryRepository},
    };

    async fn temp_db_with_memories(contents: &[&str]) -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-b4-batch-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        let db = Database::open(config).await.unwrap();
        let repo = SqliteMemoryRepository::new();
        for content in contents {
            repo.insert(
                &db,
                InsertMemoryInput {
                    content,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        }
        db
    }

    fn regex_extractor() -> Arc<dyn EntityExtractor> {
        Arc::new(RegexWikiExtractor::new())
    }

    #[tokio::test]
    async fn run_batch_extracts_all_memories() {
        let db = temp_db_with_memories(&[
            "[[Alice Smith]] met [[Bob Jones]] at [[Acme Corp]].",
            "[[Trae]] is an AI IDE by [[ByteDance]].",
            "no structured content here",
        ])
        .await;
        let summary = run_batch_extract(&db, regex_extractor(), &BatchExtractOptions::default())
            .await
            .unwrap();
        assert_eq!(summary.scanned, 3);
        assert_eq!(summary.extracted, 3);
        assert_eq!(summary.errors, 0);
        assert!(summary.entities_upserted >= 5, "entities: {}", summary.entities_upserted);
        assert!(summary.relations_created >= 2, "relations: {}", summary.relations_created);
        assert!(summary.duration_ms < 100_000);
        assert!(summary.finished_at.is_some());
    }

    #[tokio::test]
    async fn run_batch_respects_limit() {
        let db =
            temp_db_with_memories(&["[[A]] and [[B]]", "[[C]] and [[D]]", "[[E]] and [[F]]"]).await;
        let options = BatchExtractOptions {
            limit: 2,
            ..Default::default()
        };
        let summary = run_batch_extract(&db, regex_extractor(), &options).await.unwrap();
        assert_eq!(summary.scanned, 2);
        assert_eq!(summary.extracted, 2);
    }

    #[tokio::test]
    async fn run_batch_applies_whitelists_and_confidence() {
        let db =
            temp_db_with_memories(&["[[Rust|lang]] is compiled and [[Cargo|tool]] builds it."])
                .await;
        let options = BatchExtractOptions {
            entity_type_whitelist: vec!["wiki:lang".to_string()],
            min_confidence: 0.5,
            ..Default::default()
        };
        let summary = run_batch_extract(&db, regex_extractor(), &options).await.unwrap();
        assert_eq!(summary.scanned, 1);
        assert!(summary.entities_upserted >= 1, "entities: {}", summary.entities_upserted);
        let entity_rows = sqlx::query_as::<_, (String,)>("SELECT name FROM entities ORDER BY name")
            .fetch_all(&db.pool)
            .await
            .unwrap();
        let names: Vec<String> = entity_rows.into_iter().map(|(n,)| n).collect();
        assert!(names.contains(&"Rust".to_string()), "names: {names:?}");
        assert!(!names.contains(&"Cargo".to_string()), "names: {names:?}");
        assert_eq!(summary.relations_created, 0);
    }

    #[tokio::test]
    async fn run_batch_skips_archived_memories_by_default() {
        let db = temp_db_with_memories(&["[[A]] active", "[[B]] hidden"]).await;
        let repo = SqliteMemoryRepository::new();
        let hidden = repo
            .list(&db, &MemoryFilter::default(), 100, 0)
            .await
            .unwrap()
            .into_iter()
            .find(|m| m.content.contains("hidden"))
            .unwrap();
        repo.update_status(&db, hidden.uuid, MemoryStatus::Archived).await.unwrap();
        let options = BatchExtractOptions {
            filter: MemoryFilter {
                status_in: Some(vec![MemoryStatus::Active]),
                ..Default::default()
            },
            ..Default::default()
        };
        let summary = run_batch_extract(&db, regex_extractor(), &options).await.unwrap();
        assert_eq!(summary.scanned, 1);
    }

    #[tokio::test]
    async fn queue_tracks_job_until_done() {
        let db = temp_db_with_memories(&["[[Alice]] works with [[Bob]] at [[Acme Corp]]."]).await;
        let queue = BatchExtractQueue::spawn(db, regex_extractor(), 16);
        let id = queue.enqueue(BatchExtractOptions::default()).await.unwrap();
        assert!(queue.get(id).is_some());
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let state = loop {
            match queue.get(id) {
                Some(BatchJobState::Done {
                    summary,
                }) => break Some(summary),
                Some(BatchJobState::Failed {
                    error, ..
                }) => {
                    panic!("batch job failed: {error}");
                },
                _ => {
                    assert!(tokio::time::Instant::now() < deadline, "job timed out");
                    tokio::time::sleep(Duration::from_millis(20)).await;
                },
            }
        };
        let summary = state.unwrap();
        assert_eq!(summary.scanned, 1);
        assert!(summary.entities_upserted >= 3);
        assert!(queue.list().iter().any(|view| view.job_id == id));
        queue.abort();
    }
}
