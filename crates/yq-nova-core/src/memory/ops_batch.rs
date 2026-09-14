use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{MemoryService, chunk::ChunkOptions, ops_remember::RememberInput};
use crate::{
    error::{NovaError, NovaResult},
    storage::MemorySource,
};

pub const MAX_BATCH_ITEMS: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchRememberItem {
    pub content: String,
    pub source: MemorySource,
    pub importance: f32,
    pub metadata: Option<serde_json::Value>,
    pub expires_at: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
    pub embed: bool,
    pub extract_graph: bool,
    pub chunk_options: Option<ChunkOptions>,
}

impl Default for BatchRememberItem {
    fn default() -> Self {
        Self {
            content: String::new(),
            source: MemorySource::Agent,
            importance: 0.5,
            metadata: None,
            expires_at: None,
            tags: Vec::new(),
            embed: true,
            extract_graph: false,
            chunk_options: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BatchRememberInput {
    pub items: Vec<BatchRememberItem>,
    pub continue_on_error: bool,
}

impl Default for BatchRememberInput {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            continue_on_error: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRememberResult {
    pub index: usize,
    pub uuid: Option<Uuid>,
    pub duplicate: bool,
    pub embedding_stored: bool,
    pub tags: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRememberOutput {
    pub received: usize,
    pub succeeded: usize,
    pub duplicates: usize,
    pub failed: usize,
    pub results: Vec<BatchRememberResult>,
}

pub async fn remember_batch(
    svc: &MemoryService,
    input: BatchRememberInput,
) -> NovaResult<BatchRememberOutput> {
    if input.items.is_empty() {
        return Err(NovaError::validation("batch: items must not be empty"));
    }
    if input.items.len() > MAX_BATCH_ITEMS {
        return Err(NovaError::validation(format!(
            "batch: items must not exceed {MAX_BATCH_ITEMS}, got {}",
            input.items.len()
        )));
    }

    let received = input.items.len();
    let mut succeeded = 0usize;
    let mut duplicates = 0usize;
    let mut failed = 0usize;
    let mut results = Vec::with_capacity(received);

    for (index, item) in input.items.iter().enumerate() {
        let outcome = svc
            .remember(RememberInput {
                content: &item.content,
                source: item.source,
                importance: item.importance,
                metadata: item.metadata.as_ref(),
                expires_at: item.expires_at,
                tags: &item.tags,
                embed: item.embed,
                extract_graph: item.extract_graph,
                chunk_options: item.chunk_options.clone(),
                dedup: None,
            })
            .await;

        match outcome {
            Ok(out) => {
                if out.duplicate {
                    duplicates += 1;
                } else {
                    succeeded += 1;
                }
                results.push(BatchRememberResult {
                    index,
                    uuid: Some(out.uuid),
                    duplicate: out.duplicate,
                    embedding_stored: out.embedding_stored,
                    tags: out.tags,
                    error: None,
                });
            },
            Err(e) => {
                failed += 1;
                let message = e.to_string();
                results.push(BatchRememberResult {
                    index,
                    uuid: None,
                    duplicate: false,
                    embedding_stored: false,
                    tags: Vec::new(),
                    error: Some(message.clone()),
                });
                if !input.continue_on_error {
                    for skipped_index in (index + 1)..received {
                        failed += 1;
                        results.push(BatchRememberResult {
                            index: skipped_index,
                            uuid: None,
                            duplicate: false,
                            embedding_stored: false,
                            tags: Vec::new(),
                            error: Some(format!("skipped after earlier failure: {message}")),
                        });
                    }
                    break;
                }
            },
        }
    }

    Ok(BatchRememberOutput {
        received,
        succeeded,
        duplicates,
        failed,
        results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Uuid, config::StorageConfig, memory::ops_remember::service_for_tests, storage::Database,
    };

    async fn temp_svc() -> MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-batch-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        let db = Database::open(cfg).await.unwrap();
        service_for_tests(db, 8, None)
    }

    fn item(content: &str, importance: f32, tag: &str) -> BatchRememberItem {
        BatchRememberItem {
            content: content.to_string(),
            importance,
            tags: vec![tag.to_string()],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn batch_stores_every_item_and_reports_uuids() {
        let svc = temp_svc().await;
        let out = remember_batch(
            &svc,
            BatchRememberInput {
                items: vec![
                    item("batch entry one", 0.4, "batch"),
                    item("batch entry two", 0.7, "batch"),
                    item("batch entry three", 0.9, "batch"),
                ],
                continue_on_error: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(out.received, 3);
        assert_eq!(out.succeeded, 3);
        assert_eq!(out.duplicates, 0);
        assert_eq!(out.failed, 0);
        assert_eq!(out.results.len(), 3);

        let mut uuids = std::collections::HashSet::new();
        for (position, result) in out.results.iter().enumerate() {
            assert_eq!(result.index, position);
            assert!(result.error.is_none());
            let uuid = result.uuid.expect("uuid for stored item");
            assert!(uuids.insert(uuid), "uuids must be unique");
            let stored = svc.get_memory(uuid).await.unwrap();
            assert_eq!(stored.tags, vec!["batch".to_string()]);
        }
    }

    #[tokio::test]
    async fn batch_marks_duplicates_without_refetching_embeddings() {
        let svc = temp_svc().await;
        let out = remember_batch(
            &svc,
            BatchRememberInput {
                items: vec![item("same content", 0.5, "dup"), item("same content", 0.5, "dup")],
                continue_on_error: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(out.succeeded, 1);
        assert_eq!(out.duplicates, 1);
        assert_eq!(out.results[0].uuid, out.results[1].uuid);
        assert!(!out.results[0].duplicate);
        assert!(out.results[1].duplicate);
        assert!(!out.results[1].embedding_stored);
    }

    #[tokio::test]
    async fn batch_stops_on_first_error_when_configured() {
        let svc = temp_svc().await;
        let items = vec![
            item("   ", 0.5, "bad"),
            item("valid entry one", 0.5, "ok"),
            item("valid entry two", 0.5, "ok"),
        ];

        let strict = remember_batch(
            &svc,
            BatchRememberInput {
                items: items.clone(),
                continue_on_error: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(strict.received, 3);
        assert_eq!(strict.succeeded, 0);
        assert_eq!(strict.duplicates, 0);
        assert_eq!(strict.failed, 3);
        assert_eq!(strict.results.len(), 3);
        assert_eq!(strict.succeeded + strict.duplicates + strict.failed, strict.received);

        let mut seen = vec![false; strict.received];
        for result in &strict.results {
            assert!(result.index < strict.received);
            seen[result.index] = true;
            assert!(result.error.is_some());
        }
        assert!(seen.into_iter().all(|hit| hit), "every index must appear exactly once");

        let lenient = remember_batch(
            &svc,
            BatchRememberInput {
                items,
                continue_on_error: true,
            },
        )
        .await
        .unwrap();
        assert_eq!(lenient.failed, 1);
        assert_eq!(lenient.results.len(), 3);
        assert_eq!(lenient.succeeded + lenient.duplicates + lenient.failed, lenient.received);
    }

    #[tokio::test]
    async fn batch_rejects_empty_and_oversized_payloads() {
        let svc = temp_svc().await;
        let empty = remember_batch(&svc, BatchRememberInput::default()).await.unwrap_err();
        assert_eq!(empty.code(), crate::error::ErrorCode::Validation);

        let oversized = BatchRememberInput {
            items: (0..=MAX_BATCH_ITEMS).map(|_| BatchRememberItem::default()).collect(),
            continue_on_error: true,
        };
        let too_big = remember_batch(&svc, oversized).await.unwrap_err();
        assert_eq!(too_big.code(), crate::error::ErrorCode::Validation);
    }
}
