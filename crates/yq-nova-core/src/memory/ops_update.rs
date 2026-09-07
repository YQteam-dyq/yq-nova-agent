use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::MemoryService;
use crate::{
    error::{NovaError, NovaResult},
    storage::{
        memory::{MemoryRecord, MemoryRepository, attach_tags, detach_tags, list_tags_of_memory},
        vector::VectorStore,
    },
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateInput {
    pub content: Option<String>,
    pub importance: Option<f32>,
    pub metadata: Option<serde_json::Value>,
    pub tags: Option<Vec<String>>,
    pub expires_at: Option<Option<DateTime<Utc>>>,
}

pub async fn update_memory(
    svc: &MemoryService,
    uuid: Uuid,
    input: UpdateInput,
) -> NovaResult<MemoryRecord> {
    let mut record = svc.memory_repo.get_by_uuid(&svc.database, uuid).await?;

    let mut content_changed = false;
    if let Some(content) = &input.content {
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            return Err(NovaError::validation(
                "update_memory: content must not be empty",
            ));
        }
        if trimmed != record.content {
            content_changed = true;
            record.content = trimmed;
        }
    }

    if let Some(importance) = input.importance {
        if !(0.0..=1.0).contains(&importance) || !importance.is_finite() {
            return Err(NovaError::validation(format!(
                "update_memory: importance must be in [0.0, 1.0], got {}",
                importance
            )));
        }
        record.importance = importance;
    }

    if let Some(metadata) = &input.metadata {
        record.metadata = metadata.clone();
    }

    if let Some(expires_at) = input.expires_at {
        record.expires_at = expires_at;
    }

    if content_changed {
        let content_hash = crate::storage::memory::sha256_hex(&record.content);

        if let Some(conflict_uuid) = svc
            .memory_repo
            .check_content_hash_conflict(&svc.database, &content_hash, uuid)
            .await?
        {
            return Err(NovaError::conflict(format!(
                "content_hash conflict with memory {conflict_uuid}"
            )));
        }

        svc.memory_repo
            .update_content(&svc.database, uuid, &record.content, &content_hash)
            .await?;

        let meta = svc.embedding.meta();
        let vec = svc.embedding.embed_one(&record.content).await?;
        if vec.len() != meta.dims {
            return Err(NovaError::embedding_msg(format!(
                "embed_one returned dims={} expected dims={} for provider {}",
                vec.len(),
                meta.dims,
                meta.provider
            )));
        }
        svc.vector_store
            .insert_vector(uuid, &meta.provider, &meta.model, &vec)
            .await?;

        record.content_hash = content_hash;
    }

    if input.importance.is_some() {
        svc.memory_repo
            .update_importance(&svc.database, uuid, record.importance)
            .await?;
    }

    if input.metadata.is_some() {
        svc.memory_repo
            .update_metadata(&svc.database, uuid, &record.metadata)
            .await?;
    }

    if input.expires_at.is_some() {
        let expires_ts = record.expires_at.map(|t| t.timestamp());
        svc.memory_repo
            .update_expires_at(&svc.database, uuid, expires_ts)
            .await?;
    }

    if let Some(new_tags) = &input.tags {
        let old_tags = list_tags_of_memory(&svc.database.pool, uuid).await?;
        if !old_tags.is_empty() {
            detach_tags(&svc.database.pool, uuid, &old_tags).await?;
        }
        let cleaned: Vec<String> = new_tags.iter().map(|t| t.trim().to_string()).collect();
        if !cleaned.is_empty() {
            attach_tags(&svc.database.pool, uuid, &cleaned).await?;
        }
    }

    let updated = svc.memory_repo.get_by_uuid(&svc.database, uuid).await?;
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uuid;
    use crate::config::StorageConfig;
    use crate::memory::ops_remember::{RememberInput, service_for_tests};
    use crate::storage::Database;

    async fn temp_svc() -> MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-update-{}", Uuid::new_v4()));
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

    #[tokio::test]
    async fn update_partial_fields() {
        let svc = temp_svc().await;
        let out = svc
            .remember(RememberInput {
                content: "original content",
                importance: 0.3,
                tags: &["a".to_string(), "b".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!out.duplicate);

        let updated = svc
            .update(
                out.uuid,
                UpdateInput {
                    importance: Some(0.9),
                    tags: Some(vec!["c".to_string()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.importance, 0.9);
        assert_eq!(updated.content, "original content");
        assert_eq!(updated.tags, vec!["c"]);
    }

    #[tokio::test]
    async fn update_content_recomputes_embedding() {
        let svc = temp_svc().await;
        let out = svc
            .remember(RememberInput {
                content: "old content",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(out.embedding_stored);

        let updated = svc
            .update(
                out.uuid,
                UpdateInput {
                    content: Some("new content".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.content, "new content");
        assert_ne!(updated.content_hash, crate::storage::memory::sha256_hex("old content"));
    }

    #[tokio::test]
    async fn update_content_conflict_409() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "shared text",
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "different text",
                ..Default::default()
            })
            .await
            .unwrap();

        let err = svc
            .update(
                b.uuid,
                UpdateInput {
                    content: Some("shared text".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Conflict);
        let _ = (a, b);
    }

    #[tokio::test]
    async fn update_nonexistent_returns_404() {
        let svc = temp_svc().await;
        let missing = Uuid::new_v4();
        let err = svc
            .update(
                missing,
                UpdateInput {
                    importance: Some(0.5),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn update_clear_expires_at() {
        let svc = temp_svc().await;
        let out = svc
            .remember(RememberInput {
                content: "with expiry",
                expires_at: Some(Utc::now() + chrono::Duration::days(1)),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(out.duplicate == false);

        let mem_before = svc.get_memory(out.uuid).await.unwrap();
        assert!(mem_before.expires_at.is_some());

        let updated = svc
            .update(
                out.uuid,
                UpdateInput {
                    expires_at: Some(None),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(updated.expires_at.is_none());
    }
}