use serde::{Deserialize, Serialize};

use super::MemoryService;
use crate::{
    error::{NovaError, NovaResult},
    storage::{TagRecord, TagRepository},
};

pub const MAX_TAG_LIMIT: u32 = 500;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TagListInput {
    pub limit: u32,
    pub offset: u32,
}

impl Default for TagListInput {
    fn default() -> Self {
        Self {
            limit: 100,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagListOutput {
    pub total: i64,
    pub count: usize,
    pub limit: u32,
    pub offset: u32,
    pub items: Vec<TagRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRenameInput {
    pub name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRenameOutput {
    pub name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagDeleteInput {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagDeleteOutput {
    pub name: String,
    pub deleted: bool,
    pub affected_memories: i64,
}

pub async fn list_tags(svc: &MemoryService, input: TagListInput) -> NovaResult<TagListOutput> {
    if input.limit == 0 {
        return Err(NovaError::validation("tags: limit must be >= 1"));
    }
    if input.limit > MAX_TAG_LIMIT {
        return Err(NovaError::validation(format!(
            "tags: limit must be <= {MAX_TAG_LIMIT}, got {}",
            input.limit
        )));
    }
    let total = svc.tag_repo.count_all_tags(&svc.database).await?;
    let limit = input.limit as usize;
    let offset = input.offset as usize;
    let items = svc.tag_repo.list_all_tags(&svc.database, limit, offset).await?;
    Ok(TagListOutput {
        total,
        count: items.len(),
        limit: input.limit,
        offset: input.offset,
        items,
    })
}

pub async fn rename_tag(svc: &MemoryService, input: TagRenameInput) -> NovaResult<TagRenameOutput> {
    let name = input.name.trim();
    let new_name = input.new_name.trim();
    if name.is_empty() {
        return Err(NovaError::validation("tags: name must not be empty"));
    }
    if new_name.is_empty() {
        return Err(NovaError::validation("tags: new_name must not be empty"));
    }
    svc.tag_repo.rename_tag(&svc.database, name, new_name).await?;
    Ok(TagRenameOutput {
        name: name.to_string(),
        new_name: new_name.to_string(),
    })
}

pub async fn delete_tag(svc: &MemoryService, input: TagDeleteInput) -> NovaResult<TagDeleteOutput> {
    let name = input.name.trim();
    if name.is_empty() {
        return Err(NovaError::validation("tags: name must not be empty"));
    }
    let affected_memories = svc.tag_repo.delete_tag(&svc.database, name).await?;
    Ok(TagDeleteOutput {
        name: name.to_string(),
        deleted: true,
        affected_memories,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Uuid,
        config::StorageConfig,
        memory::ops_remember::{RememberInput, service_for_tests},
        storage::Database,
    };

    async fn temp_svc() -> MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-tagop-{}", Uuid::new_v4()));
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

    async fn seed(svc: &MemoryService) {
        for (content, tags) in [
            ("note one", vec!["legacy".to_string(), "keep".to_string()]),
            ("note two", vec!["legacy".to_string()]),
            ("note three", vec!["keep".to_string()]),
        ] {
            svc.remember(RememberInput {
                content,
                tags: &tags,
                extract_graph: false,
                ..Default::default()
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn list_tags_reports_counts_and_totals() {
        let svc = temp_svc().await;
        seed(&svc).await;

        let out = list_tags(&svc, TagListInput::default()).await.unwrap();
        assert_eq!(out.total, 2);
        assert_eq!(out.count, 2);

        let by_name: std::collections::HashMap<String, i64> =
            out.items.iter().map(|t| (t.name.clone(), t.memory_count)).collect();
        assert_eq!(by_name.get("legacy"), Some(&2));
        assert_eq!(by_name.get("keep"), Some(&2));

        let limited = list_tags(
            &svc,
            TagListInput {
                limit: 1,
                offset: 0,
            },
        )
        .await
        .unwrap();
        assert_eq!(limited.total, 2);
        assert_eq!(limited.count, 1);

        let bad = list_tags(
            &svc,
            TagListInput {
                limit: 0,
                offset: 0,
            },
        )
        .await
        .unwrap_err();
        assert_eq!(bad.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn rename_tag_keeps_memory_associations() {
        let svc = temp_svc().await;
        seed(&svc).await;

        let out = rename_tag(
            &svc,
            TagRenameInput {
                name: "legacy".into(),
                new_name: "archive".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(out.new_name, "archive");

        let listed = list_tags(&svc, TagListInput::default()).await.unwrap();
        let by_name: std::collections::HashMap<String, i64> =
            listed.items.iter().map(|t| (t.name.clone(), t.memory_count)).collect();
        assert_eq!(by_name.get("archive"), Some(&2));
        assert!(!by_name.contains_key("legacy"));

        let conflict = rename_tag(
            &svc,
            TagRenameInput {
                name: "keep".into(),
                new_name: "archive".into(),
            },
        )
        .await
        .unwrap_err();
        assert_eq!(conflict.code(), crate::error::ErrorCode::Conflict);

        let missing = rename_tag(
            &svc,
            TagRenameInput {
                name: "absent".into(),
                new_name: "other".into(),
            },
        )
        .await
        .unwrap_err();
        assert_eq!(missing.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn delete_tag_detaches_memories_and_reports_affected() {
        let svc = temp_svc().await;
        seed(&svc).await;

        let out = delete_tag(
            &svc,
            TagDeleteInput {
                name: "legacy".into(),
            },
        )
        .await
        .unwrap();
        assert!(out.deleted);
        assert_eq!(out.affected_memories, 2);

        let listed = list_tags(&svc, TagListInput::default()).await.unwrap();
        assert_eq!(listed.total, 1);

        let missing = delete_tag(
            &svc,
            TagDeleteInput {
                name: "legacy".into(),
            },
        )
        .await
        .unwrap_err();
        assert_eq!(missing.code(), crate::error::ErrorCode::NotFound);
    }
}
