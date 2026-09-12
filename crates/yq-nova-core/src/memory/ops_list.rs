use serde::{Deserialize, Serialize};

use super::{MemoryService, ops_recall::passes_filter};
use crate::{
    error::{NovaError, NovaResult},
    storage::{
        MemoryFilter, MemorySortOrder, MemoryStatus,
        memory::{MemoryRecord, MemoryRepository},
    },
};

pub const MAX_LIST_LIMIT: u32 = 500;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ListInput {
    pub filter: MemoryFilter,
    pub limit: u32,
    pub offset: u32,
    pub sort: MemorySortOrder,
}

impl Default for ListInput {
    fn default() -> Self {
        Self {
            filter: MemoryFilter::default(),
            limit: 50,
            offset: 0,
            sort: MemorySortOrder::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListOutput {
    pub total: i64,
    pub count: usize,
    pub limit: u32,
    pub offset: u32,
    pub sort: MemorySortOrder,
    pub items: Vec<MemoryRecord>,
}

pub async fn list_memories(svc: &MemoryService, input: ListInput) -> NovaResult<ListOutput> {
    if input.limit == 0 {
        return Err(NovaError::validation("list: limit must be >= 1"));
    }
    if input.limit > MAX_LIST_LIMIT {
        return Err(NovaError::validation(format!(
            "list: limit must be <= {MAX_LIST_LIMIT}, got {}",
            input.limit
        )));
    }

    let mut filter = input.filter;
    if filter.status_in.is_none() {
        filter.status_in = Some(vec![MemoryStatus::Active, MemoryStatus::Archived]);
    }

    let limit = input.limit as usize;
    let offset = input.offset as usize;

    let total = svc.memory_repo.count(&svc.database, &filter).await?;
    let mut items =
        svc.memory_repo.list_ordered(&svc.database, &filter, limit, offset, input.sort).await?;
    items.retain(|r| passes_filter(r, &filter));

    Ok(ListOutput {
        total,
        count: items.len(),
        limit: input.limit,
        offset: input.offset,
        sort: input.sort,
        items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Uuid,
        config::StorageConfig,
        memory::ops_remember::{RememberInput, service_for_tests},
        storage::{Database, MemorySource},
    };

    async fn temp_svc() -> MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-list-{}", Uuid::new_v4()));
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
        for (content, importance, tag, source) in [
            ("first entry", 0.2, "alpha", MemorySource::Agent),
            ("second entry", 0.9, "beta", MemorySource::User),
            ("third entry", 0.6, "alpha", MemorySource::Tool),
        ] {
            svc.remember(RememberInput {
                content,
                importance,
                source,
                tags: &[tag.to_string()],
                extract_graph: false,
                ..Default::default()
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn list_returns_paginated_items_with_total() {
        let svc = temp_svc().await;
        seed(&svc).await;

        let page1 = list_memories(
            &svc,
            ListInput {
                limit: 2,
                offset: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(page1.total, 3);
        assert_eq!(page1.count, 2);
        assert_eq!(page1.items.len(), 2);

        let page2 = list_memories(
            &svc,
            ListInput {
                limit: 2,
                offset: 2,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(page2.total, 3);
        assert_eq!(page2.count, 1);

        let page1_ids: std::collections::HashSet<_> = page1.items.iter().map(|m| m.uuid).collect();
        let page2_ids: std::collections::HashSet<_> = page2.items.iter().map(|m| m.uuid).collect();
        assert!(page1_ids.is_disjoint(&page2_ids), "pages must not overlap");
    }

    #[tokio::test]
    async fn list_supports_importance_and_tag_filters() {
        let svc = temp_svc().await;
        seed(&svc).await;

        let by_tag = list_memories(
            &svc,
            ListInput {
                filter: MemoryFilter {
                    tags_all: Some(vec!["alpha".into()]),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(by_tag.total, 2);
        assert!(by_tag.items.iter().all(|m| m.tags.contains(&"alpha".to_string())));

        let by_importance = list_memories(
            &svc,
            ListInput {
                filter: MemoryFilter {
                    importance_min: Some(0.5),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(by_importance.total, 2);
    }

    #[tokio::test]
    async fn list_sorts_by_importance_descending() {
        let svc = temp_svc().await;
        seed(&svc).await;

        let out = list_memories(
            &svc,
            ListInput {
                sort: MemorySortOrder::ImportanceDesc,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let scores: Vec<f32> = out.items.iter().map(|m| m.importance).collect();
        assert_eq!(scores, vec![0.9, 0.6, 0.2]);
    }

    #[tokio::test]
    async fn list_rejects_invalid_limit() {
        let svc = temp_svc().await;
        let zero = list_memories(
            &svc,
            ListInput {
                limit: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert_eq!(zero.code(), crate::error::ErrorCode::Validation);

        let too_big = list_memories(
            &svc,
            ListInput {
                limit: MAX_LIST_LIMIT + 1,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert_eq!(too_big.code(), crate::error::ErrorCode::Validation);
    }
}
