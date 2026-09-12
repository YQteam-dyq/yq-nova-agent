use serde::{Deserialize, Serialize};

use super::{MemoryService, ops_recall::passes_filter};
use crate::{
    error::NovaResult,
    storage::{
        MemoryFilter, MemoryStatus,
        memory::{MemoryRecord, MemoryRepository},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInput {
    pub filter: Option<MemoryFilter>,
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

impl Default for ExportInput {
    fn default() -> Self {
        Self {
            filter: None,
            limit: 500,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOutput {
    pub count: usize,
    pub offset: u32,
    pub items: Vec<MemoryRecord>,
}

const fn default_limit() -> u32 {
    500
}

pub async fn export_memories(svc: &MemoryService, input: ExportInput) -> NovaResult<ExportOutput> {
    let limit = (input.limit.min(10000)) as usize;
    let offset = input.offset as usize;

    let mut filter = input.filter.unwrap_or_default();
    if filter.status_in.is_none() {
        filter.status_in = Some(vec![MemoryStatus::Active, MemoryStatus::Archived]);
    }

    let mut items = svc.memory_repo.list(&svc.database, &filter, limit, offset).await?;
    items.retain(|r| passes_filter(r, &filter));
    let count = items.len();

    Ok(ExportOutput {
        count,
        offset: input.offset,
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

    async fn temp_svc() -> crate::memory::MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-export-{}", Uuid::new_v4()));
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
    async fn export_all_memories() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "memory one",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "memory two",
                importance: 0.8,
                ..Default::default()
            })
            .await
            .unwrap();

        let out = export_memories(&svc, ExportInput::default()).await.unwrap();
        assert_eq!(out.count, 2);
        let uuids: std::collections::HashSet<_> = out.items.iter().map(|m| m.uuid).collect();
        assert!(uuids.contains(&a.uuid));
        assert!(uuids.contains(&b.uuid));
    }

    #[tokio::test]
    async fn export_filter_by_source() {
        let svc = temp_svc().await;
        let _user = svc
            .remember(RememberInput {
                content: "user note",
                source: MemorySource::User,
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let _agent = svc
            .remember(RememberInput {
                content: "agent thought",
                source: MemorySource::Agent,
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();

        let filter = MemoryFilter {
            source_in: Some(vec![MemorySource::User]),
            ..Default::default()
        };
        let out = export_memories(
            &svc,
            ExportInput {
                filter: Some(filter),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(out.count, 1);
        assert_eq!(out.items[0].source, MemorySource::User);
    }

    #[tokio::test]
    async fn export_pagination() {
        let svc = temp_svc().await;
        for i in 0..5 {
            svc.remember(RememberInput {
                content: &format!("memory-{i}"),
                importance: 0.1,
                ..Default::default()
            })
            .await
            .unwrap();
        }

        let page1 = export_memories(
            &svc,
            ExportInput {
                limit: 2,
                offset: 0,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(page1.count, 2);
        assert_eq!(page1.offset, 0);

        let page2 = export_memories(
            &svc,
            ExportInput {
                limit: 2,
                offset: 2,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(page2.count, 2);
        assert_eq!(page2.offset, 2);

        let page3 = export_memories(
            &svc,
            ExportInput {
                limit: 2,
                offset: 4,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(page3.count, 1);
        assert_eq!(page3.offset, 4);
    }
}
