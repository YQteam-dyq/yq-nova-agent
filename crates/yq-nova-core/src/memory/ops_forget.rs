use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::MemoryService;
use crate::{
    error::{ErrorCode, NovaError, NovaResult},
    storage::{
        MemoryFilter, MemoryStatus, entity::EntityRepository, memory::MemoryRepository,
        relation::RelationRepository, vector::VectorStore,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ForgetMode {
    #[default]
    Soft,

    Hard,

    Archive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ForgetTarget {
    One(Uuid),
    Filter(MemoryFilter),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ForgetInput {
    pub target: ForgetTarget,
    pub mode: ForgetMode,

    pub gc_graph: bool,

    pub batch_limit: usize,
}

impl Default for ForgetInput {
    fn default() -> Self {
        Self {
            target: ForgetTarget::Filter(MemoryFilter::default()),
            mode: ForgetMode::Soft,
            gc_graph: false,
            batch_limit: 500,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForgetOutput {
    pub affected_memories: usize,

    pub cascade_embeddings: usize,

    pub gc_entities: usize,

    pub gc_relations: usize,

    pub relations_cleaned: usize,

    pub mode: ForgetMode,
}

pub async fn forget(svc: &MemoryService, input: ForgetInput) -> NovaResult<ForgetOutput> {
    let uuids: Vec<Uuid> = match input.target {
        ForgetTarget::One(uuid) => match svc.memory_repo.get_by_uuid(&svc.database, uuid).await {
            Ok(_) => vec![uuid],
            Err(e) if matches!(e.code(), crate::error::ErrorCode::NotFound) => {
                return Err(e);
            },
            Err(e) => return Err(e),
        },
        ForgetTarget::Filter(f) => {
            let limit = input.batch_limit.min(10_000);
            if limit == 0 {
                return Err(NovaError::validation("forget: batch_limit must be >= 1"));
            }

            let rows = svc.memory_repo.list(&svc.database, &f, limit + 1, 0).await?;
            let capped = rows.len().min(limit);
            rows.into_iter().take(capped).map(|r| r.uuid).collect()
        },
    };

    if uuids.is_empty() {
        return Ok(ForgetOutput {
            mode: input.mode,
            ..Default::default()
        });
    }

    let mut affected = 0usize;
    let mut relations_cleaned = 0usize;
    for u in &uuids {
        match input.mode {
            ForgetMode::Soft => {
                svc.memory_repo.update_status(&svc.database, *u, MemoryStatus::Deleted).await?;
            },
            ForgetMode::Archive => {
                svc.memory_repo.update_status(&svc.database, *u, MemoryStatus::Archived).await?;
            },
            ForgetMode::Hard => {
                if input.gc_graph {
                    relations_cleaned +=
                        svc.relation_repo.delete_by_memory(&svc.database, *u).await?;
                }

                svc.vector_store.delete_vector(*u).await.ok();
                svc.memory_repo.delete(&svc.database, *u).await?;
            },
        }
        affected += 1;
    }

    let mut gc_entities = 0usize;
    let mut gc_relations = 0usize;
    if input.gc_graph {
        let (ent, rel) = gc_orphan_entities(svc).await.unwrap_or((0, 0));
        gc_entities = ent;
        gc_relations = rel;
    }

    let cascade_embeddings = match input.mode {
        ForgetMode::Hard => affected,
        _ => 0,
    };

    Ok(ForgetOutput {
        affected_memories: affected,
        cascade_embeddings,
        gc_entities,
        gc_relations,
        relations_cleaned,
        mode: input.mode,
    })
}

async fn gc_orphan_entities(svc: &MemoryService) -> NovaResult<(usize, usize)> {
    use crate::storage::entity::EntityRecord;

    let all: Vec<EntityRecord> =
        svc.entity_repo.list(&svc.database, None, None, i64::MAX as usize, 0).await?;

    let mut deleted_ent = 0usize;
    let deleted_rel = 0usize;
    for e in all {
        if e.r#type != "unknown" {
            continue;
        }
        let out = svc.relation_repo.list_outgoing(&svc.database, e.uuid, None, usize::MAX).await?;
        let inc = svc.relation_repo.list_incoming(&svc.database, e.uuid, None, usize::MAX).await?;
        if out.is_empty() && inc.is_empty() {
            match svc.entity_repo.delete(&svc.database, e.uuid).await {
                Ok(()) => deleted_ent += 1,
                Err(err) if matches!(err.code(), ErrorCode::NotFound) => {},
                Err(err) => return Err(err),
            }
        }
    }

    Ok((deleted_ent, deleted_rel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Uuid,
        config::StorageConfig,
        memory::ops_remember::{RememberInput, service_for_tests},
        storage::{
            Database, MemoryStatus, entity::UpsertEntityInput, relation::InsertRelationInput,
        },
    };

    async fn temp_svc() -> crate::memory::MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-m3-forget-{}", Uuid::new_v4()));
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
    async fn forget_missing_uuid_returns_not_found() {
        let svc = temp_svc().await;
        let bad = Uuid::new_v4();
        let err = svc
            .forget(ForgetInput {
                target: ForgetTarget::One(bad),
                mode: ForgetMode::Soft,
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn forget_soft_sets_deleted_status_keeps_row() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "ephemeral scratch",
                importance: 0.05,
                ..Default::default()
            })
            .await
            .unwrap();

        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::One(a.uuid),
                mode: ForgetMode::Soft,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(out.affected_memories, 1);
        assert_eq!(out.mode, ForgetMode::Soft);

        let mem = svc.get_memory(a.uuid).await.unwrap();
        assert_eq!(mem.status, MemoryStatus::Deleted);
    }

    #[tokio::test]
    async fn forget_hard_removes_row_completely() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "one off",
                importance: 0.1,
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(svc.get_memory(a.uuid).await.is_ok());

        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::One(a.uuid),
                mode: ForgetMode::Hard,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(out.affected_memories, 1);
        assert_eq!(out.cascade_embeddings, 1);

        let err = svc.get_memory(a.uuid).await.unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn forget_archive_sets_archive_status() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "long-term archived note",
                importance: 0.3,
                ..Default::default()
            })
            .await
            .unwrap();
        svc.forget(ForgetInput {
            target: ForgetTarget::One(a.uuid),
            mode: ForgetMode::Archive,
            ..Default::default()
        })
        .await
        .unwrap();
        let mem = svc.get_memory(a.uuid).await.unwrap();
        assert_eq!(mem.status, MemoryStatus::Archived);
    }

    #[tokio::test]
    async fn forget_filter_batches_respects_limit() {
        let svc = temp_svc().await;
        for i in 0..10 {
            svc.remember(RememberInput {
                content: &format!("memory-{}", i),
                importance: 0.1 + i as f32 * 0.01,
                ..Default::default()
            })
            .await
            .unwrap();
        }

        let f = MemoryFilter {
            importance_min: Some(0.0),
            ..Default::default()
        };
        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::Filter(f.clone()),
                mode: ForgetMode::Soft,
                batch_limit: 3,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(out.affected_memories, 3);
    }

    async fn seed_memory_with_relation(svc: &crate::memory::MemoryService) -> (Uuid, Uuid) {
        let mem = svc
            .remember(RememberInput {
                content: "relation test memory",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let src = svc
            .entity_repo
            .upsert(
                &svc.database,
                UpsertEntityInput {
                    name: "Source",
                    r#type: "test",
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();
        let tgt = svc
            .entity_repo
            .upsert(
                &svc.database,
                UpsertEntityInput {
                    name: "Target",
                    r#type: "test",
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();
        svc.relation_repo
            .insert(
                &svc.database,
                InsertRelationInput {
                    source_uuid: src,
                    target_uuid: tgt,
                    predicate: "related_to",
                    confidence: 1.0,
                    memory_uuid: Some(mem.uuid),
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap();
        (mem.uuid, src)
    }

    #[tokio::test]
    async fn forget_hard_with_gc_graph_cleans_relations() {
        let svc = temp_svc().await;
        let (mem_uuid, src_uuid) = seed_memory_with_relation(&svc).await;

        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::One(mem_uuid),
                mode: ForgetMode::Hard,
                gc_graph: true,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(out.affected_memories, 1);
        assert_eq!(out.relations_cleaned, 1);
        let rels =
            svc.relation_repo.list_outgoing(&svc.database, src_uuid, None, 100).await.unwrap();
        assert!(rels.is_empty(), "relation should have been deleted");
    }

    #[tokio::test]
    async fn forget_soft_does_not_clean_relations() {
        let svc = temp_svc().await;
        let (mem_uuid, _) = seed_memory_with_relation(&svc).await;

        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::One(mem_uuid),
                mode: ForgetMode::Soft,
                gc_graph: true,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(out.affected_memories, 1);
        assert_eq!(out.relations_cleaned, 0);
    }

    #[tokio::test]
    async fn forget_archive_does_not_clean_relations() {
        let svc = temp_svc().await;
        let (mem_uuid, _) = seed_memory_with_relation(&svc).await;

        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::One(mem_uuid),
                mode: ForgetMode::Archive,
                gc_graph: true,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(out.affected_memories, 1);
        assert_eq!(out.relations_cleaned, 0);
    }

    #[tokio::test]
    async fn forget_hard_without_gc_graph_does_not_clean_relations() {
        let svc = temp_svc().await;
        let (mem_uuid, _) = seed_memory_with_relation(&svc).await;

        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::One(mem_uuid),
                mode: ForgetMode::Hard,
                gc_graph: false,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(out.affected_memories, 1);
        assert_eq!(out.relations_cleaned, 0);
    }
}
