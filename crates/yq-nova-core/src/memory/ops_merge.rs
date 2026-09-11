use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::MemoryService;
use crate::{
    error::{NovaError, NovaResult},
    storage::{MemoryStatus, memory::MemoryRepository},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeInput {
    pub uuids: Vec<Uuid>,
    pub keep_uuid: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeOutput {
    pub kept_uuid: Uuid,
    pub merged: Vec<Uuid>,
    pub remapped_relations: usize,
}

pub async fn merge_memories(svc: &MemoryService, input: MergeInput) -> NovaResult<MergeOutput> {
    let uuids = input.uuids;
    let keep_uuid = input.keep_uuid;

    if uuids.len() < 2 {
        return Err(NovaError::validation("merge: uuids must contain at least 2 entries"));
    }
    if uuids.len() > 50 {
        return Err(NovaError::validation("merge: uuids must contain at most 50 entries"));
    }

    let uuid_set: std::collections::HashSet<Uuid> = uuids.iter().copied().collect();
    if uuid_set.len() != uuids.len() {
        return Err(NovaError::validation("merge: uuids must not contain duplicates"));
    }

    if let Some(keep) = keep_uuid {
        if !uuid_set.contains(&keep) {
            return Err(NovaError::validation("merge: keep_uuid must be one of the uuids"));
        }
    }

    for u in &uuids {
        svc.memory_repo
            .get_by_uuid(&svc.database, *u)
            .await
            .map_err(|_| NovaError::validation(format!("merge: uuid {u} does not exist")))?;
    }

    let records = {
        let mut records = Vec::with_capacity(uuids.len());
        for u in &uuids {
            let rec = svc.memory_repo.get_by_uuid(&svc.database, *u).await?;
            records.push(rec);
        }
        records
    };

    let kept_uuid = match keep_uuid {
        Some(u) => u,
        None => {
            let mut best = &records[0];
            for rec in &records[1..] {
                let older = rec.created_at < best.created_at;
                let same_age = rec.created_at == best.created_at;
                if older || (same_age && rec.importance > best.importance) {
                    best = rec;
                }
            }
            best.uuid
        },
    };

    let merged: Vec<Uuid> = uuids.iter().copied().filter(|u| *u != kept_uuid).collect();

    let merged_strs: Vec<String> = merged.iter().map(|u| u.to_string()).collect();
    let merged_from_val = serde_json::Value::Array(
        merged_strs.iter().map(|s| serde_json::Value::String(s.clone())).collect(),
    );

    let keep_meta = svc.memory_repo.get_by_uuid(&svc.database, kept_uuid).await?.metadata;
    let mut keep_meta_obj = match keep_meta {
        serde_json::Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    keep_meta_obj.insert("merged_from".to_string(), merged_from_val);
    svc.memory_repo
        .update_metadata(&svc.database, kept_uuid, &serde_json::Value::Object(keep_meta_obj))
        .await?;

    for u in &merged {
        let rec = svc.memory_repo.get_by_uuid(&svc.database, *u).await?;
        svc.memory_repo.update_status(&svc.database, *u, MemoryStatus::Archived).await?;

        let mut meta_obj = match rec.metadata {
            serde_json::Value::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        meta_obj
            .insert("merged_into".to_string(), serde_json::Value::String(kept_uuid.to_string()));
        svc.memory_repo
            .update_metadata(&svc.database, *u, &serde_json::Value::Object(meta_obj))
            .await?;
    }

    let merged_uuids_strs: Vec<String> = merged.iter().map(|u| u.to_string()).collect();
    let placeholders: Vec<String> =
        (0..merged_uuids_strs.len()).map(|i| format!("?{}", i + 1)).collect();
    let sql = format!(
        "UPDATE relations SET memory_uuid = ?{} WHERE memory_uuid IN ({})",
        merged_uuids_strs.len() + 1,
        placeholders.join(","),
    );

    let keep_uuid_str = kept_uuid.to_string();
    let mut q = sqlx::query(&sql).bind(&keep_uuid_str);
    for u_str in &merged_uuids_strs {
        q = q.bind(u_str);
    }
    let result = q.execute(&svc.database.pool).await.map_err(NovaError::storage)?;
    let remapped_relations = result.rows_affected() as usize;

    Ok(MergeOutput {
        kept_uuid,
        merged,
        remapped_relations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Uuid,
        config::StorageConfig,
        memory::ops_remember::{RememberInput, service_for_tests},
        storage::{Database, MemoryStatus, entity::EntityRepository, relation::RelationRepository},
    };

    async fn temp_svc() -> crate::memory::MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-m3-merge-{}", Uuid::new_v4()));
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
    async fn basic_merge_keeps_earliest_created_at() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "memory A",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "memory B",
                importance: 0.9,
                ..Default::default()
            })
            .await
            .unwrap();

        let out = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, b.uuid],
                keep_uuid: None,
            })
            .await
            .unwrap();

        assert_eq!(out.kept_uuid, a.uuid, "earliest created_at should be kept");
        assert_eq!(out.merged, vec![b.uuid]);
        assert_eq!(out.remapped_relations, 0);
    }

    #[tokio::test]
    async fn merge_with_explicit_keep_uuid() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "memory A",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "memory B",
                importance: 0.9,
                ..Default::default()
            })
            .await
            .unwrap();

        let out = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, b.uuid],
                keep_uuid: Some(b.uuid),
            })
            .await
            .unwrap();

        assert_eq!(out.kept_uuid, b.uuid);
        assert_eq!(out.merged, vec![a.uuid]);
    }

    #[tokio::test]
    async fn merge_archives_merged_entries_and_injects_metadata() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "keep me",
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "merge me",
                ..Default::default()
            })
            .await
            .unwrap();

        let _out = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, b.uuid],
                keep_uuid: Some(a.uuid),
            })
            .await
            .unwrap();

        let kept = svc.get_memory(a.uuid).await.unwrap();
        assert_eq!(kept.status, MemoryStatus::Active);
        assert_eq!(kept.metadata["merged_from"], serde_json::json!([b.uuid.to_string()]));

        let archived = svc.get_memory(b.uuid).await.unwrap();
        assert_eq!(archived.status, MemoryStatus::Archived);
        assert_eq!(archived.metadata["merged_into"], serde_json::json!(a.uuid.to_string()));
    }

    #[tokio::test]
    async fn merge_remaps_relations() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "memory A",
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "memory B",
                ..Default::default()
            })
            .await
            .unwrap();

        let pool = &svc.database.pool;

        let entity_repo = crate::storage::entity::SqliteEntityRepository::new();
        let ent_a = entity_repo
            .upsert(
                &svc.database,
                crate::storage::entity::UpsertEntityInput {
                    name: "EntityA",
                    r#type: "test",
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();
        let ent_b = entity_repo
            .upsert(
                &svc.database,
                crate::storage::entity::UpsertEntityInput {
                    name: "EntityB",
                    r#type: "test",
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();

        let rel_repo = crate::storage::relation::SqliteRelationRepository::new();
        rel_repo
            .insert(
                &svc.database,
                crate::storage::relation::InsertRelationInput {
                    source_uuid: ent_a,
                    target_uuid: ent_b,
                    predicate: "mentions",
                    confidence: 1.0,
                    memory_uuid: Some(b.uuid),
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap();
        rel_repo
            .insert(
                &svc.database,
                crate::storage::relation::InsertRelationInput {
                    source_uuid: ent_b,
                    target_uuid: ent_a,
                    predicate: "mentions",
                    confidence: 1.0,
                    memory_uuid: Some(b.uuid),
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap();
        rel_repo
            .insert(
                &svc.database,
                crate::storage::relation::InsertRelationInput {
                    source_uuid: ent_a,
                    target_uuid: ent_b,
                    predicate: "knows",
                    confidence: 0.8,
                    memory_uuid: Some(a.uuid),
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap();

        let out = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, b.uuid],
                keep_uuid: Some(a.uuid),
            })
            .await
            .unwrap();

        assert_eq!(
            out.remapped_relations, 2,
            "two relations pointing to b should be remapped to a"
        );

        let rels_b: Vec<(String,)> =
            sqlx::query_as("SELECT memory_uuid FROM relations WHERE memory_uuid = ?1")
                .bind(b.uuid.to_string())
                .fetch_all(pool)
                .await
                .unwrap();
        assert_eq!(rels_b.len(), 0, "no relations should point to b anymore");

        let rels_a: Vec<(String,)> =
            sqlx::query_as("SELECT memory_uuid FROM relations WHERE memory_uuid = ?1")
                .bind(a.uuid.to_string())
                .fetch_all(pool)
                .await
                .unwrap();
        assert_eq!(rels_a.len(), 3, "all three relations should point to a now");
    }

    #[tokio::test]
    async fn merge_validation_rejects_less_than_two() {
        let svc = temp_svc().await;
        let err = svc
            .merge(MergeInput {
                uuids: vec![Uuid::new_v4()],
                keep_uuid: None,
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_validation_rejects_more_than_fifty() {
        let svc = temp_svc().await;
        let uuids: Vec<Uuid> = (0..51).map(|_| Uuid::new_v4()).collect();
        let err = svc
            .merge(MergeInput {
                uuids,
                keep_uuid: None,
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_validation_rejects_duplicate_uuids() {
        let svc = temp_svc().await;
        let uuid = Uuid::new_v4();
        let err = svc
            .merge(MergeInput {
                uuids: vec![uuid, uuid],
                keep_uuid: None,
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_validation_rejects_nonexistent_uuid() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "exists",
                ..Default::default()
            })
            .await
            .unwrap();
        let err = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, Uuid::new_v4()],
                keep_uuid: None,
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_validation_rejects_keep_uuid_not_in_list() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "A",
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "B",
                ..Default::default()
            })
            .await
            .unwrap();
        let err = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, b.uuid],
                keep_uuid: Some(Uuid::new_v4()),
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_idempotent_repeat_call_safe() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "keep",
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "merge me",
                ..Default::default()
            })
            .await
            .unwrap();

        let out1 = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, b.uuid],
                keep_uuid: Some(a.uuid),
            })
            .await
            .unwrap();
        assert_eq!(out1.merged, vec![b.uuid]);

        let out2 = svc
            .merge(MergeInput {
                uuids: vec![a.uuid, b.uuid],
                keep_uuid: Some(a.uuid),
            })
            .await
            .unwrap();
        assert_eq!(out2.kept_uuid, a.uuid);
        assert_eq!(out2.merged, vec![b.uuid]);

        let archived = svc.get_memory(b.uuid).await.unwrap();
        assert_eq!(archived.status, MemoryStatus::Archived);
        assert_eq!(archived.metadata["merged_into"], serde_json::json!(a.uuid.to_string()));
    }
}
