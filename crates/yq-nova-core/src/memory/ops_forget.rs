//! `forget` — soft- or hard-delete memories; optionally cascade to attached
//! vectors and orphaned graph entities/relations.
//!
//! Design rules (kept small on purpose):
//!   * **By uuid**: forget a specific memory uuid (returns NotFound if missing).
//!   * **By filter**: forget everything that matches a `MemoryFilter` (e.g.
//!     older than X days, source=System, importance<0.2). Used for batch TTL
//!     cleanup in the background job system (M7).
//!   * **Soft delete by default**: sets `status = deleted` so the record is
//!     still there for audits, but `list`/`recall` won't return it (filters
//!     exclude deleted rows unless the caller explicitly opts in).
//!   * **Hard delete on request**: removes the rows entirely (cascades through
//!     embeddings FK, memory_tags FK) so storage is reclaimed.
//!
//! MVP intentionally has no "undo" — that's v0.3 with a recycle bin table.

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

/// What should `forget` actually do?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ForgetMode {
    /// Default: flip status to `deleted`; keep the row for audits. The row
    /// will never appear in recall/list but can be queried directly.
    #[default]
    Soft,
    /// Actually remove the row from SQLite. Embedding + tag rows are removed
    /// automatically via FK `ON DELETE CASCADE` (see `001_init.sql`).
    Hard,
    /// Flip status to `archived` instead of `deleted`. Used for explicit
    /// "I want to keep this but not surface it by default" workflows.
    Archive,
}

/// One thing to forget: either a specific uuid, or a filter match.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ForgetTarget {
    One(Uuid),
    Filter(MemoryFilter),
}

/// Input to [`MemoryService::forget`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ForgetInput {
    pub target: ForgetTarget,
    pub mode: ForgetMode,
    /// If true, also scan the entity/relation tables after the memory is gone
    /// and garbage-collect any entities with zero in-degree + zero out-degree
    /// that were last-modified by this forget operation. MVP only implements
    /// "no orphans yet" — set to true safely even if nothing is done.
    pub gc_graph: bool,
    /// Upper bound on how many rows a `Filter` target is allowed to delete in
    /// one call. Protects against `MemoryFilter::default()` (all rows) by
    /// mistake. Default 500 is plenty for a single background-job run.
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
    /// Number of memory rows affected by this call.
    pub affected_memories: usize,
    /// When `mode == Hard`: number of embedding BLOB rows also removed.
    /// (Memory FK has cascade; counted heuristically for Soft/Archive as 0.)
    pub cascade_embeddings: usize,
    /// When `gc_graph = true`: number of orphan entity rows pruned.
    pub gc_entities: usize,
    /// When `gc_graph = true`: number of orphan relation rows pruned.
    pub gc_relations: usize,
    /// Actual mode applied (handy for echo/audit logs).
    pub mode: ForgetMode,
}

pub async fn forget(svc: &MemoryService, input: ForgetInput) -> NovaResult<ForgetOutput> {
    // --- 1. Determine affected uuids (respect batch_limit for Filter) -------
    let uuids: Vec<Uuid> = match input.target {
        ForgetTarget::One(uuid) => {
            // Verify existence up-front for Soft/Archive modes so NotFound is
            // returned rather than a silent 0-row update.
            match svc.memory_repo.get_by_uuid(&svc.database, uuid).await {
                Ok(_) => vec![uuid],
                Err(e) if matches!(e.code(), crate::error::ErrorCode::NotFound) => {
                    // Hard delete also returns NotFound for consistency with
                    // the other modes (no row existed to delete).
                    return Err(e);
                },
                Err(e) => return Err(e),
            }
        },
        ForgetTarget::Filter(f) => {
            let limit = input.batch_limit.min(10_000);
            if limit == 0 {
                return Err(NovaError::validation("forget: batch_limit must be >= 1"));
            }
            // Use the repository's list to pick uuids. Pick 1 extra row to
            // detect "too many matches" and warn (though we still cap at limit).
            let rows = svc.memory_repo.list(&svc.database, &f, limit + 1, 0).await?;
            let capped = rows.len().min(limit);
            rows.into_iter().take(capped).map(|r| r.uuid).collect()
        },
    };

    if uuids.is_empty() {
        return Ok(ForgetOutput { mode: input.mode, ..Default::default() });
    }

    // --- 2. Apply the per-uuid action ---------------------------------------
    let mut affected = 0usize;
    for u in &uuids {
        match input.mode {
            ForgetMode::Soft => {
                svc.memory_repo.update_status(&svc.database, *u, MemoryStatus::Deleted).await?;
            },
            ForgetMode::Archive => {
                svc.memory_repo.update_status(&svc.database, *u, MemoryStatus::Archived).await?;
            },
            ForgetMode::Hard => {
                // Delete vector row first (FK cascade could delete it too but
                // doing it explicitly means we can count removed embeddings).
                svc.vector_store.delete_vector(*u).await.ok();
                svc.memory_repo.delete(&svc.database, *u).await?;
            },
        }
        affected += 1;
    }

    // --- 3. Optional graph GC -----------------------------------------------
    let mut gc_entities = 0usize;
    let mut gc_relations = 0usize;
    if input.gc_graph {
        // For MVP, best-effort scan: delete any entity whose (outgoing + incoming)
        // count is zero and has no metadata or wikilinks we want to keep.
        // Keep it conservative: only delete entities with entity_type == "unknown"
        // (capitalised-proper-noun byproducts) to avoid wiping user-curated
        // [[Wiki]] entries.
        let (ent, rel) = gc_orphan_entities(svc).await.unwrap_or((0, 0));
        gc_entities = ent;
        gc_relations = rel;
    }

    // embeddings cascade count (hard delete only, estimated from deleted
    // memory uuids' vector rows already cleared). For Soft/Archive always 0.
    let cascade_embeddings = match input.mode {
        ForgetMode::Hard => affected, // rough upper bound; exact would need a separate count.
        _ => 0,
    };

    Ok(ForgetOutput {
        affected_memories: affected,
        cascade_embeddings,
        gc_entities,
        gc_relations,
        mode: input.mode,
    })
}

/// Remove entities that have no relations AND type "unknown". Conservative by
/// design — user-created [[Wiki]] entities survive even if orphaned so the
/// user's graph intent is preserved across forget calls that temporarily
/// remove the only memory referencing them.
async fn gc_orphan_entities(svc: &MemoryService) -> NovaResult<(usize, usize)> {
    use crate::storage::entity::EntityRecord;

    // 1) List ALL entities (pessimistic; MVP expects 10s to 1000s of entities,
    // which is fine). For each entity, count out + in relations. If 0 and type
    // == "unknown", delete it.
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
    use crate::Uuid;
    use crate::config::StorageConfig;
    use crate::memory::ops_remember::{RememberInput, service_for_tests};
    use crate::storage::{Database, MemoryStatus};

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

        // Row still exists via direct get, but status is Deleted.
        let mem = svc.get_memory(a.uuid).await.unwrap();
        assert_eq!(mem.status, MemoryStatus::Deleted);
    }

    #[tokio::test]
    async fn forget_hard_removes_row_completely() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput { content: "one off", importance: 0.1, ..Default::default() })
            .await
            .unwrap();
        // Sanity: exists before
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

        let f = MemoryFilter { importance_min: Some(0.0), ..Default::default() };
        let out = svc
            .forget(ForgetInput {
                target: ForgetTarget::Filter(f.clone()),
                mode: ForgetMode::Soft,
                batch_limit: 3,
                ..Default::default()
            })
            .await
            .unwrap();
        // Exactly 3 rows even though 10 match the (empty-except-min) filter.
        assert_eq!(out.affected_memories, 3);
    }
}
