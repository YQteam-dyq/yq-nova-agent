use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::Database,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackVerdict {
    #[default]
    Correct,
    Incorrect,
    Partial,
}

impl FeedbackVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            FeedbackVerdict::Correct => "correct",
            FeedbackVerdict::Incorrect => "incorrect",
            FeedbackVerdict::Partial => "partial",
        }
    }
}

impl TryFrom<&str> for FeedbackVerdict {
    type Error = NovaError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "correct" => FeedbackVerdict::Correct,
            "incorrect" => FeedbackVerdict::Incorrect,
            "partial" => FeedbackVerdict::Partial,
            other => {
                return Err(NovaError::validation(format!("unknown feedback verdict: {other}")));
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackTargetKind {
    #[default]
    Entity,
    Relation,
    Tag,
}

impl FeedbackTargetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FeedbackTargetKind::Entity => "entity",
            FeedbackTargetKind::Relation => "relation",
            FeedbackTargetKind::Tag => "tag",
        }
    }
}

impl TryFrom<&str> for FeedbackTargetKind {
    type Error = NovaError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "entity" => FeedbackTargetKind::Entity,
            "relation" => FeedbackTargetKind::Relation,
            "tag" => FeedbackTargetKind::Tag,
            other => {
                return Err(NovaError::validation(format!("unknown feedback target: {other}")));
            },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionFeedbackInput {
    pub memory_uuid: Uuid,
    pub kind: FeedbackTargetKind,
    pub name: String,
    pub verdict: FeedbackVerdict,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRecord {
    pub id: i64,
    pub memory_uuid: Uuid,
    pub kind: FeedbackTargetKind,
    pub name: String,
    pub verdict: FeedbackVerdict,
    pub note: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KindStats {
    pub total: u64,
    pub correct: u64,
    pub incorrect: u64,
    pub partial: u64,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityStats {
    pub total: u64,
    pub correct: u64,
    pub incorrect: u64,
    pub partial: u64,
    pub accuracy: f64,
    pub entity: KindStats,
    pub relation: KindStats,
    pub tag: KindStats,
}

pub struct SqliteQualityStore;

impl SqliteQualityStore {
    pub fn new() -> Self {
        Self
    }

    async fn ensure_schema(db: &Database) -> NovaResult<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS extraction_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_uuid TEXT NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                verdict TEXT NOT NULL,
                note TEXT,
                created_at INTEGER NOT NULL
            )",
        )
        .execute(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_feedback_memory ON extraction_feedback(memory_uuid)",
        )
        .execute(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_feedback_kind ON extraction_feedback(kind)")
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        Ok(())
    }

    pub async fn record(
        &self,
        db: &Database,
        input: ExtractionFeedbackInput,
    ) -> NovaResult<FeedbackRecord> {
        Self::ensure_schema(db).await?;
        let name = input.name.trim().to_string();
        if name.is_empty() {
            return Err(NovaError::validation("feedback name must not be empty"));
        }
        let note = input.note.map(|n| n.trim().to_string()).filter(|n| !n.is_empty());
        let now = Utc::now();
        let result = sqlx::query(
            "INSERT INTO extraction_feedback (memory_uuid, kind, name, verdict, note, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .bind(input.memory_uuid.to_string())
        .bind(input.kind.as_str())
        .bind(&name)
        .bind(input.verdict.as_str())
        .bind(note.as_deref())
        .bind(now.timestamp())
        .execute(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(FeedbackRecord {
            id: result.last_insert_rowid(),
            memory_uuid: input.memory_uuid,
            kind: input.kind,
            name,
            verdict: input.verdict,
            note,
            created_at: now,
        })
    }

    pub async fn list(
        &self,
        db: &Database,
        memory_uuid: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<FeedbackRecord>> {
        Self::ensure_schema(db).await?;
        let limit = limit.min(1000) as i64;
        let offset = offset as i64;
        let rows = match memory_uuid {
            Some(uuid) => sqlx::query(
                "SELECT id, memory_uuid, kind, name, verdict, note, created_at FROM \
                 extraction_feedback WHERE memory_uuid = ?1 ORDER BY id DESC LIMIT ?2 OFFSET ?3",
            )
            .bind(uuid.to_string())
            .bind(limit)
            .bind(offset)
            .fetch_all(&db.pool)
            .await
            .map_err(NovaError::storage)?,
            None => sqlx::query(
                "SELECT id, memory_uuid, kind, name, verdict, note, created_at FROM \
                 extraction_feedback ORDER BY id DESC LIMIT ?1 OFFSET ?2",
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&db.pool)
            .await
            .map_err(NovaError::storage)?,
        };
        rows.iter().map(row_to_record).collect()
    }

    pub async fn stats(&self, db: &Database) -> NovaResult<QualityStats> {
        Self::ensure_schema(db).await?;
        let rows: Vec<(String, String, i64)> = sqlx::query_as(
            "SELECT kind, verdict, COUNT(*) FROM extraction_feedback GROUP BY kind, verdict",
        )
        .fetch_all(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let mut out = QualityStats::default();
        for (kind_raw, verdict_raw, count) in rows {
            let kind = FeedbackTargetKind::try_from(kind_raw.as_str())?;
            let verdict = FeedbackVerdict::try_from(verdict_raw.as_str())?;
            let n = count as u64;
            let bucket = match kind {
                FeedbackTargetKind::Entity => &mut out.entity,
                FeedbackTargetKind::Relation => &mut out.relation,
                FeedbackTargetKind::Tag => &mut out.tag,
            };
            bucket.total += n;
            match verdict {
                FeedbackVerdict::Correct => bucket.correct += n,
                FeedbackVerdict::Incorrect => bucket.incorrect += n,
                FeedbackVerdict::Partial => bucket.partial += n,
            }
        }
        out.total = out.entity.total + out.relation.total + out.tag.total;
        out.correct = out.entity.correct + out.relation.correct + out.tag.correct;
        out.incorrect = out.entity.incorrect + out.relation.incorrect + out.tag.incorrect;
        out.partial = out.entity.partial + out.relation.partial + out.tag.partial;
        set_accuracy(&mut out.accuracy, out.correct, out.total);
        set_accuracy(&mut out.entity.accuracy, out.entity.correct, out.entity.total);
        set_accuracy(&mut out.relation.accuracy, out.relation.correct, out.relation.total);
        set_accuracy(&mut out.tag.accuracy, out.tag.correct, out.tag.total);
        Ok(out)
    }

    pub async fn delete(&self, db: &Database, id: i64) -> NovaResult<()> {
        Self::ensure_schema(db).await?;
        let result = sqlx::query("DELETE FROM extraction_feedback WHERE id = ?1")
            .bind(id)
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if result.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("feedback record {id}")));
        }
        Ok(())
    }
}

impl Default for SqliteQualityStore {
    fn default() -> Self {
        Self::new()
    }
}

fn set_accuracy(target: &mut f64, correct: u64, total: u64) {
    *target = if total == 0 { 0.0 } else { correct as f64 / total as f64 };
}

fn row_to_record(row: &sqlx::sqlite::SqliteRow) -> NovaResult<FeedbackRecord> {
    let id: i64 = row.try_get("id").map_err(NovaError::storage)?;
    let memory_uuid_raw: String = row.try_get("memory_uuid").map_err(NovaError::storage)?;
    let memory_uuid = Uuid::parse_str(&memory_uuid_raw)
        .map_err(|e| NovaError::storage_msg(format!("bad memory uuid: {e}")))?;
    let kind_raw: String = row.try_get("kind").map_err(NovaError::storage)?;
    let kind = FeedbackTargetKind::try_from(kind_raw.as_str())?;
    let name: String = row.try_get("name").map_err(NovaError::storage)?;
    let verdict_raw: String = row.try_get("verdict").map_err(NovaError::storage)?;
    let verdict = FeedbackVerdict::try_from(verdict_raw.as_str())?;
    let note: Option<String> = row.try_get("note").map_err(NovaError::storage)?;
    let created_ts: i64 = row.try_get("created_at").map_err(NovaError::storage)?;
    let created_at = DateTime::from_timestamp(created_ts, 0)
        .ok_or_else(|| NovaError::storage_msg(format!("bad timestamp {created_ts}")))?;
    Ok(FeedbackRecord {
        id,
        memory_uuid,
        kind,
        name,
        verdict,
        note,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::StorageConfig, storage::Database};

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-b3-quality-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        Database::open(config).await.unwrap()
    }

    fn feedback(
        memory_uuid: Uuid,
        kind: FeedbackTargetKind,
        verdict: FeedbackVerdict,
    ) -> ExtractionFeedbackInput {
        ExtractionFeedbackInput {
            memory_uuid,
            kind,
            name: "target".to_string(),
            verdict,
            note: None,
        }
    }

    #[tokio::test]
    async fn record_and_stats_aggregate_by_kind_and_verdict() {
        let db = temp_db().await;
        let store = SqliteQualityStore::new();
        let memory_uuid = Uuid::new_v4();
        store
            .record(
                &db,
                feedback(memory_uuid, FeedbackTargetKind::Entity, FeedbackVerdict::Correct),
            )
            .await
            .unwrap();
        store
            .record(
                &db,
                feedback(memory_uuid, FeedbackTargetKind::Entity, FeedbackVerdict::Correct),
            )
            .await
            .unwrap();
        store
            .record(
                &db,
                feedback(memory_uuid, FeedbackTargetKind::Relation, FeedbackVerdict::Incorrect),
            )
            .await
            .unwrap();
        store
            .record(&db, feedback(memory_uuid, FeedbackTargetKind::Tag, FeedbackVerdict::Partial))
            .await
            .unwrap();

        let stats = store.stats(&db).await.unwrap();
        assert_eq!(stats.total, 4);
        assert_eq!(stats.correct, 2);
        assert_eq!(stats.incorrect, 1);
        assert_eq!(stats.partial, 1);
        assert!((stats.accuracy - 0.5).abs() < 1e-6);
        assert_eq!(stats.entity.total, 2);
        assert!((stats.entity.accuracy - 1.0).abs() < 1e-6);
        assert_eq!(stats.relation.total, 1);
        assert_eq!(stats.relation.accuracy, 0.0);
        assert_eq!(stats.tag.total, 1);
    }

    #[tokio::test]
    async fn list_filters_by_memory_and_orders_newest_first() {
        let db = temp_db().await;
        let store = SqliteQualityStore::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let first = store
            .record(&db, feedback(a, FeedbackTargetKind::Entity, FeedbackVerdict::Correct))
            .await
            .unwrap();
        store
            .record(&db, feedback(b, FeedbackTargetKind::Relation, FeedbackVerdict::Correct))
            .await
            .unwrap();
        let rows = store.list(&db, Some(a), 100, 0).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, first.id);
        let all = store.list(&db, None, 100, 0).await.unwrap();
        assert_eq!(all.len(), 2);
        assert!(all[0].id > all[1].id);
    }

    #[tokio::test]
    async fn delete_removes_record_and_reports_missing() {
        let db = temp_db().await;
        let store = SqliteQualityStore::new();
        let record = store
            .record(
                &db,
                feedback(Uuid::new_v4(), FeedbackTargetKind::Entity, FeedbackVerdict::Correct),
            )
            .await
            .unwrap();
        store.delete(&db, record.id).await.unwrap();
        let err = store.delete(&db, record.id).await.unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn record_rejects_blank_name() {
        let db = temp_db().await;
        let store = SqliteQualityStore::new();
        let mut input =
            feedback(Uuid::new_v4(), FeedbackTargetKind::Entity, FeedbackVerdict::Correct);
        input.name = "   ".to_string();
        let err = store.record(&db, input).await.unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[test]
    fn verdict_and_kind_round_trip() {
        for (kind, raw) in [
            (FeedbackTargetKind::Entity, "entity"),
            (FeedbackTargetKind::Relation, "relation"),
            (FeedbackTargetKind::Tag, "tag"),
        ] {
            assert_eq!(FeedbackTargetKind::try_from(raw).unwrap(), kind);
            assert_eq!(kind.as_str(), raw);
        }
        for (verdict, raw) in [
            (FeedbackVerdict::Correct, "correct"),
            (FeedbackVerdict::Incorrect, "incorrect"),
            (FeedbackVerdict::Partial, "partial"),
        ] {
            assert_eq!(FeedbackVerdict::try_from(raw).unwrap(), verdict);
            assert_eq!(verdict.as_str(), raw);
        }
        assert!(FeedbackVerdict::try_from("meh").is_err());
        assert!(FeedbackTargetKind::try_from("meh").is_err());
    }
}
