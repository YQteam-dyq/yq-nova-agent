
use async_trait::async_trait;

use crate::{
    NovaResult, Uuid,
    error::NovaError,
    storage::{Database, MemoryStatus},
};

#[derive(Debug, Clone)]
pub struct KeywordHit {
    pub uuid: Uuid,

    pub score: f32,

    pub raw_bm25: f32,
}

#[async_trait]
pub trait Fts5Store: Send + Sync + std::fmt::Debug {
    async fn keyword_search(
        &self,
        db: &Database,
        query: &str,
        top_k: usize,
        statuses: &[MemoryStatus],
    ) -> NovaResult<Vec<KeywordHit>>;
}

#[derive(Debug, Clone, Default)]
pub struct SqliteFts5Store;

impl SqliteFts5Store {
    pub const fn new() -> Self {
        Self
    }
}

impl SqliteFts5Store {

    fn normalise_query(q: &str) -> String {
        let trimmed = q.trim();
        if trimmed.is_empty() {
            return String::new();
        }
        let has_meta = trimmed
            .chars()
            .any(|c| matches!(c, '"' | '(' | ')' | '*' | 'O' | 'R' | 'A' | 'N' | 'D' | 'X'));

        if has_meta
            && (trimmed.contains(" OR ")
                || trimmed.contains(" AND ")
                || trimmed.contains(" NOT ")
                || trimmed.contains('"')
                || trimmed.contains('('))
        {
            return trimmed.to_string();
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().filter(|t| !t.is_empty()).collect();
        if tokens.is_empty() {
            return String::new();
        }
        tokens
            .iter()
            .map(|t| {

                if t.chars().any(|c| matches!(c, '-' | '+' | '^' | ':' | '*' | '"' | '(' | ')')) {
                    let escaped = t.replace('"', "\"\"");
                    format!("\"{escaped}\"*")
                } else {
                    format!("{t}*")
                }
            })
            .collect::<Vec<_>>()
            .join(" AND ")
    }
}

#[async_trait]
impl Fts5Store for SqliteFts5Store {
    async fn keyword_search(
        &self,
        db: &Database,
        query: &str,
        top_k: usize,
        statuses: &[MemoryStatus],
    ) -> NovaResult<Vec<KeywordHit>> {
        let q = Self::normalise_query(query);
        if q.is_empty() {
            return Ok(vec![]);
        }
        if top_k == 0 {
            return Ok(vec![]);
        }
        if statuses.is_empty() {
            return Err(NovaError::validation_msg("keyword_search statuses must be non-empty"));
        }

        let placeholders: Vec<&str> = statuses.iter().map(|_| "?").collect();
        let in_clause = placeholders.join(",");

        let sql = format!(
            r#"
            SELECT mi.uuid AS uuid, bm25(memory_fts) AS bm
            FROM memory_fts
            JOIN memory_items mi ON mi.id = memory_fts.rowid
            WHERE memory_fts MATCH ?
              AND mi.status IN ({in_clause})
            ORDER BY bm25(memory_fts) ASC
            LIMIT ?
            "#
        );

        let mut built = sqlx::query_as::<_, (String, f64)>(&sql).bind(q.clone());
        for s in statuses {
            built = built.bind(s.as_str());
        }
        built = built.bind(top_k as i64);

        let rows = built
            .fetch_all(&db.pool)
            .await
            .map_err(|e| NovaError::validation_msg(format!("fts5 query invalid ({q:?}): {e}")))?;

        let raw: Vec<(Uuid, f32)> = rows
            .into_iter()
            .map(|(u, b)| (Uuid::parse_str(&u).map_err(NovaError::storage).unwrap(), b as f32))
            .collect();
        if raw.is_empty() {
            return Ok(vec![]);
        }
        let min_bm = raw.iter().map(|(_, b)| *b).fold(f32::INFINITY, f32::min);
        let max_bm = raw.iter().map(|(_, b)| *b).fold(f32::NEG_INFINITY, f32::max);
        let range = max_bm - min_bm;
        let hits: Vec<KeywordHit> = raw
            .into_iter()
            .map(|(uuid, b)| {
                let norm = if range <= 0.0 {
                    1.0f32
                } else {

                    let r: f32 = 1.0 - (b - min_bm) / range;
                    r.clamp(0.0_f32, 1.0_f32)
                };
                KeywordHit { uuid, score: norm, raw_bm25: b }
            })
            .collect();
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::StorageConfig,
        embedding::MockEmbeddingProvider,
        memory::{MemoryService, RememberInput},
        storage::Database,
    };
    use std::{path::PathBuf, sync::Arc, time::SystemTime};

    fn tmp(tag: &str) -> StorageConfig {
        let mut p = std::env::temp_dir();
        let stamp =
            SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        let nonce = stamp ^ (stamp >> 13) ^ 0x9e3779b97f4a7c15;
        p.push(format!("yqnova-fts-{tag}-{}-{nonce}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(PathBuf::from(format!("{}-wal", p.display())));
        let _ = std::fs::remove_file(PathBuf::from(format!("{}-shm", p.display())));
        StorageConfig {
            db_path: p,
            wal_mode: true,
            page_size: 4096,
            cache_size_kb: 8192,
            busy_timeout_ms: 5000,
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..Default::default()
        }
    }

    async fn setup() -> (Database, MemoryService, SqliteFts5Store) {
        let cfg = tmp("s");
        let db = Database::open(cfg).await.unwrap();
        let provider: crate::embedding::SharedEmbeddingProvider =
            Arc::new(MockEmbeddingProvider::new(16));
        let mem = MemoryService::new(db.clone(), provider);
        (db, mem, SqliteFts5Store::new())
    }

    #[tokio::test]
    async fn fts_matches_prefix_and_multitoken() {
        let (db, mem, store) = setup().await;
        mem.remember(RememberInput { content: "pineapple smoothie recipe", ..Default::default() })
            .await
            .unwrap();
        mem.remember(RememberInput { content: "banana smoothie bowl", ..Default::default() })
            .await
            .unwrap();
        mem.remember(RememberInput {
            content: "unrelated rust code snippet",
            ..Default::default()
        })
        .await
        .unwrap();

        let hits =
            store.keyword_search(&db, "pine smooth", 10, &[MemoryStatus::Active]).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-6, "only hit should be 1.0");

        let all_smoothie =
            store.keyword_search(&db, "smoothie", 10, &[MemoryStatus::Active]).await.unwrap();
        assert_eq!(all_smoothie.len(), 2);

        assert!(all_smoothie[0].score >= all_smoothie[1].score);
    }

    #[tokio::test]
    async fn fts_empty_query_empty_result() {
        let (db, _, store) = setup().await;
        assert!(
            store.keyword_search(&db, "   ", 10, &[MemoryStatus::Active]).await.unwrap().is_empty()
        );
    }

    #[tokio::test]
    async fn fts_filters_by_status() {
        let (db, mem, store) = setup().await;
        mem.remember(RememberInput { content: "archived note", ..Default::default() })
            .await
            .unwrap();
        sqlx::query("UPDATE memory_items SET status = 'archived' WHERE content = 'archived note'")
            .execute(&db.pool)
            .await
            .unwrap();
        mem.remember(RememberInput { content: "active note", ..Default::default() }).await.unwrap();

        let only_active =
            store.keyword_search(&db, "note", 10, &[MemoryStatus::Active]).await.unwrap();
        assert_eq!(only_active.len(), 1);

        let only_archived =
            store.keyword_search(&db, "note", 10, &[MemoryStatus::Archived]).await.unwrap();
        assert_eq!(only_archived.len(), 1);

        let both = store
            .keyword_search(&db, "note", 10, &[MemoryStatus::Active, MemoryStatus::Archived])
            .await
            .unwrap();
        assert_eq!(both.len(), 2);
    }
}
