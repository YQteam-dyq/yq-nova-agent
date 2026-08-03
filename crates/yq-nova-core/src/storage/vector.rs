//! Vector store. MVP default: embeddings stored as little-endian
//! `f32` BLOBs in the `embeddings.vec_blob` column, with linear-scan
//! cosine similarity search. This is intentionally SIMPLE — 100k rows ×
//! 1536 dims is ~1.2GB of memory-mappable BLOBs, which scans in <10ms
//! on a modern laptop. Upgrade path: enable `sqlite-vec` feature in
//! M3+ to get HNSW-backed ANN search without touching any callers of
//! `VectorStore::knn_search`.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{NovaError, NovaResult};

/// A single vector search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorHit {
    pub memory_uuid: Uuid,
    /// Cosine similarity, in [-1, 1] (higher = more similar).
    pub similarity: f32,
}

/// Unified vector-store abstraction, implemented by the linear-scan
/// default and (later) by the sqlite-vec HNSW backend.
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Debug {
    /// Expected vector dimensionality. Providers calling `insert_vector`
    /// with the wrong length get a validation error.
    fn dimensions(&self) -> usize;

    /// Insert / overwrite an embedding for `memory_uuid`.
    async fn insert_vector(
        &self,
        memory_uuid: Uuid,
        provider: &str,
        model: &str,
        vec: &[f32],
    ) -> NovaResult<()>;

    /// Delete any stored vector for `memory_uuid`. No-op if absent.
    async fn delete_vector(&self, memory_uuid: Uuid) -> NovaResult<()>;

    /// Return at most `k` rows whose cosine similarity to `query` is
    /// ≥ `threshold`. Results must be sorted descending by similarity.
    async fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        threshold: f32,
    ) -> NovaResult<Vec<VectorHit>>;
}

// -----------------------------------------------------------------------------
// Helpers (used by all vector-store impls).
// -----------------------------------------------------------------------------

/// Validate a slice's length equals the required dims; returns a stable
/// error code we can match on in tests.
pub fn check_dims(dims: usize, v: &[f32]) -> NovaResult<()> {
    if v.len() != dims {
        return Err(NovaError::validation(format!(
            "embedding dims mismatch: expected {dims}, got {}",
            v.len()
        )));
    }
    // Sanity: non-finite floats poison similarity math.
    if v.iter().any(|f| !f.is_finite()) {
        return Err(NovaError::validation("embedding contains NaN or non-finite float"));
    }
    Ok(())
}

/// Encode `v` as a little-endian byte blob for storage.
pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode a little-endian f32 blob back to `Vec<f32>`. Returns an error
/// if the length is not a multiple of 4 (corrupt blob).
pub fn blob_to_vec(blob: &[u8]) -> NovaResult<Vec<f32>> {
    if blob.len() % 4 != 0 {
        return Err(NovaError::storage_msg(format!(
            "corrupt vector blob: length {} not multiple of 4",
            blob.len()
        )));
    }
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().map_err(|_| NovaError::storage_msg("corrupt chunk"))?;
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

/// Cosine similarity of two equal-length vectors.
/// Returns `0.0` for zero-norm inputs (avoids div-by-zero; similarity
/// to a zero vector is meaningless anyway, and callers will filter it).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= f32::EPSILON {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

/// Arc-wrapped boxed vector store so we can cheaply clone a handle into
/// repositories / background tasks without requiring a generic parameter
/// on every public struct.
pub type SharedVectorStore = Arc<dyn VectorStore>;

// -----------------------------------------------------------------------------
// SqliteVectorStore — linear-scan KNN over embeddings.vec_blob blobs.
// -----------------------------------------------------------------------------

/// Default vector dimensionality used when callers don't specify otherwise.
/// Matches OpenAI `text-embedding-3-small` / Voyage `voyage-3-lite` etc.
pub const DEFAULT_DIMS: usize = 1536;

/// SQLite-backed implementation of `VectorStore`. Upserts one row per
/// `memory_uuid` into the `embeddings` table and does an in-process
/// linear scan over decoded blobs for every `knn_search` call. For
/// ~100k rows × 1536 dims this scans ~600 MB of BLOBs and fits in RAM;
/// upgrade to `sqlite-vec` HNSW (behind a feature flag) when the dataset
/// outgrows linear scan.
#[derive(Clone)]
pub struct SqliteVectorStore {
    pool: sqlx::SqlitePool,
    dims: usize,
}

impl std::fmt::Debug for SqliteVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteVectorStore").field("dims", &self.dims).finish_non_exhaustive()
    }
}

impl SqliteVectorStore {
    pub fn new(pool: sqlx::SqlitePool, dims: usize) -> Self {
        Self { pool, dims }
    }

    pub fn with_db(db: &crate::storage::Database, dims: usize) -> Self {
        Self::new(db.pool.clone(), dims)
    }

    pub fn default_dims(&self) -> usize {
        self.dims
    }
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn insert_vector(
        &self,
        memory_uuid: Uuid,
        provider: &str,
        model: &str,
        vec: &[f32],
    ) -> NovaResult<()> {
        check_dims(self.dims, vec)?;
        if provider.trim().is_empty() {
            return Err(NovaError::validation("embedding.provider must not be empty"));
        }
        if model.trim().is_empty() {
            return Err(NovaError::validation("embedding.model must not be empty"));
        }

        let blob = vec_to_blob(vec);
        let now = chrono::Utc::now().timestamp();
        let mem_s = memory_uuid.to_string();
        sqlx::query(
            "INSERT INTO embeddings (memory_uuid, dims, provider, model, vec_blob, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(memory_uuid) DO UPDATE SET \
               dims = excluded.dims, \
               provider = excluded.provider, \
               model = excluded.model, \
               vec_blob = excluded.vec_blob, \
               created_at = excluded.created_at",
        )
        .bind(&mem_s)
        .bind(self.dims as i64)
        .bind(provider.trim())
        .bind(model.trim())
        .bind(blob)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(())
    }

    async fn delete_vector(&self, memory_uuid: Uuid) -> NovaResult<()> {
        let mem_s = memory_uuid.to_string();
        sqlx::query("DELETE FROM embeddings WHERE memory_uuid = ?1")
            .bind(&mem_s)
            .execute(&self.pool)
            .await
            .map_err(NovaError::storage)?;
        Ok(())
    }

    async fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        threshold: f32,
    ) -> NovaResult<Vec<VectorHit>> {
        check_dims(self.dims, query)?;
        let k = k.min(500); // safety cap

        // Pull all stored vectors into the process. Blobs are decoded on
        // the fly. This is O(N × D) per call — fine for MVP datasets of
        // ≤ a few hundred thousand rows; switch to sqlite-vec HNSW for
        // larger collections.
        let rows: Vec<(String, Vec<u8>)> =
            sqlx::query_as("SELECT memory_uuid, vec_blob FROM embeddings WHERE dims = ?1")
                .bind(self.dims as i64)
                .fetch_all(&self.pool)
                .await
                .map_err(NovaError::storage)?;

        let mut hits: Vec<VectorHit> = Vec::with_capacity(rows.len().min(k * 2));
        for (mem_s, blob) in rows {
            let v = match blob_to_vec(&blob) {
                Ok(v) => v,
                Err(_) => continue, // corrupt row — skip
            };
            if v.len() != query.len() {
                continue;
            }
            let sim = cosine_similarity(query, &v);
            if sim >= threshold {
                let mem_uuid = match Uuid::parse_str(&mem_s) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                hits.push(VectorHit { memory_uuid: mem_uuid, similarity: sim });
            }
        }

        // Sort by similarity desc and take top k.
        hits.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }
}

// -----------------------------------------------------------------------------
// Tests (SqliteVectorStore)
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;
    use crate::storage::{
        Database,
        memory::{InsertMemoryInput, MemoryRepository, SqliteMemoryRepository},
    };

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-m2-vec-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        Database::open(cfg).await.expect("open temp db")
    }

    #[tokio::test]
    async fn insert_upsert_and_delete_vector() {
        let db = temp_db().await;
        let mem = SqliteMemoryRepository::new();
        let muuid = mem
            .insert(&db, InsertMemoryInput { content: "x", ..Default::default() })
            .await
            .unwrap()
            .uuid();
        let store = SqliteVectorStore::with_db(&db, 4);
        let v = [0.1f32, 0.2, 0.3, 0.4];
        store.insert_vector(muuid, "test", "m1", &v).await.unwrap();

        // Upsert: overwrite provider/model
        store.insert_vector(muuid, "test2", "m2", &v).await.unwrap();
        store.delete_vector(muuid).await.unwrap();

        // After delete, knn_search returns 0 hits for this uuid
        let hits = store.knn_search(&v, 10, -1.0).await.unwrap();
        assert!(hits.iter().all(|h| h.memory_uuid != muuid));
    }

    #[tokio::test]
    async fn knn_picks_most_similar_and_filters_threshold() {
        let db = temp_db().await;
        let mem = SqliteMemoryRepository::new();
        let store = SqliteVectorStore::with_db(&db, 3);

        // Sanity: insert one vector directly and verify we can fetch it back
        // through raw SQL before we trust anything cosine-related.
        let u0 = mem
            .insert(&db, InsertMemoryInput { content: "m0", ..Default::default() })
            .await
            .unwrap()
            .uuid();
        store.insert_vector(u0, "t", "m", &[1.0_f32, 0.0, 0.0]).await.unwrap();
        let rows: Vec<(String, Vec<u8>, i64)> =
            sqlx::query_as("SELECT memory_uuid, vec_blob, dims FROM embeddings")
                .fetch_all(&db.pool)
                .await
                .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].2, 3); // dims
        let decoded = blob_to_vec(&rows[0].1).unwrap();
        assert_eq!(decoded, vec![1.0_f32, 0.0, 0.0]);

        // Now add uuids1 and uuids2
        let u1 = mem
            .insert(&db, InsertMemoryInput { content: "m1", ..Default::default() })
            .await
            .unwrap()
            .uuid();
        let u2 = mem
            .insert(&db, InsertMemoryInput { content: "m2", ..Default::default() })
            .await
            .unwrap()
            .uuid();
        store.insert_vector(u1, "t", "m", &[-1.0_f32, 0.0, 0.0]).await.unwrap();
        store.insert_vector(u2, "t", "m", &[0.7_f32, 0.7, 0.0]).await.unwrap();

        let q = [1.0f32, 0.0, 0.0];
        // threshold -2: all 3 must return (cos=-1 is the lowest possible)
        let all3 = store.knn_search(&q, 10, -2.0).await.unwrap();
        assert_eq!(all3.len(), 3, "all 3 vectors should pass threshold=-2, got {all3:?}");

        // threshold 0: only u0 (cos=1) and u2 (cos≈0.707) pass
        let nonneg = store.knn_search(&q, 5, 0.0).await.unwrap();
        assert_eq!(nonneg.len(), 2, "nonneg count wrong: {nonneg:?}");
        let top = &nonneg[0];
        assert_eq!(top.memory_uuid, u0, "top should be u0, got sim={}", top.similarity);
        assert!(
            (top.similarity - 1.0).abs() < 1e-4,
            "u0 sim should be ~1.0, got {}",
            top.similarity
        );
        assert_eq!(nonneg[1].memory_uuid, u2);

        // threshold 0.9: only u0
        let hi = store.knn_search(&q, 5, 0.9).await.unwrap();
        assert_eq!(hi.len(), 1);
        assert_eq!(hi[0].memory_uuid, u0);
    }

    #[test]
    fn roundtrip_blob_identical() {
        let v: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let blob = vec_to_blob(&v);
        assert_eq!(blob.len(), 64);
        let back = blob_to_vec(&blob).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn cosine_identical_vectors_are_one() {
        let a = [1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_is_minus_one() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn bad_dims_rejected() {
        let err = check_dims(4, &[1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[test]
    fn nan_rejected() {
        let err = check_dims(2, &[1.0, f32::NAN]).unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }
}
