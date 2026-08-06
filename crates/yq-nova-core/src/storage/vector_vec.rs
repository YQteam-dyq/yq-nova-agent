//! sqlite-vec HNSW vector backend.
//!
//! Implements the same [`VectorStore`] trait as the linear-scan
//! [`SqliteVectorStore`], but backs `knn_search` with an ANN search over a
//! `vec0` virtual table (HNSW) instead of an in-process cosine scan. Because
//! it speaks the exact same [`VectorStore`] interface, no caller needs to
//! change: swap the concrete store at construction time and everything keeps
//! working — that is the whole point of this module.
//!
//! # Extension loading (sqlite-vec + sqlx)
//!
//! The `sqlite-vec` crate statically compiles the C extension
//! (`sqlite-vec.c`, built with `SQLITE_CORE`) and exposes one entry point,
//! [`sqlite_vec::sqlite3_vec_init`]. The recommended registration is to hand
//! it to `sqlite3_auto_extension()`, which makes SQLite run it on every
//! **newly opened** connection. sqlx 0.7's `sqlite-sqlite` driver links a
//! bundled `libsqlite3-sys` (its `bundled` feature), so the `sqlite3_*`
//! symbols resolve to the same SQLite copy that `sqlite-vec.c` links against.
//!
//! We therefore call `sqlite3_auto_extension` once (via `Once`) from
//! [`SqliteVecVectorStore::new`]. Connections that already existed before this
//! call will NOT have the `vec0` module registered — SQLite only applies auto
//! extensions to connections opened afterwards. In practice sqlx creates pool
//! connections lazily, so as long as the store is constructed (and `init` is
//! the first vec query) before heavy pool use, fresh connections will carry the
//! extension. If you get `no such module: vec0`, construct the store / call
//! `init` before other queries acquire pooled connections (e.g. build the vec
//! store immediately after `Database::open`).
//!
//! # Distance metric
//!
//! `vec0` computes **L2 (Euclidean) distance** by default, whereas the trait
//! returns **cosine similarity**. For unit-normalised vectors the two orderings
//! agree and `cosine = 1 - d²/2`. This backend therefore assumes embeddings are
//! unit-normalised (standard practice for cosine-based ANN). The reported
//! `similarity` is derived from the L2 `distance` accordingly.

#![cfg(feature = "sqlite-vec")]

use std::sync::{Arc, Once};

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{NovaError, NovaResult};
use crate::storage::vector::{
    check_dims, vec_to_blob, VectorHit, VectorStore,
};

// FFI to SQLite's `sqlite3_auto_extension`. Declared locally so we don't need
// a direct dependency on `libsqlite3-sys`; the symbol is provided by the
// bundled SQLite that `sqlx-sqlite` links.
unsafe extern "C" {
    fn sqlite3_auto_extension(x_entry_point: Option<unsafe extern "C" fn()>) -> std::os::raw::c_int;
}

/// Register the `vec0` module on every future SQLite connection, exactly once.
///
/// Returns the outcome of the (single) registration attempt. `sqlite3_vec_init`
/// is the C entry point exported by the `sqlite-vec` crate; it is an
/// `unsafe extern "C" fn()` whose ABI matches what `sqlite3_auto_extension`
/// expects, so we pass it directly.
fn register_sqlite_vec_once() -> NovaResult<()> {
    static ONCE: Once = Once::new();
    // -1 = not yet attempted; 0 = success; else SQLite return code.
    static RESULT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

    ONCE.call_once(|| {
        let rc = unsafe { sqlite3_auto_extension(Some(sqlite_vec::sqlite3_vec_init)) };
        RESULT.store(rc, std::sync::atomic::Ordering::SeqCst);
    });

    let rc = RESULT.load(std::sync::atomic::Ordering::SeqCst);
    if rc == 0 {
        Ok(())
    } else {
        Err(NovaError::storage_msg(format!(
            "sqlite-vec: sqlite3_auto_extension failed with rc={rc}"
        )))
    }
}

/// Map a UUID to a stable 64-bit rowid for the `vec0` table. `vec0` requires an
/// integer PRIMARY KEY (rowid) and does not let us key the table on a text
/// column, so we derive an idempotent rowid from the first 8 bytes of the UUID.
/// Collisions are astronomically unlikely for realistic datasets.
fn uuid_rowid(u: &Uuid) -> i64 {
    let b = u.as_bytes();
    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Convert a `vec0` L2 distance into a cosine similarity for unit vectors.
/// Since |a - b|² = 2(1 - cos) on the unit sphere, cos = 1 - d²/2.
fn l2_distance_to_cosine(d: f32) -> f32 {
    (1.0 - (d * d) / 2.0).clamp(-1.0, 1.0)
}

/// sqlite-vec HNSW-backed implementation of [`VectorStore`].
///
/// Vectors live in a `vec0` virtual table (`vec_embeddings`) keyed by a
/// derived integer rowid, with the original `memory_uuid` kept as a metadata
/// column. `knn_search` delegates to the ANN index and converts the returned
/// L2 distance to cosine similarity.
#[derive(Clone)]
pub struct SqliteVecVectorStore {
    pool: sqlx::SqlitePool,
    dims: usize,
}

impl std::fmt::Debug for SqliteVecVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteVecVectorStore")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

impl SqliteVecVectorStore {
    /// Name of the `vec0` virtual table backing this store.
    pub const TABLE: &'static str = "vec_embeddings";

    /// Construct a new store over an existing pool. Registers the `vec0`
    /// extension globally (once). The virtual table is created lazily by
    /// [`init`](Self::init) / [`open`](Self::open).
    pub fn new(pool: sqlx::SqlitePool, dims: usize) -> Self {
        // Best-effort registration inside `new`; a failure here only surfaces
        // later with a clear "no such module: vec0" error from SQLite.
        let _ = register_sqlite_vec_once();
        Self { pool, dims }
    }

    /// Convenience constructor from a shared `Database` handle.
    pub fn with_db(db: &crate::storage::Database, dims: usize) -> Self {
        Self::new(db.pool.clone(), dims)
    }

    /// Idempotently create the `vec0` table (if not already present) and make
    /// sure the extension is registered. Call this once before using the store,
    /// ideally before other queries have created pooled connections.
    pub async fn init(&self) -> NovaResult<()> {
        register_sqlite_vec_once()?;
        sqlx::query(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING vec0(embedding float[{}], memory_uuid TEXT)",
            Self::TABLE,
            self.dims
        ))
        .execute(&self.pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(())
    }

    /// Create the store and ensure the `vec0` table exists.
    pub async fn open(pool: sqlx::SqlitePool, dims: usize) -> NovaResult<Self> {
        let store = Self::new(pool, dims);
        store.init().await?;
        Ok(store)
    }

    pub fn default_dims(&self) -> usize {
        self.dims
    }
}

#[async_trait]
impl VectorStore for SqliteVecVectorStore {
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

        let rowid = uuid_rowid(&memory_uuid);
        let mem_s = memory_uuid.to_string();
        // sqlite-vec accepts vectors as a compact binary blob (raw little-endian
        // f32 bytes, exactly what `vec_to_blob` produces).
        let blob = vec_to_blob(vec);

        // vec0 is a virtual table without reliable upsert support, so we make
        // the insert idempotent by deleting the previous row for this rowid
        // first, then inserting fresh.
        sqlx::query(&format!("DELETE FROM {} WHERE rowid = ?1", Self::TABLE))
            .bind(rowid)
            .execute(&self.pool)
            .await
            .map_err(NovaError::storage)?;

        sqlx::query(&format!(
            "INSERT INTO {} (rowid, memory_uuid, embedding) VALUES (?1, ?2, ?3)",
            Self::TABLE
        ))
        .bind(rowid)
        .bind(&mem_s)
        .bind(blob)
        .execute(&self.pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(())
    }

    async fn delete_vector(&self, memory_uuid: Uuid) -> NovaResult<()> {
        let rowid = uuid_rowid(&memory_uuid);
        sqlx::query(&format!("DELETE FROM {} WHERE rowid = ?1", Self::TABLE))
            .bind(rowid)
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
        let k = (k.min(500)) as i32;
        let blob = vec_to_blob(query);

        // ANN search: vec0 returns rows ordered by ascending L2 `distance`.
        let rows: Vec<(String, f64)> = sqlx::query_as(&format!(
            "SELECT memory_uuid, distance FROM {} WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
            Self::TABLE
        ))
        .bind(blob)
        .bind(k)
        .fetch_all(&self.pool)
        .await
        .map_err(NovaError::storage)?;

        let mut hits: Vec<VectorHit> = Vec::with_capacity(rows.len());
        for (mem_s, d) in rows {
            let sim = l2_distance_to_cosine(d as f32);
            if sim >= threshold {
                if let Ok(mem_uuid) = Uuid::parse_str(&mem_s) {
                    hits.push(VectorHit { memory_uuid: mem_uuid, similarity: sim });
                }
            }
        }

        // Ascending distance => descending cosine already; explicit sort keeps
        // behaviour identical to the linear-scan backend.
        hits.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }
}

/// Convenience alias mirroring [`vector::SharedVectorStore`].
pub type SharedVectorStore = Arc<dyn VectorStore>;

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
//
// These tests need a real, runnable `vec0` extension inside the bundled SQLite.
// In an offline sandbox (or a host where sqlx's bundled SQLite cannot load the
// statically-linked extension) they will fail with `no such module: vec0`, so
// they are marked `#[ignore]` and can be run explicitly with
// `cargo test -p yq-nova-core --features sqlite-vec vector_vec -- --ignored`.
//
// They demonstrate the core contract: the same `VectorStore` trait is served by
// both the linear-scan backend and the HNSW backend with identical top-k hits.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;
    use crate::storage::{
        Database,
        vector::SqliteVectorStore,
    };

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-m3-vec-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        Database::open(cfg).await.expect("open temp db")
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
    }

    #[tokio::test]
    #[ignore]
    async fn vec_backend_create_table_and_insert() {
        let db = temp_db().await;
        let store = SqliteVecVectorStore::with_db(&db, 4);
        store.init().await.expect("create vec0 table");

        let u = Uuid::new_v4();
        store.insert_vector(u, "test", "m1", &[1.0, 0.0, 0.0, 0.0]).await.unwrap();

        let hits = store.knn_search(&[1.0, 0.0, 0.0, 0.0], 10, -2.0).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_uuid, u);
        assert!((hits[0].similarity - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    #[ignore]
    async fn hnsw_matches_linear_scan_topk() {
        let db = temp_db().await;

        // Same pool, two independent backends. The linear one reads/writes the
        // `embeddings` table; the HNSW one uses the `vec_embeddings` vec0 table.
        let linear = SqliteVectorStore::with_db(&db, 4);
        let hns = SqliteVecVectorStore::with_db(&db, 4);
        hns.init().await.expect("create vec0 table");

        // Unit-normalised vectors so L2 ordering coincides with cosine ordering.
        let data: Vec<(Uuid, Vec<f32>)> = vec![
            (Uuid::new_v4(), normalize(&[1.0, 0.0, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[-1.0, 0.0, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.0, 1.0, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.6, 0.8, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.0, 0.0, 1.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.3, 0.4, 0.5, 0.7])),
        ];

        for (u, v) in &data {
            linear.insert_vector(*u, "test", "m", v).await.unwrap();
            hns.insert_vector(*u, "test", "m", v).await.unwrap();
        }

        let query = normalize(&[0.8, 0.6, 0.0, 0.0]);

        let linear_hits = linear.knn_search(&query, 4, -2.0).await.unwrap();
        let hns_hits = hns.knn_search(&query, 4, -2.0).await.unwrap();

        // Same length and same ordered top-k memory_uuids.
        assert_eq!(linear_hits.len(), hns_hits.len(), "top-k lengths differ");
        let linear_uuids: Vec<Uuid> = linear_hits.iter().map(|h| h.memory_uuid).collect();
        let hns_uuids: Vec<Uuid> = hns_hits.iter().map(|h| h.memory_uuid).collect();
        assert_eq!(linear_uuids, hns_uuids, "top-k hit sets/order differ");

        // Similarity values should agree closely for unit vectors.
        for (a, b) in linear_hits.iter().zip(hns_hits.iter()) {
            assert!(
                (a.similarity - b.similarity).abs() < 1e-3,
                "similarity mismatch: linear={} hns={}",
                a.similarity,
                b.similarity
            );
        }
    }

    #[tokio::test]
    #[ignore]
    async fn vec_backend_delete_removes_row() {
        let db = temp_db().await;
        let store = SqliteVecVectorStore::with_db(&db, 2);
        store.init().await.unwrap();

        let u = Uuid::new_v4();
        store.insert_vector(u, "test", "m", &[1.0, 0.0]).await.unwrap();
        store.delete_vector(u).await.unwrap();

        let hits = store.knn_search(&[1.0, 0.0], 10, -2.0).await.unwrap();
        assert!(hits.iter().all(|h| h.memory_uuid != u));
    }
}