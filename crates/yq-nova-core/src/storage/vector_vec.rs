#![cfg(feature = "sqlite-vec")]

use std::sync::{Arc, Once};

use async_trait::async_trait;
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::vector::{VectorHit, VectorStore, check_dims, vec_to_blob},
};

unsafe extern "C" {
    fn sqlite3_auto_extension(x_entry_point: Option<unsafe extern "C" fn()>)
    -> std::os::raw::c_int;
}

fn register_sqlite_vec_once() -> NovaResult<()> {
    static ONCE: Once = Once::new();

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

fn uuid_rowid(u: &Uuid) -> i64 {
    let b = u.as_bytes();
    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

fn l2_distance_to_cosine(d: f32) -> f32 {
    (1.0 - (d * d) / 2.0).clamp(-1.0, 1.0)
}

#[derive(Clone)]
pub struct SqliteVecVectorStore {
    pool: sqlx::SqlitePool,
    dims: usize,
}

impl std::fmt::Debug for SqliteVecVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteVecVectorStore").field("dims", &self.dims).finish_non_exhaustive()
    }
}

impl SqliteVecVectorStore {
    pub const TABLE: &'static str = "vec_embeddings";

    pub fn new(pool: sqlx::SqlitePool, dims: usize) -> Self {
        let _ = register_sqlite_vec_once();
        Self {
            pool,
            dims,
        }
    }

    pub fn with_db(db: &crate::storage::Database, dims: usize) -> Self {
        Self::new(db.pool.clone(), dims)
    }

    pub async fn init(&self) -> NovaResult<()> {
        register_sqlite_vec_once()?;
        sqlx::query(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING vec0(embedding float[{}], memory_uuid \
             TEXT)",
            Self::TABLE,
            self.dims
        ))
        .execute(&self.pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(())
    }

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
        namespace_id: i64,
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

        let mem_s = memory_uuid.to_string();
        let owns: Option<(i64,)> =
            sqlx::query_as("SELECT 1 FROM memory_items WHERE uuid = ?1 AND namespace_id = ?2")
                .bind(&mem_s)
                .bind(namespace_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(NovaError::storage)?;
        if owns.is_none() {
            return Err(NovaError::not_found(format!("memory {memory_uuid} in namespace")));
        }

        let rowid = uuid_rowid(&memory_uuid);
        let blob = vec_to_blob(vec);

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

    async fn delete_vector(&self, namespace_id: i64, memory_uuid: Uuid) -> NovaResult<()> {
        let mem_s = memory_uuid.to_string();
        let owns: Option<(i64,)> =
            sqlx::query_as("SELECT 1 FROM memory_items WHERE uuid = ?1 AND namespace_id = ?2")
                .bind(&mem_s)
                .bind(namespace_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(NovaError::storage)?;
        if owns.is_some() {
            let rowid = uuid_rowid(&memory_uuid);
            sqlx::query(&format!("DELETE FROM {} WHERE rowid = ?1", Self::TABLE))
                .bind(rowid)
                .execute(&self.pool)
                .await
                .map_err(NovaError::storage)?;
        }
        Ok(())
    }

    async fn knn_search(
        &self,
        namespace_id: i64,
        query: &[f32],
        k: usize,
        threshold: f32,
    ) -> NovaResult<Vec<VectorHit>> {
        check_dims(self.dims, query)?;
        let k = (k.min(500)) as i32;
        let blob = vec_to_blob(query);

        let rows: Vec<(String, f64)> = sqlx::query_as(&format!(
            "SELECT memory_uuid, distance FROM {} WHERE memory_uuid IN (SELECT uuid FROM \
             memory_items WHERE namespace_id = ?1) AND embedding MATCH ?2 AND k = ?3",
            Self::TABLE
        ))
        .bind(namespace_id)
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
                    hits.push(VectorHit {
                        memory_uuid: mem_uuid,
                        similarity: sim,
                    });
                }
            }
        }

        hits.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }
}

pub type SharedVectorStore = Arc<dyn VectorStore>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::StorageConfig,
        storage::{Database, vector::SqliteVectorStore},
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

    const NS: i64 = crate::storage::namespace::DEFAULT_NAMESPACE_ID;

    #[tokio::test]
    #[ignore]
    async fn vec_backend_create_table_and_insert() {
        let db = temp_db().await;
        let store = SqliteVecVectorStore::with_db(&db, 4);
        store.init().await.expect("create vec0 table");

        let u = Uuid::new_v4();
        store.insert_vector(NS, u, "test", "m1", &[1.0, 0.0, 0.0, 0.0]).await.unwrap();

        let hits = store.knn_search(NS, &[1.0, 0.0, 0.0, 0.0], 10, -2.0).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_uuid, u);
        assert!((hits[0].similarity - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    #[ignore]
    async fn hnsw_matches_linear_scan_topk() {
        let db = temp_db().await;

        let linear = SqliteVectorStore::with_db(&db, 4);
        let hns = SqliteVecVectorStore::with_db(&db, 4);
        hns.init().await.expect("create vec0 table");

        let data: Vec<(Uuid, Vec<f32>)> = vec![
            (Uuid::new_v4(), normalize(&[1.0, 0.0, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[-1.0, 0.0, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.0, 1.0, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.6, 0.8, 0.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.0, 0.0, 1.0, 0.0])),
            (Uuid::new_v4(), normalize(&[0.3, 0.4, 0.5, 0.7])),
        ];

        for (u, v) in &data {
            linear.insert_vector(NS, *u, "test", "m", v).await.unwrap();
            hns.insert_vector(NS, *u, "test", "m", v).await.unwrap();
        }

        let query = normalize(&[0.8, 0.6, 0.0, 0.0]);

        let linear_hits = linear.knn_search(NS, &query, 4, -2.0).await.unwrap();
        let hns_hits = hns.knn_search(NS, &query, 4, -2.0).await.unwrap();

        assert_eq!(linear_hits.len(), hns_hits.len(), "top-k lengths differ");
        let linear_uuids: Vec<Uuid> = linear_hits.iter().map(|h| h.memory_uuid).collect();
        let hns_uuids: Vec<Uuid> = hns_hits.iter().map(|h| h.memory_uuid).collect();
        assert_eq!(linear_uuids, hns_uuids, "top-k hit sets/order differ");

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
        store.insert_vector(NS, u, "test", "m", &[1.0, 0.0]).await.unwrap();
        store.delete_vector(NS, u).await.unwrap();

        let hits = store.knn_search(NS, &[1.0, 0.0], 10, -2.0).await.unwrap();
        assert!(hits.iter().all(|h| h.memory_uuid != u));
    }
}
