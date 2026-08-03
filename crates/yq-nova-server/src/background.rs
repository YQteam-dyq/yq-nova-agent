//! Background jobs: TTL expiry + staleness/forgetting policy + cron ticker.
//!
//! Jobs don't run in-process in tests (they require tokio runtime + a real DB
//! pool). Instead each job exposes `run_once(...)` as a deterministic async
//! function that the scheduler calls each tick; tests call `run_once` directly
//! so they don't have to wall-clock sleep.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use chrono::{DateTime, Utc};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use yq_nova_core::{
    config::ForgettingConfig,
    error::NovaResult,
    memory::{ForgetInput, ForgetMode, MemoryService, ops_forget},
    storage::{MemoryFilter, MemoryStatus},
};

/// Returned by each `run_once` so logs + metrics (future) can describe what
/// actually happened.
#[derive(Debug, Default, Clone)]
pub struct JobStats {
    pub ttl_expired: u64,
    pub stale_archived: u64,
    pub stale_deleted: u64,
}

// ============== TTL expiry job ==============================================

/// Mark `memory_items` whose `expires_at < now` AND are currently `Active` as
/// `Expired`. Runs in a single SQL UPDATE so it's efficient even if tens of
/// thousands of rows expire at once.
pub async fn expire_ttl_once(memory: &MemoryService) -> NovaResult<u64> {
    let now_secs = Utc::now().timestamp();
    let pool = &memory.database.pool;
    let affected = sqlx::query(
        r#"
        UPDATE memory_items
           SET status = ?
         WHERE status = ?
           AND expires_at IS NOT NULL
           AND expires_at < ?
        "#,
    )
    .bind(MemoryStatus::Expired.as_str())
    .bind(MemoryStatus::Active.as_str())
    .bind(now_secs)
    .execute(pool)
    .await
    .map_err(yq_nova_core::NovaError::storage)?
    .rows_affected();
    if affected > 0 {
        info!(count = affected, "ttl: expired rows by expires_at");
    }
    Ok(affected)
}

// ============== Staleness / forgetting policy job ============================

/// Applies `ForgettingConfig`: memories whose `last_accessed` is older than
/// `stale_after` *and* whose `importance` is below
/// `stale_importance_threshold` get archived or deleted.
pub async fn collect_garbage_once(
    memory: &MemoryService,
    cfg: &ForgettingConfig,
) -> NovaResult<(u64, u64)> {
    if !cfg.enabled {
        return Ok((0, 0));
    }
    // Treat "never accessed" as "last_accessed = created_at" by using the
    // COALESCE filter built into MemoryFilter.
    let stale_before: DateTime<Utc> = Utc::now() - cfg.stale_after;
    let filter = MemoryFilter {
        status_in: Some(vec![MemoryStatus::Active]),
        last_accessed_before: Some(stale_before),
        importance_max: Some(cfg.stale_importance_threshold),
        ..Default::default()
    };
    let mode = if cfg.action.eq_ignore_ascii_case("delete") {
        ForgetMode::Hard
    } else {
        ForgetMode::Archive
    };
    let input = ForgetInput {
        target: ops_forget::ForgetTarget::Filter(filter),
        mode,
        gc_graph: false,
        batch_limit: 1000,
    };
    let out = memory.forget(input).await?;
    let affected = out.affected_memories as u64;
    let (archived, deleted) = match mode {
        ForgetMode::Archive => (affected, 0),
        ForgetMode::Hard => (0, affected),
        ForgetMode::Soft => (0, 0),
    };
    if archived > 0 || deleted > 0 {
        info!(archived, deleted, "gc: applied forgetting policy");
    }
    Ok((archived, deleted))
}

// ============== Ticker / scheduler ==========================================

/// Drives `expire_ttl_once` + `collect_garbage_once` in a loop.
///
/// Exits cleanly when `cancel` is triggered (SIGINT/SIGTERM from main.rs).
pub async fn run_job_loop(
    memory: MemoryService,
    forgetting_cfg: ForgettingConfig,
    ttl_interval: Duration,
    cancel: CancellationToken,
) -> JobStats {
    let mut stats = JobStats::default();
    let mut tick = tokio::time::interval(ttl_interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Concurrency guard: if a single tick's work (e.g. forgetting over 100k
    // rows) takes longer than `ttl_interval`, we *skip* subsequent ticks until
    // the in-flight run finishes. 1 permit ensures at most 1 run at a time.
    let busy = Arc::new(Semaphore::new(1));

    // GC cadence is `forgetting_cfg.check_interval` (default 600s), which is
    // typically much slower than the TTL ticker.  Rather than spawn a second
    // independent task (which adds shutdown bookkeeping), we simply remember
    // the last time GC ran and compare on every fast tick.
    let gc_interval = forgetting_cfg.check_interval;
    let mut last_gc: Option<Instant> = None;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("job scheduler: shutdown requested, exiting loop");
                break stats;
            }
            _ = tick.tick() => {
                let permit = match busy.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => {
                        warn!("job scheduler: previous tick still running, skip this one");
                        continue;
                    }
                };
                // Release the permit when the tick body returns (even on panic
                // via Semaphore's RAII guard).
                let _permit_guard = permit;

                // ---- TTL (always runs, cheap SQL single UPDATE) ----
                let started = std::time::SystemTime::now();
                match expire_ttl_once(&memory).await {
                    Ok(n) => stats.ttl_expired += n,
                    Err(e) => warn!(error = %e, "ttl job failed"),
                }
                if let Ok(d) = started.elapsed() {
                    debug!(job = "ttl", took_ms = d.as_millis() as u64, "tick finished");
                }

                // ---- GC (runs at forgetting_cfg.check_interval cadence) ----
                let gc_due = match last_gc {
                    None => true,
                    Some(t) => t.elapsed() >= gc_interval,
                };
                if gc_due {
                    let started = std::time::SystemTime::now();
                    match collect_garbage_once(&memory, &forgetting_cfg).await {
                        Ok((a, d)) => {
                            stats.stale_archived += a;
                            stats.stale_deleted += d;
                        }
                        Err(e) => warn!(error = %e, "gc job failed"),
                    }
                    last_gc = Some(Instant::now());
                    if let Ok(d) = started.elapsed() {
                        info!(
                            job = "gc",
                            took_ms = d.as_millis() as u64,
                            archived = stats.stale_archived,
                            deleted = stats.stale_deleted,
                            "gc tick finished"
                        );
                    }
                }

                debug!(?stats, "job tick complete");
            }
        }
    }
}

/// Thin wrapper: spawn the job loop on the tokio runtime. Returns a
/// `JoinHandle<JobStats>` that resolves once cancellation finishes. Caller
/// awaits it after the HTTP server joins.
pub fn spawn_job_loop(
    memory: MemoryService,
    forgetting_cfg: ForgettingConfig,
    ttl_interval: Duration,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<JobStats> {
    tokio::spawn(run_job_loop(memory, forgetting_cfg, ttl_interval, cancel))
}

/// Simple helper so callers can trivially construct a CancellationToken.
pub fn new_cancel_token() -> CancellationToken {
    CancellationToken::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration as ChronoDur, Utc};
    use std::{sync::Arc, time::Duration as StdDuration};
    use yq_nova_core::{
        config::StorageConfig,
        embedding::MockEmbeddingProvider,
        memory::{RecallInput, RememberInput, SearchMode},
        storage::{Database, MemoryRepository, MemoryStatus, SqliteMemoryRepository},
    };

    fn tmp_db_path(tag: &str) -> StorageConfig {
        let mut p = std::env::temp_dir();
        p.push(format!("yqnova-m6-{tag}-{}-{}.db", std::process::id(), randish()));
        StorageConfig {
            db_path: p,
            wal_mode: true,
            page_size: 4096,
            cache_size_kb: 32_000,
            busy_timeout_ms: 5000,
            pool_max_connections: 4,
            pool_min_connections: 0,
            ..Default::default()
        }
    }

    fn randish() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        n ^ (n >> 13) ^ 0x9e3779b97f4a7c15
    }

    async fn setup(tag: &str) -> (Database, MemoryService) {
        let cfg = tmp_db_path(tag);
        let db = Database::open(cfg).await.expect("open db");
        let provider: yq_nova_core::embedding::SharedEmbeddingProvider =
            Arc::new(MockEmbeddingProvider::new(64));
        let memory = MemoryService::new(db.clone(), provider);
        (db, memory)
    }

    #[tokio::test]
    async fn expire_ttl_marks_rows_expired() {
        let (db, memory) = setup("ttl").await;

        let repo = SqliteMemoryRepository::new();
        // Remember 3 items, then manually backdate 2 of them to have expired.
        let mut uuids = Vec::new();
        for (i, content) in ["a", "b", "c"].into_iter().enumerate() {
            let out = memory
                .remember(RememberInput {
                    content,
                    expires_at: Some(Utc::now() + ChronoDur::hours(1)),
                    ..Default::default()
                })
                .await
                .expect("remember");
            uuids.push(out.uuid);
            if i < 2 {
                sqlx::query("UPDATE memory_items SET expires_at = ? WHERE uuid = ?")
                    .bind((Utc::now() - ChronoDur::minutes(1)).timestamp())
                    .bind(uuids[i].to_string())
                    .execute(&db.pool)
                    .await
                    .unwrap();
            }
        }

        let expired = expire_ttl_once(&memory).await.unwrap();
        assert_eq!(expired, 2);

        // No repeat work on second run.
        let again = expire_ttl_once(&memory).await.unwrap();
        assert_eq!(again, 0);

        // Direct status check.
        for (i, u) in uuids.iter().enumerate() {
            let rec = repo.get_by_uuid(&db, *u).await.expect("get");
            let want = if i < 2 { MemoryStatus::Expired } else { MemoryStatus::Active };
            assert_eq!(rec.status, want, "i={i}");
        }

        // Recall excludes expired. Use a wide filter so status != deleted still
        // includes Active, but we filter to Active explicitly to exclude Expired.
        let f = yq_nova_core::storage::MemoryFilter {
            status_in: Some(vec![yq_nova_core::storage::MemoryStatus::Active]),
            ..Default::default()
        };
        let hits = memory
            .recall(RecallInput {
                query: "a",
                top_k: 10,
                mode: SearchMode::Semantic,
                filter: f,
                ..Default::default()
            })
            .await
            .expect("recall");
        assert_eq!(hits.hits.len(), 1);
    }

    #[tokio::test]
    async fn gc_archives_stale_low_importance() {
        let (_db, memory) = setup("gc-archive").await;
        let u = memory
            .remember(RememberInput {
                content: "dusty old note",
                importance: 0.1,
                ..Default::default()
            })
            .await
            .unwrap()
            .uuid;

        // Force last_accessed way into the past (never accessed = created_at,
        // so backdate created_at too so the staleness window catches it).
        sqlx::query("UPDATE memory_items SET last_accessed = ?, created_at = ? WHERE uuid = ?")
            .bind((Utc::now() - ChronoDur::days(120)).timestamp())
            .bind((Utc::now() - ChronoDur::days(120)).timestamp())
            .bind(u.to_string())
            .execute(&memory.database.pool)
            .await
            .unwrap();

        let cfg = ForgettingConfig {
            enabled: true,
            stale_after: ChronoDur::days(90).to_std().unwrap(),
            stale_importance_threshold: 0.3,
            action: "archive".into(),
            check_interval: StdDuration::from_secs(1),
        };
        let (archived, deleted) = collect_garbage_once(&memory, &cfg).await.unwrap();
        assert_eq!(archived, 1);
        assert_eq!(deleted, 0);

        // Second run is a no-op because status is no longer Active.
        let (a2, d2) = collect_garbage_once(&memory, &cfg).await.unwrap();
        assert_eq!((a2, d2), (0, 0));
    }

    #[tokio::test]
    async fn gc_skipped_when_disabled() {
        let (_db, memory) = setup("gc-disabled").await;
        memory
            .remember(RememberInput { content: "hi", importance: 0.05, ..Default::default() })
            .await
            .unwrap();

        let cfg = ForgettingConfig { stale_after: StdDuration::from_secs(1), ..Default::default() };
        assert_eq!(collect_garbage_once(&memory, &cfg).await.unwrap(), (0, 0));
    }

    #[tokio::test]
    async fn gc_high_importance_never_forgotten_even_if_stale() {
        let (_db, memory) = setup("gc-high-imp").await;
        let u = memory
            .remember(RememberInput {
                content: "important!",
                importance: 0.95,
                ..Default::default()
            })
            .await
            .unwrap()
            .uuid;
        sqlx::query("UPDATE memory_items SET last_accessed = ?, created_at = ? WHERE uuid = ?")
            .bind((Utc::now() - ChronoDur::days(365)).timestamp())
            .bind((Utc::now() - ChronoDur::days(365)).timestamp())
            .bind(u.to_string())
            .execute(&memory.database.pool)
            .await
            .unwrap();

        let cfg = ForgettingConfig {
            enabled: true,
            stale_after: ChronoDur::days(30).to_std().unwrap(),
            stale_importance_threshold: 0.3,
            action: "delete".into(),
            check_interval: StdDuration::from_secs(1),
        };
        let (a, d) = collect_garbage_once(&memory, &cfg).await.unwrap();
        assert_eq!((a, d), (0, 0));
    }

    /// Verify `Database::close()` writes everything back to the main DB file.
    /// This is the closest integration test we can do without actually
    /// spawning a unix process + sending SIGTERM: write 10 memories,
    /// close the DB, drop every handle, reopen the same file path,
    /// and confirm all 10 are still readable via recall.
    #[tokio::test]
    async fn shutdown_close_db_preserves_data() {
        use yq_nova_core::storage::Database;

        let storage_cfg = tmp_db_path("shutdown-close");
        let path = storage_cfg.db_path.clone();
        let mem_storage = storage_cfg.clone();
        {
            let db = Database::open(storage_cfg.clone()).await.unwrap();
            let provider: yq_nova_core::embedding::SharedEmbeddingProvider =
                Arc::new(MockEmbeddingProvider::new(64));
            let mem = MemoryService::new(db.clone(), provider);
            let mut contents: Vec<String> = (0..10).map(|i| format!("memory #{i}")).collect();
            for (i, c) in contents.iter_mut().enumerate() {
                mem.remember(RememberInput {
                    content: c.as_str(),
                    importance: (i as f32) / 10.0 + 0.1,
                    ..Default::default()
                })
                .await
                .unwrap();
            }
            db.close().await.unwrap();
        }

        // Reopen the *same* DB path with a fresh pool and count memories.
        let db2 = Database::open(mem_storage).await.unwrap();
        let cnt: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_items WHERE status = ? AND content LIKE 'memory #%'",
        )
        .bind(MemoryStatus::Active.as_str())
        .fetch_one(&db2.pool)
        .await
        .unwrap();
        assert_eq!(cnt, 10);
        db2.close().await.unwrap();
        // Silence unused: path kept for debug only
        let _ = path;
    }

    /// The scheduler loop uses a `Semaphore(1)` guard so a slow tick never
    /// stacks onto the previous one.  Make sure the second `try_acquire_owned`
    /// on the same Semaphore returns Err immediately (this is what drives the
    /// "skip" branch in run_job_loop).
    #[tokio::test]
    async fn busy_guard_skips_next_tick_when_inflight() {
        use tokio::sync::Semaphore;
        let busy = Arc::new(Semaphore::new(1));
        let _held = busy.clone().acquire_owned().await.unwrap();
        assert!(
            busy.clone().try_acquire_owned().is_err(),
            "second acquire must fail so scheduler skip branch activates"
        );
    }
}
