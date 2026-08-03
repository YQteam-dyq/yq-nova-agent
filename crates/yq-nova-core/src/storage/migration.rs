//! Migration runner backed by `sqlx::migrate!()`.
//!
//! All `.sql` files under `migrations/` in the workspace root are embedded
//! at compile time. The standard `_sqlx_migrations` table tracks applied
//! versions so `run()` is fully idempotent even across processes.

use async_trait::async_trait;
use sqlx::SqlitePool;

use crate::error::NovaResult;

pub struct Migrator;

impl Migrator {
    /// Apply all pending migrations, idempotently.
    ///
    /// `sqlx::migrate!()` embeds the migrations directory at compile time
    /// and uses the built-in `_sqlx_migrations` bookkeeping table.
    pub async fn run(pool: &SqlitePool) -> NovaResult<()> {
        let migrator = sqlx::migrate!("../../migrations");
        migrator.run(pool).await.map_err(crate::error::NovaError::storage)?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Unused `Repository` impl just to satisfy the async_trait dep import.
// -----------------------------------------------------------------------------

#[async_trait]
impl crate::storage::Repository<()> for Migrator {
    fn name(&self) -> &'static str {
        "migrator"
    }
}
