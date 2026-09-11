use async_trait::async_trait;
use sqlx::SqlitePool;

use crate::error::NovaResult;

pub struct Migrator;

impl Migrator {
    pub async fn run(pool: &SqlitePool) -> NovaResult<()> {
        let migrator = sqlx::migrate!("./migrations");
        migrator.run(pool).await.map_err(crate::error::NovaError::storage)?;
        Ok(())
    }
}

#[async_trait]
impl crate::storage::Repository<()> for Migrator {
    fn name(&self) -> &'static str {
        "migrator"
    }
}
