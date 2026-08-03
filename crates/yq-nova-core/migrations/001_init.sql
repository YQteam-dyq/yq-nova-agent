-- =============================================================
-- yq-nova-agent migration 001
-- Initial schema: memory_items + entities + relations + tags
-- + embedding metadata + system_config + memory_fts
-- =============================================================
-- NOTE: sqlx migration runner looks for the special `--!` magic
-- comments below. Do not change their format.

-- CREATE TABLE memory_items ------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid            TEXT    NOT NULL UNIQUE,
    content         TEXT    NOT NULL,
    content_hash    TEXT    NOT NULL,
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    source          TEXT    NOT NULL DEFAULT 'agent',
    importance      REAL    NOT NULL DEFAULT 0.5,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   INTEGER,
    created_at      INTEGER NOT NULL,
    expires_at      INTEGER,
    status          TEXT    NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_memory_created_at  ON memory_items(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_expires_at  ON memory_items(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_status      ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_memory_content_hash ON memory_items(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_importance  ON memory_items(importance);
CREATE INDEX IF NOT EXISTS idx_memory_source      ON memory_items(source);
CREATE INDEX IF NOT EXISTS idx_memory_last_accessed ON memory_items(last_accessed);

-- CREATE TABLE entities -----------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid            TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    type            TEXT    NOT NULL DEFAULT 'unknown',
    description     TEXT,
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    UNIQUE(name, type)
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

-- CREATE TABLE relations ----------------------------------------------------

CREATE TABLE IF NOT EXISTS relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid            TEXT    NOT NULL UNIQUE,
    source_uuid     TEXT    NOT NULL,
    target_uuid     TEXT    NOT NULL,
    predicate       TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    memory_uuid     TEXT,
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    created_at      INTEGER NOT NULL,
    FOREIGN KEY (source_uuid) REFERENCES entities(uuid) ON DELETE CASCADE,
    FOREIGN KEY (target_uuid) REFERENCES entities(uuid) ON DELETE CASCADE,
    FOREIGN KEY (memory_uuid) REFERENCES memory_items(uuid) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_rel_source    ON relations(source_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_target    ON relations(target_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relations(predicate);
CREATE INDEX IF NOT EXISTS idx_rel_memory    ON relations(memory_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_confidence ON relations(confidence);

-- CREATE TABLE tags / memory_tags -------------------------------------------

CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    color       TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_tags (
    memory_uuid TEXT    NOT NULL,
    tag_id      INTEGER NOT NULL,
    PRIMARY KEY (memory_uuid, tag_id),
    FOREIGN KEY (memory_uuid) REFERENCES memory_items(uuid) ON DELETE CASCADE,
    FOREIGN KEY (tag_id)      REFERENCES tags(id)         ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id);

-- CREATE TABLE embedding_metadata -------------------------------------------
-- Note: the raw vector data lives in a SQLite BLOB column `embeddings.vec_blob`.
-- We intentionally avoid depending on sqlite-vec FTS at schema-build time so
-- the project compiles with a plain sqlx SQLite build. The optional
-- `sqlite-vec` feature gate will create a vec0 virtual table on top of this
-- metadata in a later migration.

CREATE TABLE IF NOT EXISTS embeddings (
    memory_uuid     TEXT    NOT NULL PRIMARY KEY,
    dims            INTEGER NOT NULL,
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    -- little-endian f32 blob, length = dims * 4 bytes. MVP does linear
    -- scan cosine similarity; switch to sqlite-vec HNSW when ready.
    vec_blob        BLOB    NOT NULL,
    created_at      INTEGER NOT NULL,
    FOREIGN KEY (memory_uuid) REFERENCES memory_items(uuid) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_dims ON embeddings(dims);

-- CREATE TABLE system_config ------------------------------------------------

CREATE TABLE IF NOT EXISTS system_config (
    key         TEXT PRIMARY KEY,
    value_json  TEXT    NOT NULL,
    updated_at  INTEGER NOT NULL
);

INSERT OR IGNORE INTO system_config (key, value_json, updated_at)
VALUES ('schema_version', '"1"', strftime('%s', 'now'));
INSERT OR IGNORE INTO system_config (key, value_json, updated_at)
VALUES ('created_at', printf('"%s"', datetime('now')), strftime('%s', 'now'));
