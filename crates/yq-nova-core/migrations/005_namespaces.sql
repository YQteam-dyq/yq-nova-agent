CREATE TABLE IF NOT EXISTS namespaces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid        TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT,
    config_json TEXT    NOT NULL DEFAULT '{}',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

INSERT OR IGNORE INTO namespaces (uuid, name, description, config_json, created_at, updated_at)
VALUES ('00000000-0000-0000-0000-000000000001', 'default', NULL, '{}', strftime('%s','now'), strftime('%s','now'));

-- =============================================================
-- Rebuild memory_items so it carries a namespace_id foreign key.
-- SQLite cannot add a FK via ALTER TABLE, so we drop every table
-- that references memory_items (embeddings + full-text triggers),
-- rebuild memory_items with the FK, then restore the dependencies.
-- Embedding rows and the FTS index are preserved across the rebuild.
-- =============================================================

-- Back up existing embedding rows before the reference tables are dropped,
-- so no stored vectors are lost during the upgrade.
DROP TABLE IF EXISTS embeddings_bak;
CREATE TABLE embeddings_bak AS
    SELECT memory_uuid, dims, provider, model, vec_blob, created_at FROM embeddings;

-- Sever the full-text sync triggers and the embeddings FK dependency
-- first so memory_items can be dropped cleanly.
ALTER TABLE memory_items ADD COLUMN namespace_id INTEGER NOT NULL DEFAULT 1;
DROP TABLE IF EXISTS memory_fts;
DROP TABLE IF EXISTS embeddings;

CREATE TABLE memory_items_new (
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
    status          TEXT    NOT NULL DEFAULT 'active',
    namespace_id    INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id) ON DELETE CASCADE
);
INSERT INTO memory_items_new (id, uuid, content, content_hash, metadata_json, source, importance, access_count, last_accessed, created_at, expires_at, status, namespace_id)
SELECT id, uuid, content, content_hash, metadata_json, source, importance, access_count, last_accessed, created_at, expires_at, status, namespace_id FROM memory_items;

DROP TABLE memory_items;
ALTER TABLE memory_items_new RENAME TO memory_items;

-- memory_items indexes (recreated after the rebuild)
CREATE INDEX IF NOT EXISTS idx_memory_created_at  ON memory_items(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_expires_at  ON memory_items(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_status      ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_memory_content_hash ON memory_items(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_importance  ON memory_items(importance);
CREATE INDEX IF NOT EXISTS idx_memory_source      ON memory_items(source);
CREATE INDEX IF NOT EXISTS idx_memory_last_accessed ON memory_items(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memory_status_importance ON memory_items(status, importance);
CREATE INDEX IF NOT EXISTS idx_memory_status_source_created ON memory_items(status, source, created_at);

-- embeddings (restore with FK back to the rebuilt memory_items, preserving data)
CREATE TABLE IF NOT EXISTS embeddings (
    memory_uuid     TEXT    NOT NULL PRIMARY KEY,
    dims            INTEGER NOT NULL,
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    vec_blob        BLOB    NOT NULL,
    created_at      INTEGER NOT NULL,
    FOREIGN KEY (memory_uuid) REFERENCES memory_items(uuid) ON DELETE CASCADE
);
INSERT OR IGNORE INTO embeddings (memory_uuid, dims, provider, model, vec_blob, created_at)
SELECT memory_uuid, dims, provider, model, vec_blob, created_at FROM embeddings_bak;
DROP TABLE embeddings_bak;
CREATE INDEX IF NOT EXISTS idx_embeddings_dims ON embeddings(dims);
CREATE INDEX IF NOT EXISTS idx_embeddings_dims_provider ON embeddings(dims, provider);

-- full-text search index + sync triggers (restore for rebuilt memory_items)
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content       = 'memory_items',
    content_rowid = 'id',
    tokenize      = "unicode61 remove_diacritics 2 tokenchars '_'"
);

CREATE TRIGGER IF NOT EXISTS memory_fts_ai
AFTER INSERT ON memory_items
FOR EACH ROW BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_au
AFTER UPDATE OF content ON memory_items
FOR EACH ROW BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_ad
AFTER DELETE ON memory_items
FOR EACH ROW BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
        VALUES('delete', old.id, old.content);
END;

-- Rebuild the FTS index from the restored memory_items rows so keyword
-- recall also finds memories that existed before this migration.
INSERT INTO memory_fts(memory_fts) VALUES('rebuild');

-- =============================================================
-- The remaining tenant tables are namespace-aware.
-- =============================================================

ALTER TABLE entities ADD COLUMN namespace_id INTEGER NOT NULL DEFAULT 1;
ALTER TABLE relations ADD COLUMN namespace_id INTEGER NOT NULL DEFAULT 1;
ALTER TABLE tags ADD COLUMN namespace_id INTEGER NOT NULL DEFAULT 1;
ALTER TABLE memory_tags ADD COLUMN namespace_id INTEGER NOT NULL DEFAULT 1;

CREATE TABLE entities_new (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid            TEXT    NOT NULL UNIQUE,
    namespace_id    INTEGER NOT NULL DEFAULT 1,
    name            TEXT    NOT NULL,
    type            TEXT    NOT NULL DEFAULT 'unknown',
    description     TEXT,
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    UNIQUE(namespace_id, name, type),
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id) ON DELETE CASCADE
);
INSERT INTO entities_new (id, uuid, namespace_id, name, type, description, metadata_json, created_at, updated_at)
SELECT id, uuid, namespace_id, name, type, description, metadata_json, created_at, updated_at FROM entities;

CREATE TABLE relations_new (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid            TEXT    NOT NULL UNIQUE,
    namespace_id    INTEGER NOT NULL DEFAULT 1,
    source_uuid     TEXT    NOT NULL,
    target_uuid     TEXT    NOT NULL,
    predicate       TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    memory_uuid     TEXT,
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    created_at      INTEGER NOT NULL,
    FOREIGN KEY (source_uuid) REFERENCES entities_new(uuid) ON DELETE CASCADE,
    FOREIGN KEY (target_uuid) REFERENCES entities_new(uuid) ON DELETE CASCADE,
    FOREIGN KEY (memory_uuid) REFERENCES memory_items(uuid) ON DELETE SET NULL,
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id) ON DELETE CASCADE
);
INSERT INTO relations_new (id, uuid, namespace_id, source_uuid, target_uuid, predicate, confidence, memory_uuid, metadata_json, created_at)
SELECT id, uuid, namespace_id, source_uuid, target_uuid, predicate, confidence, memory_uuid, metadata_json, created_at FROM relations;
DROP TABLE relations;
DROP TABLE entities;
ALTER TABLE entities_new RENAME TO entities;
ALTER TABLE relations_new RENAME TO relations;

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_created_type ON entities(created_at, type);
CREATE INDEX IF NOT EXISTS idx_rel_source    ON relations(source_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_target    ON relations(target_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relations(predicate);
CREATE INDEX IF NOT EXISTS idx_rel_memory    ON relations(memory_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_confidence ON relations(confidence);
CREATE INDEX IF NOT EXISTS idx_rel_src_pred      ON relations(source_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_rel_tgt_pred      ON relations(target_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_rel_memory_source ON relations(memory_uuid, source_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_memory_target ON relations(memory_uuid, target_uuid);

CREATE TABLE tags_new (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace_id INTEGER NOT NULL DEFAULT 1,
    name        TEXT    NOT NULL,
    color       TEXT,
    created_at  INTEGER NOT NULL,
    UNIQUE(namespace_id, name),
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id) ON DELETE CASCADE
);
INSERT INTO tags_new (id, namespace_id, name, color, created_at)
SELECT id, namespace_id, name, color, created_at FROM tags;

CREATE TABLE memory_tags_new (
    memory_uuid     TEXT    NOT NULL,
    tag_id          INTEGER NOT NULL,
    namespace_id    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (memory_uuid, tag_id),
    FOREIGN KEY (memory_uuid) REFERENCES memory_items(uuid) ON DELETE CASCADE,
    FOREIGN KEY (tag_id)      REFERENCES tags_new(id)     ON DELETE CASCADE,
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id)  ON DELETE CASCADE
);
INSERT INTO memory_tags_new (memory_uuid, tag_id, namespace_id)
SELECT memory_uuid, tag_id, namespace_id FROM memory_tags;
DROP TABLE memory_tags;
DROP TABLE tags;
ALTER TABLE tags_new RENAME TO tags;
ALTER TABLE memory_tags_new RENAME TO memory_tags;

CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_memory_tags_reverse ON memory_tags(tag_id, memory_uuid);

CREATE INDEX IF NOT EXISTS idx_memory_items_namespace ON memory_items(namespace_id);
CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace_id);
CREATE INDEX IF NOT EXISTS idx_entities_namespace_name_type ON entities(namespace_id, name, type);
CREATE INDEX IF NOT EXISTS idx_relations_namespace ON relations(namespace_id);
CREATE INDEX IF NOT EXISTS idx_relations_namespace_src ON relations(namespace_id, source_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_relations_namespace_tgt ON relations(namespace_id, target_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag_namespace ON memory_tags(namespace_id, tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_namespace ON tags(namespace_id, name);
CREATE INDEX IF NOT EXISTS idx_memory_tags_namespace_uuid ON memory_tags(namespace_id, memory_uuid, tag_id);