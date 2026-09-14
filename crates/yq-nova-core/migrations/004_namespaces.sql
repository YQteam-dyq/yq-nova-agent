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

ALTER TABLE memory_items ADD COLUMN namespace_id INTEGER NOT NULL DEFAULT 1;
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

CREATE INDEX IF NOT EXISTS idx_memory_items_namespace ON memory_items(namespace_id);
CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace_id);
CREATE INDEX IF NOT EXISTS idx_entities_namespace_name_type ON entities(namespace_id, name, type);
CREATE INDEX IF NOT EXISTS idx_relations_namespace ON relations(namespace_id);
CREATE INDEX IF NOT EXISTS idx_relations_namespace_src ON relations(namespace_id, source_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_relations_namespace_tgt ON relations(namespace_id, target_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_namespace ON tags(namespace_id, name);
CREATE INDEX IF NOT EXISTS idx_memory_tags_namespace_uuid ON memory_tags(namespace_id, memory_uuid, tag_id);