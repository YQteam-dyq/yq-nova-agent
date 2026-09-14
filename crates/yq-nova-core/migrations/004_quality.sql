CREATE TABLE IF NOT EXISTS memory_links (
    source_memory_uuid TEXT    NOT NULL,
    target_memory_uuid TEXT    NOT NULL,
    kind                TEXT    NOT NULL DEFAULT 'semantic',
    weight              REAL    NOT NULL DEFAULT 0.0,
    metadata_json       TEXT    NOT NULL DEFAULT '{}',
    created_at          INTEGER NOT NULL,
    updated_at          INTEGER NOT NULL,
    PRIMARY KEY (source_memory_uuid, target_memory_uuid, kind),
    FOREIGN KEY (source_memory_uuid) REFERENCES memory_items(uuid) ON DELETE CASCADE,
    FOREIGN KEY (target_memory_uuid) REFERENCES memory_items(uuid) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_memory_uuid);
CREATE INDEX IF NOT EXISTS idx_memory_links_kind   ON memory_links(kind);
CREATE INDEX IF NOT EXISTS idx_memory_links_weight ON memory_links(weight);

CREATE TABLE IF NOT EXISTS importance_log (
    memory_uuid TEXT    NOT NULL,
    importance   REAL    NOT NULL,
    recorded_at  INTEGER NOT NULL,
    PRIMARY KEY (memory_uuid, recorded_at)
);

CREATE INDEX IF NOT EXISTS idx_importance_log_memory ON importance_log(memory_uuid);