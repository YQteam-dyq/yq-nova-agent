-- =============================================================
-- yq-nova-agent migration 003
-- Supplementary composite + covering indexes for hot query paths.
-- =============================================================
-- NOTE: sqlx migration runner looks for the special `--!` magic
-- comments below. Do not change their format.

-- relations: (source, predicate) and (target, predicate) are the two
-- hottest paths — used by both BFS traversal and per-entity lookups.
-- We also add a (memory_uuid, source_uuid) pair so the graph-expansion
-- step's "which memories reference entity X after BFS" can be answered
-- purely from the index.
CREATE INDEX IF NOT EXISTS idx_rel_src_pred      ON relations(source_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_rel_tgt_pred      ON relations(target_uuid, predicate);
CREATE INDEX IF NOT EXISTS idx_rel_memory_source ON relations(memory_uuid, source_uuid);
CREATE INDEX IF NOT EXISTS idx_rel_memory_target ON relations(memory_uuid, target_uuid);

-- entities: (name, type) is already backed by a UNIQUE constraint (which
-- SQLite implements with an internal index), but we add an explicit index
-- so `EXPLAIN QUERY PLAN` callers can see/annotate the usage; and a
-- (created_at, type) covering index for the admin-style list endpoints
-- that filter by entity_type and order by recency.
CREATE INDEX IF NOT EXISTS idx_entities_created_type ON entities(created_at, type);

-- memory_items: (status, importance DESC) accelerates the GC staleness
-- scan ("status=active AND importance < ?").
-- (status, source, created_at DESC) covers the list/filter hot path
-- when the caller restricts by source + status + time window.
CREATE INDEX IF NOT EXISTS idx_memory_status_importance ON memory_items(status, importance);
CREATE INDEX IF NOT EXISTS idx_memory_status_source_created ON memory_items(status, source, created_at);

-- embeddings: (dims, provider) covering so we can skip rows whose dims
-- don't match the current configured provider without reading the blob.
CREATE INDEX IF NOT EXISTS idx_embeddings_dims_provider ON embeddings(dims, provider);

-- memory_tags: (tag_id, memory_uuid) reverse index — needed for
-- filter.tag_any / tags_all joins; the primary key is (memory_uuid, tag_id).
CREATE INDEX IF NOT EXISTS idx_memory_tags_reverse ON memory_tags(tag_id, memory_uuid);
