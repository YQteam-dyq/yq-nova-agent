-- ============================================================
-- M9: FTS5 full-text search index on memory_items.content.
-- ============================================================
--
-- Uses an "external content table" pattern: FTS5 never stores the
-- raw text itself, it just reads from memory_items.content via
-- rowid.  Triggers keep the FTS index in sync with INSERT/UPDATE/
-- DELETE on memory_items so the application layer doesn't need to
-- know FTS5 even exists.

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content       = 'memory_items',
    content_rowid = 'id',
    -- tokenize with the Unicode 6.1 tokenizer: strip diacritics so
    -- café matches cafe; treat underscore as a word char so
    -- session_id / yq_nova stay as single tokens.
    tokenize      = "unicode61 remove_diacritics 2 tokenchars '_'"
);

-- ------------------------------------------------------------
-- INSERT sync
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS memory_fts_ai
AFTER INSERT ON memory_items
FOR EACH ROW BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ------------------------------------------------------------
-- UPDATE sync (only if content actually changed; FTS5 docs
-- recommend INSERT+DELETE via rowid instead of UPDATE on FTS)
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS memory_fts_au
AFTER UPDATE OF content ON memory_items
FOR EACH ROW BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ------------------------------------------------------------
-- DELETE sync
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS memory_fts_ad
AFTER DELETE ON memory_items
FOR EACH ROW BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
        VALUES('delete', old.id, old.content);
END;
