export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export type MemorySource = "agent" | "user" | "assistant" | "system";

export type MemoryStatus = "active" | "archived";

export type MemorySortOrder =
  | "created_desc"
  | "created_asc"
  | "importance_desc"
  | "importance_asc"
  | "accessed_desc";

export type ForgetMode = "soft" | "archive" | "hard";

export type SearchMode = "semantic" | "keyword" | "hybrid";

export interface MemoryRecord {
  id: number;
  uuid: string;
  content: string;
  content_hash: string;
  metadata: Record<string, JsonValue>;
  source: MemorySource;
  importance: number;
  access_count: number;
  last_accessed: string | null;
  created_at: string;
  expires_at: string | null;
  status: MemoryStatus;
  tags: string[];
}

export interface MemoryFilter {
  source_in?: MemorySource[];
  created_after?: string;
  created_before?: string;
  importance_min?: number;
  importance_max?: number;
  status_in?: MemoryStatus[];
  access_count_lt?: number;
  last_accessed_before?: string;
  tags_all?: string[];
}

export interface RememberInput {
  content: string;
  source?: MemorySource;
  importance?: number;
  metadata?: Record<string, JsonValue>;
  expires_at?: string | null;
  tags?: string[];
  embed?: boolean;
  extract_graph?: boolean;
  chunk_options?: Record<string, JsonValue>;
}

export interface RememberOutput {
  uuid: string;
  duplicate: boolean;
  memory?: MemoryRecord;
  chunks?: MemoryRecord[];
  deduplicated?: boolean;
}

export interface RecallHit {
  memory: MemoryRecord;
  final_score: number;
  raw_similarity: number | null;
  from_graph: boolean;
  components: Record<string, JsonValue>;
}

export interface RecallOutput {
  hits: RecallHit[];
  total_candidates: number;
  query: string;
}

export interface GraphTraversalOpts {
  enabled?: boolean;
  max_depth?: number;
  predicate_whitelist?: string[];
}

export interface HybridWeights {
  semantic?: number;
  keyword?: number;
  graph?: number;
}

export interface RecallInput {
  query: string;
  top_k?: number;
  mode?: SearchMode;
  score_threshold?: number;
  similarity_threshold?: number;
  graph?: GraphTraversalOpts;
  hybrid_weights?: HybridWeights;
  rrf_k?: number;
  rank_weights?: Record<string, JsonValue>;
  filter?: MemoryFilter;
  group_chunks?: boolean;
  entity_focus?: string[];
}

export interface ListInput {
  filter?: MemoryFilter;
  limit?: number;
  offset?: number;
  sort?: MemorySortOrder;
}

export interface ListOutput {
  total: number;
  count: number;
  limit: number;
  offset: number;
  sort: MemorySortOrder;
  items: MemoryRecord[];
}

export interface BatchRememberItem {
  content: string;
  source?: MemorySource;
  importance?: number;
  metadata?: Record<string, JsonValue>;
  expires_at?: string | null;
  tags?: string[];
  embed?: boolean;
  extract_graph?: boolean;
}

export interface BatchRememberInput {
  items: BatchRememberItem[];
  continue_on_error?: boolean;
}

export interface BatchRememberResult {
  index: number;
  uuid: string | null;
  duplicate: boolean;
  error: string | null;
}

export interface BatchRememberOutput {
  received: number;
  succeeded: number;
  duplicates: number;
  failed: number;
  results: BatchRememberResult[];
}

export interface ForgetInput {
  target:
    | { type: "one"; value: string }
    | { type: "filter"; value: MemoryFilter }
    | { type: string; value: unknown };
  mode?: ForgetMode;
  gc_graph?: boolean;
  batch_limit?: number;
}

export interface ForgetOutput {
  affected_memories: number;
  cascade_embeddings: number;
  gc_entities: number;
  gc_relations: number;
  relations_cleaned: number;
  mode: ForgetMode;
}

export interface ExportInput {
  filter?: MemoryFilter;
  limit?: number;
  offset?: number;
}

export interface ImportInput {
  items: Array<Record<string, JsonValue>>;
  embed?: boolean;
  on_conflict?: "skip" | "replace";
}

export interface ImportOutput {
  received: number;
  imported: number;
  skipped: number;
  replaced: number;
  results?: Array<Record<string, JsonValue>>;
}

export interface MergeInput {
  uuids: string[];
  keep_uuid?: string;
}

export interface MergeOutput {
  kept_uuid: string;
  merged: number;
  discarded_uuids: string[];
}

export interface UpdateMemoryInput {
  content?: string;
  importance?: number;
  metadata?: Record<string, JsonValue>;
  tags?: string[];
  expires_at?: string | null;
}

export interface TagRecord {
  id: number;
  name: string;
  color: string | null;
  created_at: string;
  memory_count: number;
}

export interface ListTagsParams {
  limit?: number;
  offset?: number;
}

export interface EntityRecord {
  id: number;
  uuid: string;
  name: string;
  type: string;
  description: string | null;
  metadata: Record<string, JsonValue>;
  created_at: string;
  updated_at: string;
}

export interface UpsertEntityInput {
  name: string;
  type?: string;
  description?: string | null;
  metadata?: Record<string, JsonValue>;
}

export interface UpsertEntityOutput {
  created: boolean;
  updated: boolean;
  entity: EntityRecord;
}

export interface ListEntitiesParams {
  name_prefix?: string;
  entity_type?: string;
  limit?: number;
  offset?: number;
}

export interface MergeEntitiesInput {
  keep_uuid: string;
  discard_uuids: string[];
}

export interface MergeEntitiesOutput {
  kept_uuid: string;
  discard_uuids: string[];
}

export interface RelationRecord {
  id: number;
  uuid: string;
  source_uuid: string;
  target_uuid: string;
  predicate: string;
  confidence: number;
  memory_uuid: string | null;
  metadata: Record<string, JsonValue>;
  created_at: string;
}

export interface UpsertRelationInput {
  source_uuid: string;
  target_uuid: string;
  predicate: string;
  confidence?: number;
  metadata?: Record<string, JsonValue>;
  idempotent?: boolean;
  memory_uuid?: string | null;
}

export interface UpsertRelationOutput {
  inserted: boolean;
  updated: boolean;
  relation_uuid: string;
  relation: RelationRecord;
}

export interface ListRelationsParams {
  source?: string;
  target?: string;
  predicate?: string;
  limit?: number;
  offset?: number;
}

export interface TraverseInput {
  start: string;
  max_depth?: number;
  max_nodes?: number;
  predicate_whitelist?: string[];
  min_confidence?: number;
}

export interface ExtractLinkInput {
  text: string;
  opts?: {
    enabled?: boolean;
    upsert_entities?: boolean;
    create_relations?: boolean;
    min_confidence?: number;
  };
}

export interface HealthOutput {
  status: string;
  version: string;
  git_sha: string;
  uptime_secs: number;
}

export interface StatsOutput {
  uptime_secs: number;
  database_size_bytes: number | null;
  memory_active: number;
  memory_archived: number;
  memory_total: number;
  entity_count: number;
  relation_count: number;
  tag_count: number;
}