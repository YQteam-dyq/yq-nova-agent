import type {
  BatchRememberInput,
  BatchRememberOutput,
  EntityRecord,
  ExportInput,
  ExtractLinkInput,
  ForgetInput,
  ForgetOutput,
  HealthOutput,
  ImportInput,
  ImportOutput,
  ListEntitiesParams,
  ListInput,
  ListOutput,
  ListRelationsParams,
  ListTagsParams,
  MergeEntitiesInput,
  MergeEntitiesOutput,
  MergeInput,
  MergeOutput,
  MemoryRecord,
  RecallInput,
  RecallOutput,
  RememberInput,
  RememberOutput,
  RelationRecord,
  StatsOutput,
  TagListOutput,
  TraverseInput,
  UpdateMemoryInput,
  UpsertEntityInput,
  UpsertEntityOutput,
  UpsertRelationInput,
  UpsertRelationOutput,
} from "./types.js";

export interface NovaClientOptions {
  apiKey?: string;
  timeoutMs?: number;
  fetch?: typeof fetch;
}

export class NovaApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly traceId: string | null;

  constructor(code: string, message: string, status: number, traceId: string | null = null) {
    super(message);
    this.name = "NovaApiError";
    this.code = code;
    this.status = status;
    this.traceId = traceId;
  }
}

interface RequestOptions {
  method: string;
  path: string;
  body?: unknown;
  params?: object;
}

export class NovaClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string, options: NovaClientOptions = {}) {
    if (!baseUrl) {
      throw new Error("baseUrl must not be empty");
    }
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 30000;
    this.fetchImpl = options.fetch ?? ((typeof globalThis !== "undefined" && globalThis.fetch) as typeof fetch);
    if (!this.fetchImpl) {
      throw new Error("A fetch implementation is required; Node 18+ and browsers provide one globally");
    }
  }

  health() {
    return this.request<HealthOutput>("GET", "/v1/health");
  }

  stats() {
    return this.request<StatsOutput>("GET", "/v1/stats");
  }

  remember(input: RememberInput) {
    return this.request<RememberOutput>("POST", "/v1/memory/remember", input);
  }

  rememberBatch(input: BatchRememberInput) {
    return this.request<BatchRememberOutput>("POST", "/v1/memory/remember-batch", input);
  }

  recall(input: RecallInput) {
    return this.request<RecallOutput>("POST", "/v1/memory/recall", input);
  }

  listMemories(input: ListInput = {}) {
    return this.request<ListOutput>("POST", "/v1/memory/list", input);
  }

  forget(input: ForgetInput) {
    return this.request<ForgetOutput>("POST", "/v1/memory/forget", input);
  }

  getMemory(uuid: string) {
    return this.request<MemoryRecord>("GET", `/v1/memory/${this.quote(uuid)}`);
  }

  updateMemory(uuid: string, input: UpdateMemoryInput) {
    return this.request<MemoryRecord>("PATCH", `/v1/memory/${this.quote(uuid)}`, input);
  }

  deleteMemory(uuid: string) {
    return this.request<ForgetOutput>("DELETE", `/v1/memory/${this.quote(uuid)}`);
  }

  exportMemories(input: ExportInput = {}) {
    return this.request<Record<string, unknown>>("POST", "/v1/memory/export", input);
  }

  importMemories(input: ImportInput) {
    return this.request<ImportOutput>("POST", "/v1/memory/import", input);
  }

  mergeMemories(input: MergeInput) {
    return this.request<MergeOutput>("POST", "/v1/memory/merge", input);
  }

  listTags(params: ListTagsParams = {}) {
    return this.request<TagListOutput>("GET", "/v1/tags", undefined, params);
  }

  renameTag(name: string, newName: string) {
    return this.request<Record<string, unknown>>("PATCH", `/v1/tags/${this.quote(name)}`, { new_name: newName });
  }

  deleteTag(name: string) {
    return this.request<Record<string, unknown>>("DELETE", `/v1/tags/${this.quote(name)}`);
  }

  upsertEntity(input: UpsertEntityInput) {
    return this.request<UpsertEntityOutput>("POST", "/v1/graph/entities", input);
  }

  listEntities(params: ListEntitiesParams = {}) {
    return this.request<EntityRecord[]>("GET", "/v1/graph/entities", undefined, params);
  }

  mergeEntities(input: MergeEntitiesInput) {
    return this.request<MergeEntitiesOutput>("POST", "/v1/graph/entities/merge", input);
  }

  upsertRelation(input: UpsertRelationInput) {
    return this.request<UpsertRelationOutput>("POST", "/v1/graph/relations", input);
  }

  listRelations(params: ListRelationsParams = {}) {
    return this.request<RelationRecord[]>("GET", "/v1/graph/relations", undefined, params);
  }

  traverse(input: TraverseInput) {
    return this.request<Array<Record<string, unknown>>>("POST", "/v1/graph/traverse", input);
  }

  extractAndLink(input: ExtractLinkInput) {
    return this.request<Record<string, unknown>>("POST", "/v1/graph/extract-and-link", input);
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    params?: object,
  ): Promise<T> {
    const query = params ? this.encodeParams(params) : "";
    const url = this.baseUrl + path + (query ? `?${query}` : "");

    const headers: Record<string, string> = { Accept: "application/json" };
    let payload: BodyInit | undefined;
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const controller = "AbortController" in globalThis ? new AbortController() : undefined;
    const timer =
      controller && this.timeoutMs > 0
        ? setTimeout(() => controller.abort(), this.timeoutMs)
        : undefined;

    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        method,
        headers,
        body: payload,
        signal: controller ? controller.signal : undefined,
      });
    } catch (err) {
      if (controller && err instanceof Error && err.name === "AbortError") {
        throw new NovaApiError("timeout", `request to ${url} timed out`, 0);
      }
      throw new NovaApiError("network", `request to ${url} failed: ${this.errorMessage(err)}`, 0);
    } finally {
      if (timer) {
        clearTimeout(timer);
      }
    }

    const text = await response.text();
    let data: unknown = undefined;
    let parseFailed = false;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        parseFailed = true;
      }
    }

    if (!response.ok || parseFailed) {
      const body: Record<string, unknown> =
        typeof data === "object" && data !== null ? (data as Record<string, unknown>) : {};
      const code = typeof body.code === "string" ? body.code : `http_${response.status}`;
      const message =
        typeof body.message === "string"
          ? body.message
          : parseFailed
            ? `invalid JSON response body: ${bodySnippet(text)}`
            : `HTTP ${response.status} returned no body`;
      const traceId = typeof body.trace_id === "string" ? body.trace_id : null;
      throw new NovaApiError(code, message, response.status, traceId);
    }

    return data as T;
  }

  private encodeParams(params: object): string {
    const parts: string[] = [];
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null || value === "") {
        continue;
      }
      parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);
    }
    return parts.join("&");
  }

  private quote(value: string): string {
    return encodeURIComponent(value);
  }

  private errorMessage(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }
}

function bodySnippet(text: string): string {
  const s = text.replace(/\s+/g, " ").trim();
  return s.length > 160 ? `${s.slice(0, 160)}...` : s;
}