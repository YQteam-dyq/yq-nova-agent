import { test } from "node:test";
import assert from "node:assert/strict";
import { NovaClient, NovaApiError } from "../dist/index.js";

function recordClient(opts = {}) {
  const calls = [];
  const stubFetch = async (url, init) => {
    calls.push({ url, init });
    const body = opts.body ?? { ok: true };
    const status = opts.status ?? 200;
    return new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    });
  };
  const client = new NovaClient("http://127.0.0.1:7999", {
    apiKey: "k",
    fetch: stubFetch,
  });
  return { client, calls };
}

test("remember posts JSON to the right path", async () => {
  const { client, calls } = recordClient({
    body: { uuid: "u1", duplicate: false },
  });
  const out = await client.remember({ content: "hello", tags: ["a"] });
  assert.equal(out.uuid, "u1");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://127.0.0.1:7999/v1/memory/remember");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["Authorization"], "Bearer k");
  assert.deepEqual(JSON.parse(calls[0].init.body), { content: "hello", tags: ["a"] });
});

test("GET endpoints encode query params", async () => {
  const { client, calls } = recordClient({ body: [] });
  await client.listRelations({ source: "s", limit: 5, offset: 2 });
  await client.listTags({ limit: 10 });
  assert.equal(
    calls[0].url,
    "http://127.0.0.1:7999/v1/graph/relations?source=s&limit=5&offset=2",
  );
  assert.equal(calls[1].url, "http://127.0.0.1:7999/v1/tags?limit=10");
});

test("path segments are URL encoded", async () => {
  const { client, calls } = recordClient({ body: {} });
  await client.renameTag("a/b", "c");
  assert.equal(calls[0].url, "http://127.0.0.1:7999/v1/tags/a%2Fb");
});

test("non-2xx responses throw NovaApiError", async () => {
  const { client } = recordClient({
    status: 404,
    body: { code: "not_found", message: "missing", trace_id: "t1" },
  });
  await assert.rejects(
    () => client.getMemory("nope"),
    (err) =>
      err instanceof NovaApiError &&
      err.code === "not_found" &&
      err.status === 404 &&
      err.traceId === "t1",
  );
});

test("empty base url is rejected", () => {
  assert.throws(() => new NovaClient(""), /must not be empty/);
});