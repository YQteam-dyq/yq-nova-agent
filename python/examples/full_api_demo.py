import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yq_nova import Client, NovaApiError


def main() -> None:
    base_url = os.environ.get("YQ_NOVA_URL", "http://127.0.0.1:7999")
    api_key = os.environ.get("YQ_NOVA_API_KEY")
    client = Client(base_url, api_key=api_key)

    health = client.health()
    stats = client.stats()
    print("health", health["status"], "version", health["version"])
    print("stats", stats["memory_total"], "memories", stats["entity_count"], "entities")

    first = client.remember(
        "yq-nova is a lightweight agent memory layer",
        importance=0.9,
        tags=["meta", "nova"],
    )
    second = client.remember(
        "graph traversal walks entity relations",
        importance=0.6,
        tags=["graph"],
        extract_graph=True,
    )
    print("remembered", first["uuid"], "duplicate", first["duplicate"])

    batch = client.remember_batch(
        [
            {"content": "first batch item", "tags": ["batch"]},
            {"content": "second batch item", "tags": ["batch"]},
        ]
    )
    print("batch", batch["succeeded"], "succeeded of", batch["received"])

    page = client.list_memories(filter={"tags_all": ["batch"]}, limit=10, sort="created_desc")
    print("list total", page["total"], "count", page["count"])

    recalled = client.recall("agent memory", top_k=5, mode="hybrid")
    print("recall", len(recalled["hits"]), "hits", recalled["total_candidates"], "candidates")

    for hit in recalled["hits"]:
        uuid = hit["memory"]["uuid"]
        fetched = client.get_memory(uuid)
        print("fetched", fetched["content"][:24])
        client.update_memory(uuid, importance=0.99)
        print("updated importance", client.get_memory(uuid)["importance"])

    tags = client.list_tags(limit=100)
    print("tags", [t["name"] for t in tags["items"]])
    client.rename_tag("graph", "graph-traversal")
    client.delete_tag("graph-traversal")

    alice = client.upsert_entity("Alice", entity_type="person", description="demo user")
    bob = client.upsert_entity("Bob", entity_type="person", description="demo user")
    print("entities", alice["entity"]["name"], bob["entity"]["uuid"])

    all_entities = client.list_entities(entity_type="person", limit=50)
    print("listed entities", len(all_entities))

    client.upsert_relation(
        alice["entity"]["uuid"],
        bob["entity"]["uuid"],
        "reports_to",
        confidence=0.9,
        memory_uuid=first["uuid"],
    )
    relations = client.list_relations(source=alice["entity"]["uuid"], limit=50)
    print("relations", len(relations))

    traversed = client.traverse(alice["entity"]["uuid"], max_depth=2)
    print("traversed nodes", len(traversed))

    linked = client.extract_and_link("[[Alice]] works with [[Rust]]")
    print("extract linked entities", len(linked.get("entities", [])))

    exported = client.export_memories(limit=100)
    print("exported", len(exported.get("items", [])))

    imported = client.import_memories(
        [{"content": "imported one", "tags": ["imported"]}],
        embed=True,
        on_conflict="skip",
    )
    print("imported", imported.get("imported", imported.get("success", 0)))

    merged = client.merge_memories([first["uuid"], second["uuid"]])
    print("merged", merged["kept_uuid"] if isinstance(merged, dict) else merged)

    forgotten = client.forget(uuid=first["uuid"], mode="archive")
    print("forgotten archived", forgotten["affected_memories"])

    client.delete_memory(second["uuid"])
    print("hard deleted second memory")

    missed = os.environ.get("YQ_NOVA_EXPECT_MISS", "0") == "1"
    if missed:
        try:
            client.get_memory("00000000-0000-0000-0000-000000000000")
        except NovaApiError as err:
            print("expected error", err.status, err.code)


if __name__ == "__main__":
    main()