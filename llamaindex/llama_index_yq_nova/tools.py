from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama_index.core.tools import FunctionTool

from yq_nova import Client


def create_remember_tool(client: Client, name: str = "nova_remember") -> FunctionTool:
    def remember(content: str, importance: float = 0.5, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        return client.remember(content, importance=importance, tags=tags or [])

    return FunctionTool.from_defaults(
        fn=remember,
        name=name,
        description="Store a memory in the yq-nova agent memory layer",
    )


def create_recall_tool(
    client: Client,
    name: str = "nova_recall",
    top_k_default: int = 5,
) -> FunctionTool:
    def recall(query: str, top_k: int = top_k_default) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        return client.recall(query, top_k=top_k, mode="hybrid")

    return FunctionTool.from_defaults(
        fn=recall,
        name=name,
        description="Retrieve the most relevant memories for a query",
    )


def create_forget_tool(client: Client, name: str = "nova_forget") -> FunctionTool:
    def forget(uuid: str, mode: str = "soft") -> Dict[str, Any]:
        if not uuid or not uuid.strip():
            raise ValueError("uuid must be non-empty")
        return client.forget(uuid=uuid, mode=mode)

    return FunctionTool.from_defaults(
        fn=forget,
        name=name,
        description="Archive or delete a memory by its uuid",
    )


def create_memory_tools(client: Client) -> List[FunctionTool]:
    return [
        create_remember_tool(client),
        create_recall_tool(client),
        create_forget_tool(client),
    ]