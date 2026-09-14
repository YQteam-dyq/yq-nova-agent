from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

from yq_nova import Client


class YqNovaChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        client: Client,
        namespace: str = "default",
        tag: str = "langchain",
        top_k: int = 500,
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.tag = tag
        self.top_k = top_k

    def _tags(self) -> List[str]:
        return [self.tag, f"{self.tag}:{self.namespace}"]

    def add_message(self, message: BaseMessage) -> None:
        payload = message_to_dict(message)
        data = payload.get("data") or {}
        content = data.get("content") if isinstance(data, dict) else None
        text = content if isinstance(content, str) else str(payload)
        self.client.remember(
            text,
            importance=0.5,
            tags=self._tags(),
            metadata={
                "role": message.type,
                "namespace": self.namespace,
                "message": payload,
            },
        )

    async def aadd_message(self, message: BaseMessage) -> None:
        await asyncio.to_thread(self.add_message, message)

    def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        page = self.client.list_memories(
            filter={"tags_all": self._tags()},
            limit=limit or self.top_k,
            offset=0,
            sort="created_asc",
        )
        result: List[BaseMessage] = []
        for item in page.get("items", []):
            meta = item.get("metadata") if isinstance(item, dict) else None
            if isinstance(meta, dict):
                payload = meta.get("message")
                if isinstance(payload, dict):
                    try:
                        result.extend(messages_from_dict([payload]))
                        continue
                    except Exception:
                        pass
            text = item.get("content", "")
            role = (meta or {}).get("role", "human") if isinstance(meta, dict) else "human"
            result.append(_message_from_role(role, text))
        return result

    async def aget_messages(self) -> List[BaseMessage]:
        return await asyncio.to_thread(self.get_messages)

    @property
    def messages(self) -> List[BaseMessage]:
        return self.get_messages()

    def clear(self) -> None:
        self.client.forget(
            filter={"tags_all": self._tags()},
            mode="hard",
            batch_limit=500,
        )

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)


def _message_from_role(role: str, content: str) -> BaseMessage:
    text = str(content or "")
    if role == "ai":
        return AIMessage(content=text)
    if role == "system":
        return SystemMessage(content=text)
    if role == "function":
        return FunctionMessage(content=text, name="nova")
    if role == "tool":
        return ToolMessage(content=text, tool_call_id="nova")
    return HumanMessage(content=text)


class YqNovaMemory(BaseMemory):
    namespace: str = "default"
    tag: str = "langchain"
    top_k: int = 20

    def __init__(
        self,
        client: Client,
        namespace: str = "default",
        tag: str = "langchain",
        top_k: int = 20,
    ) -> None:
        super().__init__(namespace=namespace, tag=tag, top_k=top_k)
        self._client = client

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]

    def _tags(self) -> List[str]:
        return [self.tag, f"{self.tag}:{self.namespace}"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("input", inputs.get("question", ""))
        text = query if isinstance(query, str) else str(query)
        if not text.strip():
            return {"history": ""}
        recalled = self._client.recall(
            text,
            top_k=self.top_k,
            mode="hybrid",
            filter={"tags_all": self._tags()},
        )
        hits = recalled.get("hits", [])
        lines = []
        for hit in hits:
            mem = hit.get("memory", {})
            meta = mem.get("metadata") if isinstance(mem, dict) else None
            role = (meta or {}).get("role", "human") if isinstance(meta, dict) else "human"
            content = mem.get("content", "")
            prefix = "AI" if role == "ai" else ("Human" if role == "human" else role or "Human")
            lines.append(f"{prefix}: {content}")
        return {"history": "\n".join(lines) if lines else ""}

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.load_memory_variables, inputs)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        for key, value in inputs.items():
            if value is None:
                continue
            self._client.remember(
                str(value),
                importance=0.5,
                tags=self._tags(),
                metadata={"role": "human", "namespace": self.namespace},
            )
        for key, value in outputs.items():
            if value is None:
                continue
            self._client.remember(
                str(value),
                importance=0.5,
                tags=self._tags(),
                metadata={"role": "ai", "namespace": self.namespace},
            )

    async def asave_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str],
    ) -> None:
        await asyncio.to_thread(self.save_context, inputs, outputs)

    def clear(self) -> None:
        self._client.forget(
            filter={"tags_all": self._tags()},
            mode="hard",
            batch_limit=500,
        )

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)