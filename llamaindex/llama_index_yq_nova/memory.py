from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory

from yq_nova import Client


class YqNovaMemory(BaseMemory):
    namespace: str = "default"
    tag: str = "llamaindex"
    top_k: int = 20

    def __init__(
        self,
        client: Client,
        namespace: str = "default",
        tag: str = "llamaindex",
        top_k: int = 20,
    ) -> None:
        super().__init__(namespace=namespace, tag=tag, top_k=top_k)
        self._client = client

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "YqNovaMemory":
        return cls(**kwargs)

    def _tags(self) -> List[str]:
        return [self.tag, f"{self.tag}:{self.namespace}"]

    def put(self, message: ChatMessage) -> None:
        content: Optional[str] = None
        if message.content is not None:
            content = str(message.content)
        elif message.blocks:
            content = "\n".join(str(block) for block in message.blocks)
        text = content if content else message.model_dump_json()
        role = message.role
        role_name = role.value if isinstance(role, MessageRole) else str(role)
        self._client.remember(
            text,
            importance=0.5,
            tags=self._tags(),
            metadata={"role": role_name, "namespace": self.namespace},
        )

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        if not input or not str(input).strip():
            return self.get_all()
        recalled = self._client.recall(
            str(input),
            top_k=self.top_k,
            mode="hybrid",
            filter={"tags_all": self._tags()},
        )
        messages: List[ChatMessage] = []
        for hit in recalled.get("hits", []):
            mem = hit.get("memory", {}) if isinstance(hit, dict) else {}
            meta = mem.get("metadata") if isinstance(mem, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            messages.append(self._to_message(mem.get("content", ""), meta.get("role", "user")))
        return messages

    def get_all(self) -> List[ChatMessage]:
        page = self._client.list_memories(
            filter={"tags_all": self._tags()},
            limit=self.top_k,
            offset=0,
            sort="created_asc",
        )
        messages: List[ChatMessage] = []
        for item in page.get("items", []):
            meta = item.get("metadata") if isinstance(item, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            messages.append(self._to_message(item.get("content", ""), meta.get("role", "user")))
        return messages

    def set(self, messages: List[ChatMessage]) -> None:
        self.reset()
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        self._client.forget(
            filter={"tags_all": self._tags()},
            mode="hard",
            batch_limit=500,
        )

    def _to_message(self, content: Any, role: Any) -> ChatMessage:
        text = str(content or "")
        return ChatMessage(role=_parse_role(role), content=text)


def _parse_role(role: Any) -> MessageRole:
    if isinstance(role, MessageRole):
        return role
    if isinstance(role, str):
        try:
            return MessageRole(role)
        except ValueError:
            pass
    return MessageRole.USER