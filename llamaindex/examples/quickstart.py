import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "python"))

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index_yq_nova import YqNovaMemory, create_memory_tools
from yq_nova import Client


def main() -> None:
    base_url = os.environ.get("YQ_NOVA_URL", "http://127.0.0.1:7999")
    client = Client(base_url, api_key=os.environ.get("YQ_NOVA_API_KEY"))

    memory = YqNovaMemory(client, namespace="demo", top_k=10)
    memory.put(ChatMessage(role=MessageRole.USER, content="My name is Alice"))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Nice to meet you Alice"))

    recalled = memory.get("What is my name?")
    print("recalled messages", len(recalled))
    for message in recalled:
        print(message.role.value, "-", str(message.content)[:60])

    tools = create_memory_tools(client)
    print("tools", [tool.metadata.name for tool in tools])

    memory.reset()
    print("memory reset")


if __name__ == "__main__":
    main()