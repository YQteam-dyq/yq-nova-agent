import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "python"))

from langchain_core.messages import HumanMessage
from langchain_yq_nova import YqNovaChatMessageHistory, YqNovaMemory, create_memory_tools
from yq_nova import Client


def main() -> None:
    base_url = os.environ.get("YQ_NOVA_URL", "http://127.0.0.1:7999")
    client = Client(base_url, api_key=os.environ.get("YQ_NOVA_API_KEY"))

    history = YqNovaChatMessageHistory(client, namespace="demo")
    history.add_message(HumanMessage(content="Hello, I am building a LangChain agent"))
    print("history messages", len(history.messages))

    provider = YqNovaMemory(client, namespace="demo", top_k=10)
    provider.save_context({"input": "My name is Alice"}, {"output": "Nice to meet you Alice"})
    loaded = provider.load_memory_variables({"input": "What is my name?"})
    print("recalled history:")
    print(loaded["history"])

    tools = create_memory_tools(client)
    print("tools", [tool.name for tool in tools])

    history.clear()
    provider.clear()


if __name__ == "__main__":
    main()