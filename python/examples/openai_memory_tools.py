"""
Example: Using yq-nova with OpenAI function-calling style.
Requires a running yq-nova server.
"""
import json
from yq_nova import Client

client = Client("http://127.0.0.1:7999")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "nova_remember",
            "description": "Store a memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "importance": {"type": "number", "default": 0.5},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "nova_recall",
            "description": "Retrieve relevant memories",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "nova_forget",
            "description": "Archive or delete a memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "uuid": {"type": "string"},
                    "mode": {"type": "string", "enum": ["soft", "archive", "hard"], "default": "soft"}
                },
                "required": ["uuid"]
            }
        }
    }
]

def execute_tool_call(name: str, arguments: dict) -> dict:
    if name == "nova_remember":
        result = client.remember(**arguments)
        return {"result": "stored", "uuid": result["uuid"]}
    elif name == "nova_recall":
        result = client.recall(**arguments)
        return {"result": "recalled", "hits": len(result["hits"])}
    elif name == "nova_forget":
        result = client.forget(**arguments)
        return {"result": "forgotten", "affected": result["affected_memories"]}
    return {"error": f"unknown tool: {name}"}

if __name__ == "__main__":
    call1 = {"name": "nova_remember", "arguments": {"content": "Agent memory is stored in SQLite", "importance": 0.8, "tags": ["demo"]}}
    print("Tool call:", call1["name"])
    print(execute_tool_call(**call1))

    call2 = {"name": "nova_recall", "arguments": {"query": "SQLite memory", "top_k": 3}}
    print("Tool call:", call2["name"])
    print(execute_tool_call(**call2))