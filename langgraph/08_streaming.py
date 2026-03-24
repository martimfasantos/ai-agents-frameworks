import asyncio

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Streaming with mode "updates" (node-by-node outputs)
- Streaming with mode "messages" (LLM token-by-token)
- Streaming tool calls and AI responses in real-time

LangGraph supports multiple streaming modes: "updates" emits the full
output of each node as it completes, while "messages" streams individual
LLM tokens as they're generated. This enables responsive UIs that show
progress during long-running agent workflows.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/streaming
-----------------------------------------------------------------------
"""


# --- 1. Set up a tool-calling agent for streaming ---
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Only supports basic arithmetic."""
    allowed = set("0123456789+-*/.(). ")
    if all(c in allowed for c in expression):
        try:
            return str(eval(expression))
        except Exception:
            return f"Could not evaluate: {expression}"
    return f"Invalid expression: {expression}"


llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)
llm_with_tools = llm.bind_tools([calculate])


def chatbot(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode([calculate]))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile()


# --- 2. Stream mode: "updates" ---
async def stream_updates():
    """Stream node-by-node updates."""
    print("=== Stream Mode: updates ===\n")

    async for event in graph.astream(
        {
            "messages": [
                SystemMessage(content="You are a helpful math assistant. Be concise."),
                HumanMessage(content="What is 42 * 17 + 3? Use the calculate tool."),
            ]
        },
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            print(f"[Node: {node_name}]")
            if "messages" in node_output:
                last_msg = node_output["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"  Tool call: {tc['name']}({tc['args']})")
                elif hasattr(last_msg, "content") and last_msg.content:
                    print(f"  Content: {last_msg.content[:150]}")
        print()


# --- 3. Stream mode: "messages" (token-by-token) ---
async def stream_messages():
    """Stream LLM tokens as they are generated."""
    print("=== Stream Mode: messages (token-by-token) ===\n")

    print("AI: ", end="", flush=True)
    async for msg, metadata in graph.astream(
        {
            "messages": [
                SystemMessage(content="You are a helpful assistant. Be concise."),
                HumanMessage(content="Explain what pi is in exactly two sentences."),
            ]
        },
        stream_mode="messages",
    ):
        # Only print content tokens from the chatbot node
        if metadata.get("langgraph_node") == "chatbot" and msg.content:
            print(msg.content, end="", flush=True)
    print("\n")


# --- 4. Run both streaming examples ---
async def main():
    await stream_updates()
    await stream_messages()


if __name__ == "__main__":
    asyncio.run(main())
