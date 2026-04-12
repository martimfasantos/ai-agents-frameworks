from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Token usage tracking via AIMessage.usage_metadata
- Per-message input, output, and total token counts
- Aggregating tokens across the full agent conversation
- Inspecting token detail breakdowns (cached, reasoning)

LangGraph inherits token tracking from LangChain's message model.
Every AIMessage returned by an OpenAI-compatible model includes a
usage_metadata dict with input_tokens, output_tokens, total_tokens,
and optional detail breakdowns (input_token_details,
output_token_details). To get full-conversation totals, iterate
over all AIMessages in the final state and sum their usage.

For more details, visit:
https://python.langchain.com/docs/concepts/messages/#aimessage
-----------------------------------------------------------------------
"""


# --- 1. Define a simple tool ---


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "Cloudy, 14°C, 70% humidity",
        "tokyo": "Sunny, 28°C, 45% humidity",
        "new york": "Rainy, 18°C, 85% humidity",
        "lisbon": "Sunny, 25°C, 50% humidity",
    }
    return weather_data.get(city.lower(), f"No weather data available for {city}")


tools = [get_weather]

# --- 2. Create the LLM with bound tools ---
llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME).bind_tools(tools)


# --- 3. Define the chatbot node ---
def chatbot(state: MessagesState):
    """Call the LLM — it may request tool calls or produce a final answer."""
    return {"messages": [llm.invoke(state["messages"])]}


# --- 4. Build the ReAct agent graph ---
builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile()

# --- 5. Run the agent ---
print("=== LangGraph Token Usage Tracking ===\n")
print("--- Running tool-calling agent ---\n")

response = graph.invoke(
    {
        "messages": [
            SystemMessage(content="You are a helpful weather assistant. Be concise."),
            HumanMessage(content="What's the weather in Tokyo and London?"),
        ]
    }
)

# --- 6. Display the conversation ---
print("--- Conversation ---")
for msg in response["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"  {role}: [Calling tools: {[tc['name'] for tc in msg.tool_calls]}]")
    elif hasattr(msg, "content") and msg.content:
        content_preview = msg.content[:120]
        print(f"  {role}: {content_preview}")

# --- 7. Display per-message token usage ---
print("\n--- Per-Message Token Usage (AIMessage.usage_metadata) ---")
ai_messages = [m for m in response["messages"] if isinstance(m, AIMessage)]

total_input = 0
total_output = 0
total_tokens = 0

for i, msg in enumerate(ai_messages):
    usage = msg.usage_metadata
    if usage:
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)
        msg_total = usage.get("total_tokens", 0)
        total_input += input_tok
        total_output += output_tok
        total_tokens += msg_total

        has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
        label = "tool call request" if has_tool_calls else "final answer"
        print(f"\n  AIMessage {i + 1} ({label}):")
        print(f"    Input tokens:  {input_tok}")
        print(f"    Output tokens: {output_tok}")
        print(f"    Total tokens:  {msg_total}")

        # Show detail breakdowns if available
        input_details = usage.get("input_token_details", {})
        output_details = usage.get("output_token_details", {})
        if input_details:
            cached = input_details.get("cached", 0)
            print(f"    Cached input tokens: {cached}")
        if output_details:
            reasoning = output_details.get("reasoning", 0)
            if reasoning:
                print(f"    Reasoning tokens: {reasoning}")
    else:
        print(f"\n  AIMessage {i + 1}: No usage_metadata available")

# --- 8. Display aggregated totals ---
print("\n--- Aggregated Token Usage ---")
print(f"  Total input tokens:  {total_input}")
print(f"  Total output tokens: {total_output}")
print(f"  Total tokens:        {total_tokens}")
print(f"  LLM calls:           {len(ai_messages)}")

print("\n=== Token Usage Demo Complete ===")
