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
- Defining tools with the @tool decorator
- Binding tools to a ChatOpenAI model
- Using the prebuilt ToolNode for automatic tool execution
- Conditional edges with tools_condition for the ReAct loop

Tool calling is the foundation of agentic behavior: the LLM decides
which tools to invoke, LangGraph executes them, and the results feed
back into the conversation. The tools_condition helper automatically
routes to the tool node when the LLM requests a tool call, or to END
when it produces a final answer.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/quickstart
-----------------------------------------------------------------------
"""


# --- 1. Define tools ---


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


@tool
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    population_data = {
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "new york": "8.3 million",
        "lisbon": "0.5 million",
    }
    return population_data.get(city.lower(), f"No population data for {city}")


tools = [get_weather, get_population]

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
builder.add_conditional_edges("chatbot", tools_condition)  # Routes to "tools" or END
builder.add_edge("tools", "chatbot")  # After tool execution, go back to chatbot

graph = builder.compile()

# --- 5. Run the agent ---
print("=== Tool Calling Agent ===\n")

response = graph.invoke(
    {
        "messages": [
            SystemMessage(
                content="You are a helpful city information assistant. Be concise."
            ),
            HumanMessage(content="What's the weather and population of Tokyo?"),
        ]
    }
)

# Print the conversation
for msg in response["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"{role}: [Calling tools: {[tc['name'] for tc in msg.tool_calls]}]")
    elif hasattr(msg, "content") and msg.content:
        print(f"{role}: {msg.content}")
