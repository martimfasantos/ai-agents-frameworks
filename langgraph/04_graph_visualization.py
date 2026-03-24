import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Generating Mermaid diagram syntax from compiled graphs
- ASCII graph rendering for terminal output
- Visualizing complex graphs with conditional edges

Graph visualization helps you understand and debug your workflow
structure. LangGraph can export graphs as Mermaid diagrams (for docs
and web rendering) or ASCII art (for terminal debugging). This is
especially useful for complex graphs with conditional routing.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""

# --- 1. Build a graph with conditional edges to visualize ---
llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)


class AgentState(MessagesState):
    next_action: str


def classifier_node(state: AgentState) -> dict:
    """Classify the user's intent."""
    return {"next_action": "technical"}


def technical_node(state: AgentState) -> dict:
    """Handle technical questions."""
    return {"messages": [("assistant", "Here's a technical answer.")]}


def general_node(state: AgentState) -> dict:
    """Handle general questions."""
    return {"messages": [("assistant", "Here's a general answer.")]}


def review_node(state: AgentState) -> dict:
    """Review and finalize the response."""
    return {"messages": [("assistant", "Response reviewed and approved.")]}


def route_by_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    if state.get("next_action") == "technical":
        return "technical"
    return "general"


# --- 2. Assemble the graph ---
builder = StateGraph(AgentState)

builder.add_node("classifier", classifier_node)
builder.add_node("technical", technical_node)
builder.add_node("general", general_node)
builder.add_node("review", review_node)

builder.add_edge(START, "classifier")
builder.add_conditional_edges(
    "classifier",
    route_by_intent,
    {
        "technical": "technical",
        "general": "general",
    },
)
builder.add_edge("technical", "review")
builder.add_edge("general", "review")
builder.add_edge("review", END)

graph = builder.compile()

# --- 3. Mermaid diagram ---
print("=== Mermaid Diagram ===\n")
mermaid_syntax = graph.get_graph().draw_mermaid()
print(mermaid_syntax)

# Save the Mermaid syntax to a file for rendering
os.makedirs("res", exist_ok=True)
with open("res/agent_graph.mmd", "w") as f:
    f.write(mermaid_syntax)
print("\nMermaid diagram saved to res/agent_graph.mmd")

# --- 4. ASCII representation ---
print("\n=== ASCII Graph ===\n")
ascii_graph = graph.get_graph().draw_ascii()
print(ascii_graph)
