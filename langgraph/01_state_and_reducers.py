import operator
from typing import Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Defining custom state with TypedDict
- Using Annotated reducers (operator.add) to accumulate values
- Conditional edges for dynamic routing
- Multi-path graph execution

State in LangGraph is a shared data structure passed between nodes.
Reducers define how node outputs merge into existing state — for example,
operator.add appends to a list instead of replacing it. Conditional
edges let you route execution based on runtime state values.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""


# --- 1. Define custom state with reducers ---
class WorkflowState(TypedDict):
    topic: str
    research_notes: Annotated[list[str], operator.add]  # Appends across nodes
    quality_score: int
    final_summary: str


# --- 2. Define node functions ---
def research_node(state: WorkflowState) -> dict:
    """Simulate gathering research notes on the topic."""
    topic = state["topic"]
    return {
        "research_notes": [
            f"Note 1: {topic} has a rich history dating back decades.",
            f"Note 2: {topic} is widely used in modern applications.",
        ],
        "quality_score": 7,
    }


def deep_dive_node(state: WorkflowState) -> dict:
    """Add more detailed notes when quality is high enough."""
    topic = state["topic"]
    return {
        "research_notes": [
            f"Deep dive: {topic} has influenced many related fields.",
        ],
    }


def quick_summary_node(state: WorkflowState) -> dict:
    """Produce a quick summary when quality is low."""
    notes = state["research_notes"]
    return {"final_summary": f"Quick summary based on {len(notes)} notes: {notes[0]}"}


def detailed_summary_node(state: WorkflowState) -> dict:
    """Produce a detailed summary combining all notes."""
    notes = state["research_notes"]
    combined = " | ".join(notes)
    return {"final_summary": f"Detailed summary from {len(notes)} notes: {combined}"}


# --- 3. Define conditional routing ---
def route_by_quality(state: WorkflowState) -> str:
    """Route to deep_dive if quality >= 6, otherwise quick_summary."""
    if state["quality_score"] >= 6:
        return "deep_dive"
    return "quick_summary"


# --- 4. Build the graph ---
builder = StateGraph(WorkflowState)

builder.add_node("research", research_node)
builder.add_node("deep_dive", deep_dive_node)
builder.add_node("quick_summary", quick_summary_node)
builder.add_node("detailed_summary", detailed_summary_node)

builder.add_edge(START, "research")
builder.add_conditional_edges(
    "research",
    route_by_quality,
    {
        "deep_dive": "deep_dive",
        "quick_summary": "quick_summary",
    },
)
builder.add_edge("deep_dive", "detailed_summary")
builder.add_edge("quick_summary", END)
builder.add_edge("detailed_summary", END)

graph = builder.compile()

# --- 5. Run the graph ---
print("=== State & Reducers Example ===\n")

result = graph.invoke(
    {
        "topic": "Artificial Intelligence",
        "research_notes": [],
        "quality_score": 0,
        "final_summary": "",
    }
)

print(f"Topic: {result['topic']}")
print(f"Quality Score: {result['quality_score']}")
print(f"Research Notes ({len(result['research_notes'])} total):")
for note in result["research_notes"]:
    print(f"  - {note}")
print(f"\nFinal Summary: {result['final_summary']}")
