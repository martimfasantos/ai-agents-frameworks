import operator
from typing import Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Map-reduce pattern using the Send API
- Dynamic fan-out: spawning parallel workers per item
- Aggregating results with reducers (operator.add)
- Combining LLM processing with parallel execution

The Send API enables map-reduce workflows: a "map" node fans out by
returning Send() objects, each spawning a parallel worker with its own
state. Workers process independently, and their results are aggregated
back using reducers. This is ideal for tasks like analyzing multiple
documents, processing items in parallel, or multi-perspective evaluation.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""

llm = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)


# --- 1. Define state schemas ---
class OverallState(TypedDict):
    topics: list[str]
    analyses: Annotated[list[str], operator.add]  # Aggregates results from workers
    final_report: str


class WorkerState(TypedDict):
    topic: str
    analyses: Annotated[list[str], operator.add]


# --- 2. Define the fan-out function ---
def fan_out_to_workers(state: OverallState) -> list[Send]:
    """Spawn one 'analyze' worker per topic."""
    return [
        Send("analyze", {"topic": topic, "analyses": []}) for topic in state["topics"]
    ]


# --- 3. Define the worker node ---
def analyze_node(state: WorkerState) -> dict:
    """Each worker analyzes one topic using the LLM."""
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a research analyst. Provide a brief 2-sentence analysis of the given topic's current state and future outlook."
            ),
            HumanMessage(content=f"Analyze: {state['topic']}"),
        ]
    )
    return {"analyses": [f"**{state['topic']}**: {response.content}"]}


# --- 4. Define the aggregation node ---
def synthesize_node(state: OverallState) -> dict:
    """Combine all analyses into a final report."""
    all_analyses = "\n\n".join(state["analyses"])
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a senior analyst. Synthesize the following individual analyses into a cohesive 3-4 sentence summary report."
            ),
            HumanMessage(content=f"Individual analyses:\n\n{all_analyses}"),
        ]
    )
    return {"final_report": response.content}


# --- 5. Build the graph ---
builder = StateGraph(OverallState)

builder.add_node("analyze", analyze_node)
builder.add_node("synthesize", synthesize_node)

# Fan-out: START -> conditional edges using Send
builder.add_conditional_edges(START, fan_out_to_workers, ["analyze"])
# Fan-in: all workers -> synthesize
builder.add_edge("analyze", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()

# --- 6. Run the map-reduce pipeline ---
print("=== Map-Reduce with Send API ===\n")

topics = [
    "Quantum Computing",
    "Renewable Energy",
    "Space Exploration",
    "Artificial General Intelligence",
]

print(f"Analyzing {len(topics)} topics in parallel...\n")

result = graph.invoke(
    {
        "topics": topics,
        "analyses": [],
        "final_report": "",
    }
)

print("--- Individual Analyses ---\n")
for analysis in result["analyses"]:
    print(f"{analysis}\n")

print("--- Synthesized Report ---\n")
print(result["final_report"])
